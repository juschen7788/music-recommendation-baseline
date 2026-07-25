"""Cold-start baseline using one warm-trained implicit-feedback ALS model."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy import sparse
from threadpoolctl import threadpool_limits

from .config import CONFIG, FIGURES_DIR, PREPARED_DIR, RESULTS_DIR, Config


def ndcg_at_k(ranked: np.ndarray, relevant: set[int], k: int) -> float:
    hits = np.fromiter((int(item in relevant) for item in ranked[:k]), dtype=np.float64)
    dcg = float(np.sum(hits / np.log2(np.arange(2, len(hits) + 2))))
    ideal_hits = min(k, len(relevant))
    if ideal_hits == 0:
        return 0.0
    ideal = float(np.sum(1.0 / np.log2(np.arange(2, ideal_hits + 2))))
    return dcg / ideal


def recall_at_k(ranked: np.ndarray, relevant: set[int], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0

def precision_at_k(ranked: np.ndarray, relevant: set[int], k: int) -> float:
    if len(ranked) == 0 or k <= 0:
        return 0.0
    top_k = ranked[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return float(hits / k)

def popularity_at_k(global_popular_items: np.ndarray, relevant: set[int], k: int) -> tuple[float, float, float, float]:
    ranked = global_popular_items[:k]
    hits = len(set(ranked) & relevant)
    
    prec = hits / k if k > 0 else 0.0
    rec = hits / len(relevant) if relevant else 0.0
    hr = 1.0 if hits > 0 else 0.0
    
    hit_array = np.fromiter((int(item in relevant) for item in ranked), dtype=np.float64)
    dcg = float(np.sum(hit_array / np.log2(np.arange(2, len(hit_array) + 2))))
    ideal_hits = min(k, len(relevant))
    ideal = float(np.sum(1.0 / np.log2(np.arange(2, ideal_hits + 2)))) if ideal_hits > 0 else 1.0
    ndcg = dcg / ideal if ideal > 0 else 0.0
    
    return ndcg, rec, prec, hr


def _top_k(scores: np.ndarray, excluded: set[int], k: int) -> np.ndarray:
    safe = scores.copy()
    if excluded:
        safe[np.fromiter(excluded, dtype=np.int32)] = -np.inf
    candidates = np.argpartition(safe, -k)[-k:]
    return candidates[np.argsort(safe[candidates])[::-1]]


def _confidence(counts: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 + alpha * np.log1p(counts)).astype(np.float32)


def _user_row(items: np.ndarray, values: np.ndarray, n_items: int) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (values.astype(np.float32), (np.zeros(len(items), dtype=np.int32), items)),
        shape=(1, n_items),
    )


def _bootstrap_interval(values: np.ndarray, rng: np.random.Generator, resamples: int) -> list[float]:
    means = np.empty(resamples, dtype=np.float64)
    batch = 250
    for start in range(0, resamples, batch):
        size = min(batch, resamples - start)
        samples = rng.choice(values, size=(size, len(values)), replace=True)
        means[start : start + size] = samples.mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def _summarize(metrics: pd.DataFrame, config: Config) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_seed + 99)
    rows = []
    for condition, frame in metrics.groupby("condition", sort=False):
        for metric in ("ndcg_at_10", "recall_at_10", "precision_at_10", "hit_rate_at_10"):
            values = frame[metric].to_numpy(dtype=float)
            low, high = _bootstrap_interval(values, rng, config.bootstrap_resamples)
            rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "mean": values.mean(),
                    "median": np.median(values),
                    "std": values.std(ddof=1),
                    "ci_95_low": low,
                    "ci_95_high": high,
                    "users": len(values),
                }
            )
    return pd.DataFrame(rows)


def _plot_results(metrics: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    order = ["global_popularity_baseline", "five_seed_baseline", "fifteen_real_interaction_reference"]
    labels = ["Popularity", "5-Seed Baseline", "15-Interaction Ref"]
    
    grouped = metrics.groupby("condition")
    ndcg = grouped["ndcg_at_10"].mean().reindex(order)
    recall = grouped["recall_at_10"].mean().reindex(order)
    hit_rate = grouped["hit_rate_at_10"].mean().reindex(order)
    
    baseline = metrics.loc[
        metrics.condition == "five_seed_baseline"
    ].set_index("user_id")["ndcg_at_10"]
    reference = metrics.loc[
        metrics.condition == "fifteen_real_interaction_reference"
    ].set_index("user_id")["ndcg_at_10"]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    x = np.arange(3)
    width = 0.25
    
    pop_means = [ndcg.iloc[0], recall.iloc[0], hit_rate.iloc[0]]
    baseline_means = [ndcg.iloc[1], recall.iloc[1], hit_rate.iloc[1]]
    reference_means = [ndcg.iloc[2], recall.iloc[2], hit_rate.iloc[2]]
    
    axes[0].bar(x - width, pop_means, width, label=labels[0], color="#888888")
    axes[0].bar(x, baseline_means, width, label=labels[1], color="#35618f")
    axes[0].bar(x + width, reference_means, width, label=labels[2], color="#7b4ab5")
    
    axes[0].set_xticks(x, ["NDCG@10", "Recall@10", "Hit Rate@10"])
    axes[0].set(
        title="Average quality across baseline conditions",
        ylabel="Average score across 1,000 users",
        ylim=(0, 1.05),
    )
    axes[0].legend()
    
    axes[1].hist(baseline, bins=35, color="#35618f")
    axes[1].axvline(
        baseline.mean(),
        color="black",
        linestyle="--",
        label=f"mean = {baseline.mean():.3f}",
    )
    axes[1].set(
        title="How baseline NDCG@10 varies across users",
        xlabel="Individual user's 5-seed NDCG@10 (0 worst, 1 best)",
        ylabel="Number of users",
    )
    axes[1].legend()
    
    delta = reference - baseline
    axes[2].hist(delta, bins=35, color="#7b4ab5")
    axes[2].axvline(0, color="black", linewidth=1)
    axes[2].set(
        title="Effect of adding 10 more real interactions",
        xlabel="Change in NDCG@10 (negative = worse, positive = better)",
        ylabel="Number of users",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "baseline_results.png", dpi=160)
    plt.close(fig)


def run_experiment(config: Config = CONFIG) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    required = [
        PREPARED_DIR / "interactions.parquet",
        PREPARED_DIR / "catalog.json",
        PREPARED_DIR / "user_assignments.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Prepared data is missing. Run `python -m src.cli all` to download "
            "the source data, prepare it, and run the experiment."
        )
    interactions = pd.read_parquet(PREPARED_DIR / "interactions.parquet")
    catalog = json.loads(
        (PREPARED_DIR / "catalog.json").read_text(encoding="utf-8")
    )
    assignments = json.loads(
        (PREPARED_DIR / "user_assignments.json").read_text(encoding="utf-8")
    )
    item_index = {track_id: index for index, track_id in enumerate(catalog)}
    interactions["item_index"] = interactions["track_id"].map(item_index).astype("int32")
    warm = interactions.loc[interactions.cohort == "warm"].copy()
    cold = interactions.loc[interactions.cohort == "cold"].copy()

    warm_users = assignments["warm_users"]
    warm_index = {user: index for index, user in enumerate(warm_users)}
    warm["user_index"] = warm["user_id"].map(warm_index).astype("int32")
    warm_values = _confidence(warm["count"].to_numpy(), config.real_confidence_alpha)
    warm_matrix = sparse.csr_matrix(
        (warm_values, (warm["user_index"], warm["item_index"])),
        shape=(len(warm_users), len(catalog)),
    )

    # 1. Compute Top-10 Global Popularity baseline items
    global_item_counts = np.asarray(warm_matrix.sum(axis=0)).ravel()
    global_top_10 = np.argsort(global_item_counts)[::-1][:config.recommendation_k]

    model = AlternatingLeastSquares(
        factors=config.als_factors,
        regularization=config.als_regularization,
        alpha=1.0,
        iterations=config.als_iterations,
        random_state=config.random_seed,
    )
    with threadpool_limits(limits=1, user_api="blas"):
        model.fit(warm_matrix, show_progress=True)
    cold_groups = {int(user): frame for user, frame in cold.groupby("user_id")}
    rng = np.random.default_rng(config.random_seed)
    metric_rows: list[dict] = []

    for user_id in assignments["cold_users"]:
        frame = cold_groups[int(user_id)]
        items = frame["item_index"].to_numpy(dtype=np.int32)
        counts = frame["count"].to_numpy(dtype=np.float32)
        permutation = rng.permutation(len(items))
        seed_pos = permutation[: config.seed_tracks]
        probe_pos = permutation[config.seed_tracks : config.seed_tracks + config.probe_tracks]
        test_pos = permutation[config.seed_tracks + config.probe_tracks :]
        seed_items, seed_counts = items[seed_pos], counts[seed_pos]
        probe_items, probe_counts = items[probe_pos], counts[probe_pos]
        test_items = items[test_pos]

        relevant = set(map(int, test_items))
        excluded = set(map(int, seed_items)) | set(map(int, probe_items))

        # 2. Evaluate Global Popularity Baseline for this user
        pop_ndcg, pop_rec, pop_prec, pop_hr = popularity_at_k(global_top_10, relevant, config.recommendation_k)
        metric_rows.append(
            {
                "user_id": user_id,
                "condition": "global_popularity_baseline",
                "ndcg_at_10": pop_ndcg,
                "recall_at_10": pop_rec,
                "precision_at_10": pop_prec,
                "hit_rate_at_10": pop_hr,
            }
        )

        # 3. Evaluate ALS Conditions (5-Seed and 15-Interaction)
        als_conditions = {
            "five_seed_baseline": (seed_items, _confidence(seed_counts, config.real_confidence_alpha)),
            "fifteen_real_interaction_reference": (
                np.r_[seed_items, probe_items],
                np.r_[
                    _confidence(seed_counts, config.real_confidence_alpha),
                    _confidence(probe_counts, config.real_confidence_alpha),
                ],
            ),
        }

        for condition, (condition_items, condition_values) in als_conditions.items():
            row = _user_row(condition_items, condition_values, len(catalog))
            with threadpool_limits(limits=1, user_api="blas"):
                user_factor = model.recalculate_user(0, row)
                scores = model.item_factors @ user_factor
            ranked = _top_k(scores, excluded, config.recommendation_k)

            metric_rows.append(
                {
                    "user_id": user_id,
                    "condition": condition,
                    "ndcg_at_10": ndcg_at_k(ranked, relevant, config.recommendation_k),
                    "recall_at_10": recall_at_k(ranked, relevant, config.recommendation_k),
                    "precision_at_10": precision_at_k(ranked, relevant, config.recommendation_k),
                    "hit_rate_at_10": float(bool(set(ranked) & relevant)),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    summaries = _summarize(metrics, config)
    summaries.to_csv(RESULTS_DIR / "baseline_summary_metrics.csv", index=False)
    _plot_results(metrics)

    means = metrics.groupby("condition")[["ndcg_at_10", "recall_at_10", "precision_at_10", "hit_rate_at_10"]].mean()
    baseline_ndcg = float(means.loc["five_seed_baseline", "ndcg_at_10"])
    reference_ndcg = float(means.loc["fifteen_real_interaction_reference", "ndcg_at_10"])
    summary = {
        "system": "one implicit-feedback ALS recommender trained on warm users",
        "users_evaluated": len(assignments["cold_users"]),
        "catalog_tracks": len(catalog),
        "visible_seed_tracks_per_user": config.seed_tracks,
        "metrics": means.to_dict(orient="index"),
        "headroom": {
            "fifteen_real_reference_minus_baseline_ndcg_at_10": reference_ndcg - baseline_ndcg,
            "fifteen_real_reference_relative_improvement": (
                reference_ndcg - baseline_ndcg
            ) / baseline_ndcg,
        },
        "next_phase_success_goals": {
            "primary": "augmentation NDCG@10 must be higher than the five-seed baseline",
            "minimum_promising_relative_ndcg_gain": 0.05,
            "required_uncertainty_test": "paired 95% bootstrap CI for augmentation minus baseline must be above zero",
            "do_not_regress": ["recall_at_10", "hit_rate_at_10"],
            "upper_reference": "fifteen_real_interaction_reference",
        },
    }
    (RESULTS_DIR / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary