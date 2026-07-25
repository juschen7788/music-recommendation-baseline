"""Prepare the Music4All data used by the baseline experiment."""

from __future__ import annotations

import bz2
import hashlib
import json
import shutil
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CONFIG, FIGURES_DIR, PREPARED_DIR, RAW_DIR, ROOT, Config


INTERACTION_FILE = RAW_DIR / "userid_trackid_count.tsv.bz2"
GENRE_FILE = RAW_DIR / "id_genres_tf-idf.tsv.bz2"
ZENODO_RECORD = "15394646"
REQUIRED_FILES = {
    INTERACTION_FILE.name: "314b51196a9c8f333c7fefc0711760a1",
    GENRE_FILE.name: "a742b5fa1d2e2ce780101773e57bb7f5",
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download() -> dict:
    """Download and verify only the two source files used by this project."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for filename, expected_md5 in REQUIRED_FILES.items():
        destination = RAW_DIR / filename
        if not destination.exists() or _md5(destination) != expected_md5:
            temporary = destination.with_suffix(destination.suffix + ".part")
            url = (
                f"https://zenodo.org/records/{ZENODO_RECORD}/files/"
                f"{filename}?download=1"
            )
            print(f"Downloading {filename}...")
            try:
                with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                if _md5(temporary) != expected_md5:
                    raise ValueError(f"Checksum failed for {filename}")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            print(f"Using existing {filename}")
        downloaded.append(str(destination.relative_to(ROOT)))
    return {"zenodo_record": ZENODO_RECORD, "files": downloaded}


def _interaction_chunks(config: Config) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(
        INTERACTION_FILE,
        sep="\t",
        compression="bz2",
        chunksize=config.chunk_size,
        dtype={"user_id": "int32", "track_id": "string", "count": "int32"},
    )


def _tracks_with_genres(path: Path) -> set[str]:
    ids: set[str] = set()
    with bz2.open(path, "rt", encoding="utf-8") as stream:
        next(stream)
        for line in stream:
            ids.add(line.partition("\t")[0])
    return ids


def _plot_dataset_distributions(item_listeners: dict[str, int], user_activity: dict[int, int]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    item_values = np.asarray(list(item_listeners.values()))
    user_values = np.asarray(list(user_activity.values()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(np.log10(item_values + 1), bins=45, color="#35618f")
    axes[0].set(
        title="How popular are the 10,000 catalog tracks?",
        xlabel="Retained listeners per track (base-10 log scale)",
        ylabel="Number of tracks",
    )
    axes[1].hist(np.clip(user_values, 0, 500), bins=45, color="#a4553c")
    axes[1].set(
        title="How broadly does each user listen?",
        xlabel="Distinct retained catalog tracks per user (maximum 500)",
        ylabel="Number of users",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dataset_distributions.png", dpi=160)
    plt.close(fig)


def prepare(config: Config = CONFIG) -> dict:
    for path in (INTERACTION_FILE, GENRE_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Required dataset file is missing: {path}. "
                "Run `python -m src.cli download` first."
            )
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)

    genre_ids = _tracks_with_genres(GENRE_FILE)
    item_listeners: Counter[str] = Counter()
    all_users: set[int] = set()
    total_rows = 0
    retained_rows = 0

    for chunk in _interaction_chunks(config):
        total_rows += len(chunk)
        all_users.update(chunk["user_id"].unique().tolist())
        retained = chunk.loc[
            (chunk["count"] >= config.min_play_count) & chunk["track_id"].isin(genre_ids),
            "track_id",
        ]
        retained_rows += len(retained)
        item_listeners.update(retained.value_counts().to_dict())

    ranked_items = sorted(item_listeners.items(), key=lambda pair: (-pair[1], pair[0]))
    catalog = [track_id for track_id, _ in ranked_items[: config.catalog_size]]
    if len(catalog) < config.catalog_size:
        raise ValueError(f"Only {len(catalog)} eligible catalog tracks; requested {config.catalog_size}")
    catalog_set = set(catalog)

    user_activity: Counter[int] = Counter()
    for chunk in _interaction_chunks(config):
        retained = chunk.loc[
            (chunk["count"] >= config.min_play_count) & chunk["track_id"].isin(catalog_set),
            "user_id",
        ]
        user_activity.update(retained.value_counts().to_dict())

    rng = np.random.default_rng(config.random_seed)
    cold_candidates = np.asarray(
        sorted(
            user for user, size in user_activity.items()
            if config.min_cold_interactions <= size <= config.max_user_interactions
        ),
        dtype=np.int32,
    )
    if len(cold_candidates) < config.cold_users:
        raise ValueError(
            f"Only {len(cold_candidates)} cold-user candidates; "
            f"requested {config.cold_users}"
        )
    cold_users = rng.choice(cold_candidates, size=config.cold_users, replace=False)
    cold_set = set(cold_users.tolist())
    warm_candidates = np.asarray(
        sorted(
            user for user, size in user_activity.items()
            if user not in cold_set
            and config.min_warm_interactions <= size <= config.max_user_interactions
        ),
        dtype=np.int32,
    )
    if len(warm_candidates) < config.warm_users:
        raise ValueError(
            f"Only {len(warm_candidates)} warm-user candidates; "
            f"requested {config.warm_users}"
        )
    warm_users = rng.choice(warm_candidates, size=config.warm_users, replace=False)
    selected_users = set(warm_users.tolist()) | cold_set

    selected_chunks: list[pd.DataFrame] = []
    for chunk in _interaction_chunks(config):
        retained = chunk.loc[
            (chunk["count"] >= config.min_play_count)
            & chunk["track_id"].isin(catalog_set)
            & chunk["user_id"].isin(selected_users)
        ].copy()
        if not retained.empty:
            selected_chunks.append(retained)
    interactions = pd.concat(selected_chunks, ignore_index=True)
    interactions["cohort"] = np.where(interactions["user_id"].isin(cold_set), "cold", "warm")
    interactions.to_parquet(PREPARED_DIR / "interactions.parquet", index=False)

    assignments = {
        "warm_users": sorted(map(int, warm_users)),
        "cold_users": sorted(map(int, cold_users)),
    }
    _write_json(PREPARED_DIR / "catalog.json", catalog)
    _write_json(PREPARED_DIR / "user_assignments.json", assignments)

    summary = {
        "raw": {
            "interaction_rows": int(total_rows),
            "users": len(all_users),
            "interaction_tracks_documented": 56_512,
            "genre_tracks": len(genre_ids),
        },
        "prepared": {
            "retained_rows_with_genres": int(retained_rows),
            "catalog_tracks": len(catalog),
            "selected_interactions": len(interactions),
            "warm_users": len(warm_users),
            "cold_users": len(cold_users),
            "cold_candidate_users": len(cold_candidates),
            "warm_candidate_users": len(warm_candidates),
        },
    }
    _write_json(PREPARED_DIR / "dataset_summary.json", summary)
    _plot_dataset_distributions(
        {track_id: item_listeners[track_id] for track_id in catalog}, dict(user_activity)
    )
    return summary
