from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PREPARED_DIR = ROOT / "data" / "prepared"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


@dataclass(frozen=True)
class Config:
    random_seed: int = 11
    min_play_count: int = 2
    catalog_size: int = 10_000
    warm_users: int = 5_000
    cold_users: int = 1_000
    min_warm_interactions: int = 5
    min_cold_interactions: int = 30
    max_user_interactions: int = 500
    seed_tracks: int = 5
    probe_tracks: int = 10
    als_factors: int = 64
    als_regularization: float = 0.05
    als_iterations: int = 20
    real_confidence_alpha: float = 10.0
    recommendation_k: int = 10
    bootstrap_resamples: int = 5_000
    chunk_size: int = 2_000_000


CONFIG = Config()
