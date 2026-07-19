# Music Recommendation Baseline

This repository measures how well a music recommender works for a new user when
only five of their listened-to tracks are known. The baseline uses collaborative
filtering with Alternating Least Squares (ALS).

**Reported result:** `0.5282 NDCG@10` across 1,000 evaluation users.

## Read this first

1. Read [REPORT.md](REPORT.md) for the problem, method, charts, and conclusion.
2. Read [src/config.py](src/config.py) for every experiment setting.
3. Follow the code in execution order:
   [src/cli.py](src/cli.py) → [src/data.py](src/data.py) →
   [src/experiment.py](src/experiment.py).

You do not need recommender-system experience before reading the report.

## Setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the result

Run the complete pipeline:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.cli all
```

This command performs three steps:

1. downloads the two required Music4All files from Zenodo and verifies them;
2. filters them into the fixed experiment dataset;
3. trains and evaluates the recommender and writes `results/`.

The source interaction archive is about 550 MB, and preparation scans its 50
million rows three times. Expect this command to take much longer than model
training. The download is public and does not require an account.

Tiny numeric differences can occur across operating systems or math libraries.

Run the metric tests with:

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
```

## Repository map

```text
.
├── README.md              # Setup, reading order, and commands
├── REPORT.md              # Explanation and reported findings
├── requirements.txt       # Python dependencies
├── src/
│   ├── config.py          # All fixed experiment settings
│   ├── data.py            # Download, filtering, and user selection
│   ├── experiment.py      # ALS training, ranking, metrics, and plots
│   └── cli.py             # Commands that connect the steps
├── data/README.md         # Data files and storage notes
├── results/               # Summary tables and figures used by the report
└── tests/                 # Checks for the ranking metrics
```

Downloaded and prepared data are deliberately excluded so the repository stays
small enough to share.

## Commands

| Command | What it does |
|---|---|
| `python -m src.cli download` | Downloads only the two original files this project uses. |
| `python -m src.cli prepare` | Builds the fixed catalog and user groups from downloaded data. |
| `python -m src.cli experiment` | Trains and evaluates ALS from prepared data. |
| `python -m src.cli all` | Runs all three steps in order. |

The repository contains only the baseline recommender. It has no data
augmentation implementation.
