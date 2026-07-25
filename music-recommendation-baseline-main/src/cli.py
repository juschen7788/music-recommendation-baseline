"""Command-line entry point for data preparation and evaluation."""

from __future__ import annotations

import argparse
import json

from .data import download, prepare
from .experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare data or run the Music4All cold-start baseline."
    )
    parser.add_argument(
        "command",
        choices=("download", "prepare", "experiment", "all"),
        help="download data, prepare it, run the experiment, or run every step",
    )
    args = parser.parse_args()
    output = {}
    if args.command in ("download", "all"):
        output["download"] = download()
    if args.command in ("prepare", "all"):
        output["dataset"] = prepare()
    if args.command in ("experiment", "all"):
        output["experiment"] = run_experiment()
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
