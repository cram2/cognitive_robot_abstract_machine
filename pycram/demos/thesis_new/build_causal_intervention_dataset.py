"""
Build the paired causal intervention dataset from PyCharm.

Run this after run_causal_intervention_experiment.py finished. It reads the raw
cutting results, joins them with the intervention manifest, and writes the
paired causal dataset plus pairwise treatment effects.
"""

import json

from src.causal_intervention_experiment import (
    PAIRED_DATASET_CSV,
    PAIRWISE_EFFECTS_CSV,
    SUMMARY_JSON,
    build_paired_dataset,
)


def main() -> None:
    _, summary = build_paired_dataset()
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote paired interventional dataset: {PAIRED_DATASET_CSV}")
    print(f"Wrote paired causal effects: {PAIRWISE_EFFECTS_CSV}")
    print(f"Wrote summary: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
