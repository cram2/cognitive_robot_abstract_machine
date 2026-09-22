"""
The knobs every experiment's run shares, as read from its command line.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.causal_reasoning.comparison.run import RunSettings


def test_the_shared_flags_are_read_back_into_settings():
    parser = argparse.ArgumentParser()
    RunSettings.add_arguments(parser, Path("results.md"))
    arguments = parser.parse_args(
        [
            "--seed",
            "3",
            "--orderings",
            "2",
            "--splits",
            "1",
            "--min-region-support",
            "4",
        ]
    )
    assert RunSettings.from_arguments(arguments) == RunSettings(
        output=Path("results.md"),
        seed=3,
        ordering_count=2,
        split_count=1,
        min_region_support=4,
    )


def test_the_defaults_leave_the_leaf_sizes_to_the_pipelines():
    parser = argparse.ArgumentParser()
    RunSettings.add_arguments(parser, Path("results.md"))
    settings = RunSettings.from_arguments(parser.parse_args([]))
    assert settings.min_samples_per_leaf is None
    assert settings.plain_min_samples_per_leaf is None
    assert settings.split_count == 0
