"""
Tests that every krrood distribution loads with only the distributions it declares, so a
plain install of it works on its own.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import krrood.adapters
from krrood.adapters.json_serializer import PurePathJSONSerializer
from .dataset.declared_dependencies import (
    CheckoutDistribution,
    UndeclaredModules,
    checkout_distributions,
)

DATASET_DIRECTORY = Path(__file__).parent / "dataset"
"""
The directory holding the scripts these tests run in a fresh interpreter.
"""


def run_with_blocked_modules(
    *, distribution: CheckoutDistribution, target: list[str]
) -> subprocess.CompletedProcess:
    """
    Run the target in a fresh interpreter in which every module the distribution does
    not declare is blocked.

    :param target: The ``--modules`` or ``--script`` arguments of the blocking script.
    """
    blocked = UndeclaredModules(
        distribution=distribution, checkout=checkout_distributions()
    ).module_names
    return subprocess.run(
        [
            sys.executable,
            "-I",
            str(DATASET_DIRECTORY / "import_with_blocked_modules.py"),
            "--blocked",
            *blocked,
            *target,
        ],
        capture_output=True,
        text=True,
    )


# %% loading every module of a distribution


@pytest.mark.parametrize(
    "distribution",
    checkout_distributions(),
    ids=lambda distribution: distribution.distribution_name,
)
def test_every_module_loads_with_only_declared_dependencies(
    distribution: CheckoutDistribution,
) -> None:
    """
    Every module of the distribution imports when every distribution it does not
    declare, directly or through what it requires, is blocked.
    """
    loading = run_with_blocked_modules(
        distribution=distribution,
        target=["--modules", *distribution.module_names],
    )

    assert loading.returncode == 0, loading.stderr


# %% finding the serializer of a subclass


def test_finding_the_serializer_of_a_subclass_needs_only_declared_dependencies() -> (
    None
):
    """
    The serializer of the nearest serialized superclass is found with only what the
    serialization distribution declares.
    """
    serialization = next(
        distribution
        for distribution in checkout_distributions()
        if krrood.adapters.__name__ in distribution.subpackage_names
    )

    lookup = run_with_blocked_modules(
        distribution=serialization,
        target=["--script", str(DATASET_DIRECTORY / "subclass_serializer_lookup.py")],
    )

    assert lookup.returncode == 0, lookup.stderr
    assert lookup.stdout.strip() == PurePathJSONSerializer.__qualname__
