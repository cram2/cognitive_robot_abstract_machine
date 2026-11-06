"""
Keep shared installation independent of checkout directory names.
"""

from __future__ import annotations

# %% Imports
from typing_extensions import TYPE_CHECKING

import pytest

from scripts.cramployer.installation import Installation
from scripts.cramployer.vocabulary import DependencySource, OptionalPackage

from .dataset.paths import FixturePath

if TYPE_CHECKING:
    from pathlib import Path


# %% Checkout locations
@pytest.mark.parametrize(
    "checkout_name", [str(FixturePath.CHECKOUT), "checkout with spaces"]
)
def test_installation_accepts_arbitrary_checkout_names(checkout: Path) -> None:
    """
    Install directly in checkouts including paths requiring shell quoting.
    """
    Installation(checkout, DependencySource.PROCESSOR_ONLY).install()
    assert (checkout / FixturePath.FIRST_OUTPUT).read_text().splitlines() == [
        str(checkout),
        DependencySource.PROCESSOR_ONLY,
        OptionalPackage.ROBOKUDO,
    ]


def test_installation_preserves_unrelated_sibling_directory(checkout: Path) -> None:
    """
    Leave a sibling named like the CI checkout untouched.
    """
    sibling = checkout.parent / "cognitive_robot_abstract_machine"
    sibling.mkdir()
    Installation(checkout, DependencySource.DEFAULT).install()
    assert list(sibling.iterdir()) == []
