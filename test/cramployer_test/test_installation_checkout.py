"""
Keep installation repeatable across checkout locations.
"""

from __future__ import annotations

# %% Imports
from typing_extensions import TYPE_CHECKING

import pytest
import yaml

from scripts.cramployer.installation import Installation
from scripts.cramployer.vocabulary import FileName, DependencySource
from scripts.cramployer.workflow import Workflow

from .dataset.paths import FixturePath

if TYPE_CHECKING:
    from pathlib import Path


# %% Checkout aliases
def test_installation_reuses_existing_checkout_alias(checkout: Path) -> None:
    """
    Allow subsequent installation without replacing the checkout alias.
    """
    alias = checkout.parent / FileName.CHECKOUT
    alias.symlink_to(checkout, target_is_directory=True)
    Installation(checkout, DependencySource.PROCESSOR_ONLY).install()
    assert alias.resolve() == checkout


@pytest.mark.parametrize("checkout_name", [FileName.CHECKOUT])
def test_installation_accepts_checkout_with_workflow_directory_name(
    checkout: Path,
) -> None:
    """
    Use the checkout directly when its directory already matches the workflow.
    """
    Installation(checkout, DependencySource.PROCESSOR_ONLY).install()
    step = Workflow.from_json(
        yaml.safe_load(FixturePath.INSTALLATION.read_text())
    ).steps[0]
    assert (checkout / FixturePath.FIRST_OUTPUT).read_text() == next(
        iter(step.environment.values())
    )
