"""
Parse installer selections without an installed workspace.
"""

from __future__ import annotations

# %% Imports
import subprocess
import sys

from typing_extensions import TYPE_CHECKING

from scripts.cramployer.installation import Installation
from scripts.cramployer.vocabulary import (
    CommandArgument,
    FileName,
    OptionalPackage,
    DependencySource,
)

from .dataset.paths import FixturePath

if TYPE_CHECKING:
    from pathlib import Path


# %% Argument defaults
def test_argument_defaults_select_optional_vision_dependencies(tmp_path: Path) -> None:
    """
    Default to the supported optional vision installation without workspace extras.
    """
    installation = Installation.from_arguments(
        [str(tmp_path), DependencySource.PROCESSOR_ONLY]
    )
    assert installation == Installation(tmp_path, DependencySource.PROCESSOR_ONLY)


def test_argument_selections_preserve_package_order(tmp_path: Path) -> None:
    """
    Parse comma-separated package and extra selections without empty entries.
    """
    packages = (OptionalPackage.ROBOKUDO, OptionalPackage.SEMANTIC_DIGITAL_TWIN)
    extras = ("semantic_digital_twin:iris",)
    installation = Installation.from_arguments(
        [
            str(tmp_path),
            DependencySource.DEFAULT,
            ",".join(packages) + ",",
            ",".join(extras) + ",",
        ]
    )
    assert installation == Installation(
        tmp_path,
        DependencySource.DEFAULT,
        additional_packages=packages,
        workspace_extras=extras,
    )


# %% Standalone entry point
def test_installer_entry_point_runs_fixture_commands(checkout: Path) -> None:
    """
    Execute local workflow commands through the module entry point.
    """
    subprocess.run(
        [
            sys.executable,
            CommandArgument.MODULE,
            Installation.__module__,
            str(checkout),
            DependencySource.PROCESSOR_ONLY,
        ],
        cwd=FixturePath.REPOSITORY,
        check=True,
    )
    assert (checkout.parent / FileName.CHECKOUT).resolve() == checkout
