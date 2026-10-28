"""Reject installation selections that cannot run independently of GitHub Actions."""

from __future__ import annotations

# %% Imports
from enum import StrEnum

from typing_extensions import TYPE_CHECKING

import pytest

from scripts.cramployer.installation import Installation
from scripts.cramployer.exceptions import (
    CheckoutConflict,
    InvalidDependencySource,
    InvalidWorkflowStep,
    MissingWorkflowStep,
    UnknownWorkspacePackage,
)
from scripts.cramployer.vocabulary import FileName, OptionalPackage, DependencySource

from .dataset.paths import FixturePath

if TYPE_CHECKING:
    from pathlib import Path


# %% Invalid workflow inputs
class WorkflowFile(StrEnum):
    """Workflow fixtures containing unsupported installation steps."""

    COMMAND_EXPRESSION = "unsupported-command.yaml"
    """Commands depending on GitHub expression evaluation."""
    ENVIRONMENT_EXPRESSION = "unsupported-environment.yaml"
    """Environment depending on GitHub expression evaluation."""
    NONSTRING_ENVIRONMENT = "nonstring-environment.yaml"
    """Environment values not supplied as strings."""
    MISSING_COMMANDS = "missing-commands.yaml"
    """Action without installation shell commands."""
    DUPLICATE_STEP = "duplicate-step.yaml"
    """Two indistinguishable installation steps."""


# %% Workflow validation
@pytest.mark.parametrize(
    "workflow_file",
    [member for member in WorkflowFile if member != WorkflowFile.DUPLICATE_STEP],
)
def test_workflow_rejects_steps_requiring_review(
    tmp_path: Path, workflow_file: WorkflowFile
) -> None:
    """Reject unsupported commands or environment values before installation."""
    installation = Installation(
        tmp_path,
        DependencySource.PROCESSOR_ONLY,
        additional_packages=(),
        workflow_path=FixturePath.DATASET / workflow_file,
    )
    with pytest.raises(InvalidWorkflowStep):
        installation.build_commands()


def test_workflow_rejects_duplicate_installation_steps(tmp_path: Path) -> None:
    """Require one unambiguous common installation step."""
    installation = Installation(
        tmp_path,
        DependencySource.PROCESSOR_ONLY,
        additional_packages=(),
        workflow_path=FixturePath.DATASET / WorkflowFile.DUPLICATE_STEP,
    )
    with pytest.raises(MissingWorkflowStep):
        installation.build_commands()


# %% Dependency selection
@pytest.mark.parametrize("selection", ["missing:feature", OptionalPackage.ROBOKUDO])
def test_workspace_extra_requires_a_workspace_package(
    repository: Path, selection: str
) -> None:
    """Reject unknown packages and selections without an extra separator."""
    with pytest.raises(UnknownWorkspacePackage):
        Installation(
            repository, DependencySource.PROCESSOR_ONLY, workspace_extras=(selection,)
        ).build_commands()


def test_unknown_dependency_source_is_rejected(repository: Path) -> None:
    """Reject dependency sources outside the supported modes."""
    with pytest.raises(InvalidDependencySource):
        Installation(repository, "unsupported").build_commands()


# %% Checkout protection
@pytest.mark.parametrize("broken_symlink", [False, True])
def test_installation_preserves_conflicting_checkout_paths(
    tmp_path: Path, broken_symlink: bool
) -> None:
    """Reject existing checkout paths that belong to another installation."""
    repository = tmp_path / FixturePath.CHECKOUT
    repository.mkdir()
    checkout = tmp_path / FileName.CHECKOUT
    if broken_symlink:
        checkout.symlink_to(tmp_path / "missing", target_is_directory=True)
    else:
        checkout.mkdir()
    expected_destination = checkout.resolve()
    installation = Installation(repository, DependencySource.PROCESSOR_ONLY)
    with pytest.raises(CheckoutConflict):
        installation.prepare_checkout()
    assert checkout.resolve() == expected_destination
