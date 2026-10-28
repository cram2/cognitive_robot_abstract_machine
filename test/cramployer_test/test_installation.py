"""
Check CI command reuse without installing packages or contacting services.
"""

from __future__ import annotations

# %% Imports
import subprocess

from typing_extensions import TYPE_CHECKING

import pytest
import yaml

from scripts.cramployer.installation import Installation
from scripts.cramployer.exceptions import (
    MissingWorkflowStep,
    UnsupportedPackages,
    UnknownWorkspaceExtra,
)
from scripts.cramployer.vocabulary import (
    CommandArgument,
    OptionalPackage,
    Program,
    ShellSyntax,
    StepTitle,
    DependencySource,
)
from scripts.cramployer.workflow import Workflow

from .dataset.paths import FixturePath
from scripts.cramployer.launcher_vocabulary import ExitStatus

if TYPE_CHECKING:
    from pathlib import Path


# %% Installation commands
def test_processor_installation_preserves_current_workflow_commands(
    repository: Path,
) -> None:
    """
    Keep both installation blocks unchanged in CPU mode.
    """
    workflow = Workflow.from_json(
        yaml.safe_load((repository / FixturePath.WORKFLOW).read_text())
    )
    selected = [
        step.shell_commands
        for step in workflow.steps
        if step.title in (StepTitle.WORKSPACE, StepTitle.ROBOKUDO)
    ]
    assert Installation(
        repository, DependencySource.PROCESSOR_ONLY
    ).build_commands() == "\n".join("(\n" + command + "\n)" for command in selected)


def test_default_installation_removes_processor_index(repository: Path) -> None:
    """
    Allow the package index to supply GPU-capable PyTorch wheels.
    """
    assert Installation(
        repository, DependencySource.DEFAULT
    ).build_commands() == Installation(
        repository, DependencySource.PROCESSOR_ONLY
    ).build_commands().replace(
        ShellSyntax.PROCESSOR_INDEX, ""
    )


def test_missing_workflow_step_fails_before_installation(tmp_path: Path) -> None:
    """
    Require review when the workflow no longer exposes the selected steps.
    """
    workflow_directory = (tmp_path / FixturePath.WORKFLOW).parent
    workflow_directory.mkdir(parents=True)
    (workflow_directory / FixturePath.WORKFLOW.name).write_text(
        FixturePath.MISSING_STEP.read_text()
    )
    with pytest.raises(MissingWorkflowStep):
        Installation(tmp_path, DependencySource.PROCESSOR_ONLY).build_commands()


def test_installation_blocks_have_valid_shell_syntax(repository: Path) -> None:
    """
    Check generated subshell boundaries against the actual workflow.
    """
    result = subprocess.run(
        [Program.BASH, CommandArgument.SYNTAX_CHECK],
        input=Installation(repository, DependencySource.DEFAULT).build_commands(),
        text=True,
        check=False,
    )
    assert result.returncode == ExitStatus.SUCCESS


def test_workflow_steps_start_from_the_checkout_parent(checkout: Path) -> None:
    """
    Resolve each CI checkout command independently of the previous step.
    """
    Installation(checkout, DependencySource.PROCESSOR_ONLY).install()
    workflow = Workflow.from_json(yaml.safe_load(FixturePath.INSTALLATION.read_text()))
    outputs = {
        FixturePath.FIRST_OUTPUT: StepTitle.WORKSPACE,
        FixturePath.SECOND_OUTPUT: StepTitle.ROBOKUDO,
    }
    for output_path, step_title in outputs.items():
        step = next(step for step in workflow.steps if step.title == step_title)
        assert (checkout / output_path).read_text() == next(
            iter(step.environment.values())
        )


# %% Optional package features
def test_common_installation_can_skip_package_specific_steps(repository: Path) -> None:
    """
    Prepare all workspace members without selecting optional CI packages.
    """
    workflow = Workflow.from_json(
        yaml.safe_load((repository / FixturePath.WORKFLOW).read_text())
    )
    common = next(
        step.shell_commands
        for step in workflow.steps
        if step.title == StepTitle.WORKSPACE
    )
    assert (
        Installation(
            repository, DependencySource.PROCESSOR_ONLY, additional_packages=()
        ).build_commands()
        == "(\n" + common + "\n)"
    )


def test_semantic_digital_twin_selects_its_installationation(repository: Path) -> None:
    """
    Reuse Drake's installation commands from the current CI workflow.
    """
    workflow = Workflow.from_json(
        yaml.safe_load((repository / FixturePath.WORKFLOW).read_text())
    )
    expected = [
        step.shell_commands
        for step in workflow.steps
        if step.title in (StepTitle.WORKSPACE, StepTitle.DRAKE)
    ]
    assert Installation(
        repository,
        DependencySource.PROCESSOR_ONLY,
        additional_packages=(OptionalPackage.SEMANTIC_DIGITAL_TWIN,),
    ).build_commands() == "\n".join("(\n" + command + "\n)" for command in expected)


def test_unknown_additional_package_is_rejected(repository: Path) -> None:
    """
    Reject a package selection without a supported optional CI step.
    """
    with pytest.raises(UnsupportedPackages):
        Installation(
            repository,
            DependencySource.PROCESSOR_ONLY,
            additional_packages=("unknown-package",),
        ).build_commands()


def test_workspace_extra_selects_the_package_extra(repository: Path) -> None:
    """
    Add the requested optional feature while retaining installed packages.
    """
    commands = Installation(
        repository,
        DependencySource.PROCESSOR_ONLY,
        workspace_extras=("semantic_digital_twin:iris",),
    ).build_commands()
    assert commands.splitlines()[-2] == (
        " ".join(
            [
                Program.UV,
                CommandArgument.SYNC,
                CommandArgument.ACTIVE,
                CommandArgument.INEXACT,
                CommandArgument.PACKAGE,
                OptionalPackage.SEMANTIC_DIGITAL_TWIN,
                CommandArgument.EXTRA,
                "iris",
            ]
        )
    )


def test_unknown_workspace_extra_is_rejected(repository: Path) -> None:
    """
    Validate optional features against the selected checkout before installing.
    """
    with pytest.raises(UnknownWorkspaceExtra):
        Installation(
            repository,
            DependencySource.PROCESSOR_ONLY,
            workspace_extras=("semantic_digital_twin:unknown-extra",),
        ).build_commands()
