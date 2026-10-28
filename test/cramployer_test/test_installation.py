"""
Check shared installation commands without installing packages or contacting services.
"""

from __future__ import annotations

# %% Imports
import shlex
import subprocess

from typing_extensions import TYPE_CHECKING

import pytest

from scripts.cramployer.installation import Installation
from scripts.cramployer.exceptions import (
    UnsupportedPackages,
    UnknownWorkspaceExtra,
)
from scripts.cramployer.vocabulary import (
    CommandArgument,
    OptionalPackage,
    Program,
    DependencySource,
    FileName,
)

from .dataset.paths import FixturePath
from scripts.cramployer.launcher_vocabulary import ExitStatus

if TYPE_CHECKING:
    from pathlib import Path


# %% Installation commands
@pytest.mark.parametrize("mode", list(DependencySource))
def test_installation_calls_shared_script(
    repository: Path, mode: DependencySource
) -> None:
    """
    Pass dependency selections to the checkout's shared installer.
    """
    assert shlex.split(Installation(repository, mode).build_commands()) == [
        Program.BASH,
        str(repository / FileName.DEPENDENCY_INSTALLER),
        mode,
        OptionalPackage.ROBOKUDO,
    ]


def test_missing_shared_installer_fails_before_installation(tmp_path: Path) -> None:
    """
    Reject a checkout without the shared dependency installer.
    """
    with pytest.raises(FileNotFoundError):
        Installation(tmp_path, DependencySource.PROCESSOR_ONLY).build_commands()


def test_installation_blocks_have_valid_shell_syntax(repository: Path) -> None:
    """
    Check the generated installation commands for valid shell syntax.
    """
    result = subprocess.run(
        [Program.BASH, CommandArgument.SYNTAX_CHECK],
        input=Installation(repository, DependencySource.DEFAULT).build_commands(),
        text=True,
        check=False,
    )
    assert result.returncode == ExitStatus.SUCCESS


def test_installation_runs_in_the_selected_checkout(checkout: Path) -> None:
    """
    Run shared dependency installation without a checkout alias.
    """
    Installation(checkout, DependencySource.PROCESSOR_ONLY).install()
    assert (checkout / FixturePath.FIRST_OUTPUT).read_text().splitlines() == [
        str(checkout),
        DependencySource.PROCESSOR_ONLY,
        OptionalPackage.ROBOKUDO,
    ]


# %% Optional package features
@pytest.mark.parametrize(
    "packages", [(), (OptionalPackage.SEMANTIC_DIGITAL_TWIN,), tuple(OptionalPackage)]
)
def test_installation_passes_optional_packages(
    repository: Path, packages: tuple[str, ...]
) -> None:
    """
    Pass selected optional packages to the shared script in order.
    """
    assert shlex.split(
        Installation(
            repository, DependencySource.DEFAULT, additional_packages=packages
        ).build_commands()
    )[-1] == ",".join(packages)


def test_unknown_additional_package_is_rejected(repository: Path) -> None:
    """
    Reject unsupported optional dependency selections.
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
    assert commands.splitlines()[-1] == (
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
