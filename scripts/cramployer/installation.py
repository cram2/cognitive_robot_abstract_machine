"""Install a checkout through the shared dependency script and selected extras."""

from __future__ import annotations

# %% Imports
import argparse
import shlex
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from typing_extensions import TYPE_CHECKING

from scripts.cramployer.exceptions import (
    InvalidDependencySource,
    UnsupportedPackages,
)
from scripts.cramployer.vocabulary import (
    CommandArgument,
    FileName,
    OptionalPackage,
    Program,
    DependencySource,
)
from scripts.cramployer.workspace import Workspace, WorkspaceFeature

if TYPE_CHECKING:
    from typing_extensions import Self, Sequence


# %% Command-line configuration
class ArgumentName(StrEnum):
    """Positional installation arguments."""

    REPOSITORY = "repository"
    """Checkout whose dependencies are installed."""
    DEPENDENCY_SOURCE = "dependency_source"
    """PyTorch dependency source."""
    ADDITIONAL_PACKAGES = "additional_packages"
    """Packages whose optional dependencies are installed."""
    WORKSPACE_EXTRAS = "workspace_extras"
    """Optional workspace dependency groups."""


class ArgumentCardinality(StrEnum):
    """Cardinality of optional positional command-line selections."""

    OPTIONAL = "?"
    """Accept zero or one positional value."""


# %% Dependency installation
@dataclass
class Installation:
    """Install shared workspace dependencies and selected optional features."""

    repository: Path
    """Checkout containing the shared dependency installation script."""
    dependency_source: str
    """Requested source for PyTorch dependencies."""
    additional_packages: tuple[str, ...] = (OptionalPackage.ROBOKUDO,)
    """Packages whose optional dependencies are installed."""
    workspace_extras: tuple[str, ...] = ()
    """Optional dependency selections in package:extra form."""

    @classmethod
    def from_arguments(class_type, arguments: Sequence[str]) -> Self:
        """Parse the checkout and dependency selections supplied on the command line."""
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(ArgumentName.REPOSITORY, type=Path)
        parser.add_argument(
            ArgumentName.DEPENDENCY_SOURCE, choices=list(DependencySource)
        )
        parser.add_argument(
            ArgumentName.ADDITIONAL_PACKAGES,
            nargs=ArgumentCardinality.OPTIONAL,
            default=OptionalPackage.ROBOKUDO,
        )
        parser.add_argument(
            ArgumentName.WORKSPACE_EXTRAS,
            nargs=ArgumentCardinality.OPTIONAL,
            default="",
        )
        configuration = parser.parse_args(arguments)
        return class_type(
            repository=configuration.repository.resolve(),
            dependency_source=configuration.dependency_source,
            additional_packages=tuple(
                filter(None, configuration.additional_packages.split(","))
            ),
            workspace_extras=tuple(
                filter(None, configuration.workspace_extras.split(","))
            ),
        )

    def build_commands(self) -> str:
        """Build validated dependency installation commands for the checkout."""
        if self.dependency_source not in tuple(DependencySource):
            raise InvalidDependencySource(self.dependency_source)
        unknown_packages = set(self.additional_packages) - set(OptionalPackage)
        if unknown_packages:
            raise UnsupportedPackages(tuple(sorted(unknown_packages)))
        installer = (self.repository / FileName.DEPENDENCY_INSTALLER).resolve(
            strict=True
        )
        commands = [
            shlex.join(
                [
                    Program.BASH,
                    str(installer),
                    self.dependency_source,
                    ",".join(dict.fromkeys(self.additional_packages)),
                ]
            )
        ]
        if self.workspace_extras:
            workspace = Workspace.from_json(
                tomllib.loads((self.repository / FileName.PROJECT).read_text())
            )
            extras = [
                WorkspaceFeature.from_selection(selection, self.repository, workspace)
                for selection in self.workspace_extras
            ]
            commands.extend(extra.build_commands() for extra in extras)
        return "\n".join(commands)

    def install(self) -> None:
        """Install selected dependencies in the active environment."""
        subprocess.run(
            [
                Program.BASH,
                CommandArgument.EXIT_ON_FAILURE,
                CommandArgument.OPTION,
                CommandArgument.PIPELINE_FAILURE,
                CommandArgument.COMMAND,
                self.build_commands(),
            ],
            cwd=self.repository,
            check=True,
        )


# %% Command line
if __name__ == "__main__":
    Installation.from_arguments(sys.argv[1:]).install()
