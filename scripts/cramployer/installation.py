"""Install a checkout using its CI dependency commands and selected extras."""

from __future__ import annotations

# %% Imports
import argparse
import os
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml
from typing_extensions import TYPE_CHECKING

from scripts.cramployer.exceptions import (
    CheckoutConflict,
    UnsupportedPackages,
)
from scripts.cramployer.vocabulary import (
    CommandArgument,
    FileName,
    OptionalPackage,
    Program,
    StepTitle,
    DependencySource,
)
from scripts.cramployer.dependency_index import DependencyIndex
from scripts.cramployer.workflow import Workflow
from scripts.cramployer.workspace import Workspace, WorkspaceFeature

if TYPE_CHECKING:
    from typing_extensions import Self, Sequence

    from scripts.cramployer.workflow import WorkflowStep


# %% Command-line configuration
class ArgumentName(StrEnum):
    """Positional installation arguments."""

    REPOSITORY = "repository"
    """Checkout whose dependencies are installed."""
    DEPENDENCY_SOURCE = "dependency_source"
    """PyTorch dependency source."""
    ADDITIONAL_PACKAGES = "additional_packages"
    """Additional package-specific CI installations."""
    WORKSPACE_EXTRAS = "workspace_extras"
    """Optional workspace dependency groups."""


class ArgumentCardinality(StrEnum):
    """Cardinality of optional positional command-line selections."""

    OPTIONAL = "?"
    """Accept zero or one positional value."""


# %% CI installation
@dataclass
class Installation:
    """Install workspace dependencies selected from checkout configuration."""

    repository: Path
    """Checkout whose workflow supplies installation commands."""
    dependency_source: str
    """Requested source for PyTorch dependencies."""
    additional_packages: tuple[str, ...] = (OptionalPackage.ROBOKUDO,)
    """Packages whose additional CI installation steps are enabled."""
    workspace_extras: tuple[str, ...] = ()
    """Optional dependency selections in package:extra form."""
    workflow_path: Path = Path(".github/workflows/ci_reusable.yml")
    """Workflow path relative to the checkout."""

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

    def select_steps(self) -> list[WorkflowStep]:
        """Resolve required and selected optional CI installation steps."""
        optional_steps: dict[str, StepTitle] = {
            OptionalPackage.ROBOKUDO: StepTitle.ROBOKUDO,
            OptionalPackage.SEMANTIC_DIGITAL_TWIN: StepTitle.DRAKE,
        }
        unknown_packages = set(self.additional_packages) - optional_steps.keys()
        if unknown_packages:
            raise UnsupportedPackages(tuple(sorted(unknown_packages)))
        titles = [
            StepTitle.WORKSPACE,
            *(
                optional_steps[package]
                for package in dict.fromkeys(self.additional_packages)
            ),
        ]
        workflow = Workflow.from_json(
            yaml.safe_load((self.repository / self.workflow_path).read_text())
        )
        return workflow.select_steps(titles)

    def build_commands(self, steps: list[WorkflowStep] | None = None) -> str:
        """Build validated installation commands for the selected dependencies."""
        dependency_index = DependencyIndex.from_mode(self.dependency_source)
        selected = self.select_steps() if steps is None else steps
        commands = [step.build_commands() for step in selected]
        if self.workspace_extras:
            workspace = Workspace.from_json(
                tomllib.loads((self.repository / FileName.PROJECT).read_text())
            )
            extras = [
                WorkspaceFeature.from_selection(selection, self.repository, workspace)
                for selection in self.workspace_extras
            ]
            commands.extend(extra.build_commands() for extra in extras)
        shell_commands = "\n".join(commands)
        return dependency_index.configure_commands(shell_commands)

    def prepare_checkout(self) -> Path:
        """Make CI's checkout directory resolve to this repository without replacing paths."""
        repository = self.repository.resolve()
        checkout = repository.parent / FileName.CHECKOUT
        if checkout.resolve() == repository:
            return repository.parent
        if checkout.exists() or checkout.is_symlink():
            raise CheckoutConflict(checkout)
        checkout.symlink_to(repository, target_is_directory=True)
        return repository.parent

    def install(self) -> None:
        """Install the selected dependencies in the active image environment."""
        steps = self.select_steps()
        commands = self.build_commands(steps)
        environment = os.environ.copy()
        for step in steps:
            environment.update(step.environment)
        working_directory = self.prepare_checkout()
        preamble = (Path(__file__).parent / FileName.SHELL_PREAMBLE).read_text()
        subprocess.run(
            [
                Program.BASH,
                CommandArgument.EXIT_ON_FAILURE,
                CommandArgument.OPTION,
                CommandArgument.PIPELINE_FAILURE,
                CommandArgument.COMMAND,
                preamble + "\n" + commands,
            ],
            cwd=working_directory,
            env=environment,
            check=True,
        )


# %% Command line
if __name__ == "__main__":
    Installation.from_arguments(sys.argv[1:]).install()
