"""
Dependency index policies selected for PyTorch installation.
"""

from __future__ import annotations

# %% Imports
from abc import ABC, abstractmethod
from dataclasses import dataclass

from scripts.cramployer.exceptions import InvalidDependencySource
from scripts.cramployer.vocabulary import ShellSyntax, DependencySource


# %% Dependency index policies
@dataclass
class DependencyIndex(ABC):
    """
    Configure package indexes in dependency installation commands.
    """

    @classmethod
    def from_mode(class_type, selection: str) -> DependencyIndex:
        """
        Select a supported index policy from an installation setting.
        """
        implementations: dict[str, type[DependencyIndex]] = {
            DependencySource.PROCESSOR_ONLY: CentralProcessorIndex,
            DependencySource.DEFAULT: DefaultIndex,
        }
        if selection not in implementations:
            raise InvalidDependencySource(selection)
        return implementations[selection]()

    @abstractmethod
    def configure_commands(self, shell_commands: str) -> str:
        """
        Apply the dependency index policy to installation commands.
        """


@dataclass
class CentralProcessorIndex(DependencyIndex):
    """
    Retain the workflow's index for CPU-only PyTorch dependencies.
    """

    def configure_commands(self, shell_commands: str) -> str:
        return shell_commands


@dataclass
class DefaultIndex(DependencyIndex):
    """
    Resolve PyTorch dependencies through the normal package index.
    """

    def configure_commands(self, shell_commands: str) -> str:
        return shell_commands.replace(ShellSyntax.PROCESSOR_INDEX, "")
