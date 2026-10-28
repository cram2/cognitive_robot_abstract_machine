"""
Installation failures available before workspace dependencies are installed.

These exceptions deliberately inherit from :class:`ValueError` rather than
:class:`krrood.exceptions.DataclassException`: the installer must validate a fresh image
before installing krrood or any other workspace package.
"""

from __future__ import annotations

# %% Imports
from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


# %% Standalone installation failures
@dataclass
class InstallationError(ValueError, ABC):
    """
    An invalid installation selection with corrective advice.
    """

    @abstractmethod
    def error_message(self) -> str:
        """
        Describe the invalid installation state.
        """

    @abstractmethod
    def suggest_correction(self) -> str:
        """
        Explain how to make the installation valid.
        """

    def __str__(self) -> str:
        return f"{self.error_message()}\n{self.suggest_correction()}"


@dataclass
class UnsupportedPackages(InstallationError):
    """
    Optional package selections without corresponding CI installation steps.
    """

    packages: tuple[str, ...]
    """Unsupported package selections."""

    def error_message(self) -> str:
        return f"Unsupported CI packages: {', '.join(self.packages)}."

    def suggest_correction(self) -> str:
        return "Select packages listed in cramployer's CI dependency documentation."


@dataclass
class MissingWorkflowStep(InstallationError):
    """
    A required installation step is absent or ambiguous.
    """

    title: str
    """Expected installation step title."""
    occurrences: int
    """
    Number of matching workflow steps.
    """

    def error_message(self) -> str:
        return f"Expected one CI step titled {self.title!r}; found {self.occurrences}."

    def suggest_correction(self) -> str:
        return "Review the installation step titles in the reusable CI workflow."


@dataclass
class InvalidWorkflowStep(InstallationError):
    """
    An installation step contains commands or environment values requiring review.
    """

    title: str
    """Installation step requiring review."""
    setting: str
    """
    Command or environment setting that cannot be executed.
    """

    def error_message(self) -> str:
        return f"CI step {self.title!r} has an unsupported setting: {self.setting}."

    def suggest_correction(self) -> str:
        return "Use literal shell commands and string environment values without GitHub expressions."


@dataclass
class UnknownWorkspacePackage(InstallationError):
    """
    An extra selection does not identify a workspace member.
    """

    selection: str
    """Requested package and extra."""

    def error_message(self) -> str:
        return f"Unknown workspace package in extra selection: {self.selection!r}."

    def suggest_correction(self) -> str:
        return (
            "Use package:extra with a package declared in the root workspace metadata."
        )


@dataclass
class UnknownWorkspaceExtra(InstallationError):
    """
    A workspace member does not declare the requested extra.
    """

    selection: str
    """Requested package and extra."""

    def error_message(self) -> str:
        return f"Unknown workspace extra: {self.selection!r}."

    def suggest_correction(self) -> str:
        return "Select an extra declared in the package's optional dependencies."


@dataclass
class InvalidDependencySource(InstallationError):
    """
    The requested PyTorch dependency source is unsupported.
    """

    selection: str
    """Requested dependency mode."""

    def error_message(self) -> str:
        return f"Unsupported PyTorch dependency mode: {self.selection!r}."

    def suggest_correction(self) -> str:
        return "Set TORCH_MODE to default or cpu."


@dataclass
class CheckoutConflict(InstallationError):
    """
    CI's checkout path belongs to another directory or file.
    """

    checkout: Path
    """Existing path conflicting with the installation checkout."""

    def error_message(self) -> str:
        return f"The CI checkout path points elsewhere: {self.checkout}."

    def suggest_correction(self) -> str:
        return "Use an installation directory without a conflicting CI checkout path."


# %% Host resource failures
@dataclass
class DesktopUnavailable(InstallationError):
    """
    Desktop authorization cannot be supplied to a graphical container.
    """

    reason: str
    """Host resource preventing desktop authorization."""

    def error_message(self) -> str:
        return self.reason

    def suggest_correction(self) -> str:
        return "Use a desktop session with an X11 cookie, or disable GUI access."


@dataclass
class SourceCopyFailed(InstallationError):
    """
    ROS source extraction from the base image failed.
    """

    container_name: str
    """Container providing the source archive."""
    exit_status: int
    """
    Failure status returned by the source-copy process.
    """

    def error_message(self) -> str:
        return f"Copying sources from {self.container_name} failed with status {self.exit_status}."

    def suggest_correction(self) -> str:
        return "Check Docker access and the base image's ROS source directory."
