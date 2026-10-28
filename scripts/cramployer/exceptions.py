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
    Optional package selections unsupported by the shared installer.
    """

    packages: tuple[str, ...]
    """Unsupported package selections."""

    def error_message(self) -> str:
        return f"Unsupported optional packages: {', '.join(self.packages)}."

    def suggest_correction(self) -> str:
        return (
            "Select optional packages listed in cramployer's dependency documentation."
        )


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
