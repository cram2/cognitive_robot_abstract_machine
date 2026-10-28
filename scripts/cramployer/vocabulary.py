"""Names shared by dependency installation and deployment configuration."""

from __future__ import annotations

# %% Imports
from enum import StrEnum


# %% Package and service metadata
class PayloadKey(StrEnum):
    """Keys in Compose and project metadata."""

    TOOL = "tool"
    """Tool-specific project configuration."""
    UV = "uv"
    """Workspace manager configuration."""
    WORKSPACE = "workspace"
    """Workspace definition."""
    MEMBERS = "members"
    """Workspace package directories."""
    PROJECT = "project"
    """Project metadata."""
    OPTIONAL_DEPENDENCIES = "optional-dependencies"
    """Declared package extras."""
    SERVICES = "services"
    """Compose services."""
    CRAM = "cram"
    """Development service."""
    ENVIRONMENT = "environment"
    """Resolved service environment."""


class OptionalPackage(StrEnum):
    """Packages with optional dependency installation."""

    ROBOKUDO = "robokudo"
    """Visual perception package."""
    SEMANTIC_DIGITAL_TWIN = "semantic_digital_twin"
    """World representation package."""


class DependencySource(StrEnum):
    """Sources from which PyTorch dependencies are resolved."""

    DEFAULT = "default"
    """Use the normal package index."""
    PROCESSOR_ONLY = "cpu"
    """Select the CPU-only PyTorch package index."""


# %% Installation files and shell vocabulary
class FileName(StrEnum):
    """Files and directories shared with installation commands."""

    ENVIRONMENT = ".env"
    """Local deployment environment settings."""
    GIT_METADATA = ".git"
    """Metadata identifying an existing source checkout."""
    DOCKERFILE = "Dockerfile"
    """Image build instructions."""
    PROJECT = "pyproject.toml"
    """Package and workspace metadata."""
    DEPENDENCY_INSTALLER = "scripts/install_dependencies.sh"
    """Shared dependency installation entry point relative to a checkout."""


class Program(StrEnum):
    """Executables invoked by the installer."""

    BASH = "bash"
    """Shell executing installation steps."""
    UV = "uv"
    """Python workspace manager."""


class CommandArgument(StrEnum):
    """Arguments used to install workspace features and execute shell commands."""

    NO_SITE_PACKAGES = "-S"
    """Disable loading Python site packages."""
    MODULE = "-m"
    """Run an importable Python module."""
    SYNTAX_CHECK = "-n"
    """Check shell syntax without executing commands."""
    SYNC = "sync"
    """Synchronize workspace dependencies."""
    ACTIVE = "--active"
    """Install into the activated environment."""
    INEXACT = "--inexact"
    """Preserve already installed packages."""
    PACKAGE = "--package"
    """Select a workspace package."""
    EXTRA = "--extra"
    """Select an optional dependency group."""
    EXIT_ON_FAILURE = "-e"
    """Stop a shell when a command fails."""
    OPTION = "-o"
    """Enable a named shell option."""
    PIPELINE_FAILURE = "pipefail"
    """Propagate failures in pipelines."""
    COMMAND = "-c"
    """Execute supplied shell commands."""
