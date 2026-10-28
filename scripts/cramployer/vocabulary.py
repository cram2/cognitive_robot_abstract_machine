"""Names shared by installation configuration and workflow parsing."""

from __future__ import annotations

# %% Imports
from enum import StrEnum


# %% Workflow and package metadata
class PayloadKey(StrEnum):
    """Keys in GitHub Actions, Compose, and project metadata."""

    JOBS = "jobs"
    """Workflow jobs."""
    TEST = "test"
    """Job providing installation steps."""
    STEPS = "steps"
    """Ordered job steps."""
    NAME = "name"
    """Step title."""
    RUN = "run"
    """Step shell commands."""
    STEP_ENVIRONMENT = "env"
    """Step environment variables."""
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


class StepTitle(StrEnum):
    """Installation steps supplied by the repository workflow."""

    WORKSPACE = "Install dependencies"
    """Required workspace installation."""
    ROBOKUDO = "Install robokudo optional requirements"
    """Optional RoboKudo installation."""
    DRAKE = "Install drake"
    """Optional Drake installation."""


class OptionalPackage(StrEnum):
    """Packages with additional installation steps in CI."""

    ROBOKUDO = "robokudo"
    """Visual perception package."""
    SEMANTIC_DIGITAL_TWIN = "semantic_digital_twin"
    """World representation package."""


class DependencySource(StrEnum):
    """Sources from which PyTorch dependencies are resolved."""

    DEFAULT = "default"
    """Use the normal package index."""
    PROCESSOR_ONLY = "cpu"
    """Retain CI's CPU-only package index."""


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
    CHECKOUT = "cognitive_robot_abstract_machine"
    """Checkout directory expected by CI steps."""
    SHELL_PREAMBLE = "installation.sh"
    """Shell environment for image installation."""
    EXTRA_TEMPLATE = "workspace-extra.sh"
    """Command template for an optional workspace feature."""


class ShellSyntax(StrEnum):
    """Foreign syntax that cannot be evaluated during image installation."""

    GITHUB_EXPRESSION = "${{"
    """Expression requiring a GitHub Actions runner."""
    PROCESSOR_INDEX = "--extra-index-url https://download.pytorch.org/whl/cpu"
    """CI's explicit PyTorch CPU index selection."""


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
