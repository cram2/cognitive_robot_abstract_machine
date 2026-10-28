"""External command and configuration names understood by the host launcher."""

from __future__ import annotations

# %% Imports
from enum import IntEnum, StrEnum


# %% Host configuration
class EnvironmentVariable(StrEnum):
    """Environment supplied by the launcher's shell configuration."""

    EXECUTABLE_PATH = "PATH"
    """Directories searched for host executables."""
    SOURCE_CHECKOUT = "CRAM_SOURCE"
    """Editable CRAM checkout."""
    ROBOT_WORKSPACE_SOURCES = "ROS_SOURCE"
    """Editable ROS package sources."""
    REPOSITORY_ADDRESS = "REPOSITORY_URL"
    """Repository to clone for a missing source checkout."""
    REPOSITORY_BRANCH = "REPOSITORY_BRANCH"
    """Branch to clone for a missing source checkout."""
    USER_IDENTIFIER = "HOST_UID"
    """Host user's numeric identifier."""
    GROUP_IDENTIFIER = "HOST_GID"
    """Host group's numeric identifier."""
    DEPENDENCY_SOURCE = "TORCH_MODE"
    """PyTorch dependency index selection."""
    ADDITIONAL_PACKAGES = "CI_PACKAGES"
    """Optional package-specific CI installations."""
    WORKSPACE_EXTRAS = "WORKSPACE_EXTRAS"
    """Optional workspace dependency groups."""
    PROJECT_NAME = "COMPOSE_PROJECT_NAME"
    """Compose project identifying running development containers."""
    NVIDIA_ACCESS = "ENABLE_NVIDIA"
    """Select the NVIDIA container feature."""
    GRAPHICAL_ACCESS = "ENABLE_GUI"
    """Select the graphical container feature."""
    DISPLAY = "DISPLAY"
    """X11 display used for graphical windows."""


class FeatureSetting(StrEnum):
    """Feature enablement values accepted by the launcher."""

    ENABLED = "true"
    """Enable the corresponding feature."""
    DISABLED = "false"
    """Leave the corresponding feature inactive."""


class ExitStatus(IntEnum):
    """Outcomes of launcher command selection and resource checks."""

    SUCCESS = 0
    """Command completed successfully."""
    UNAVAILABLE = 1
    """A required resource is unavailable."""
    INVALID_SELECTION = 2
    """Arguments do not identify a valid command target."""


# %% Command selection
class CommandName(StrEnum):
    """Public commands and aliases exposed by the launcher."""

    SETUP = "setup"
    """Prepare the development environment."""
    SHELL = "shell"
    """Open a development shell or run an application command."""
    OFFLINE_SHELL = "shell-offline"
    """Compatibility alias for a development shell."""
    ROOT_SHELL = "root-shell"
    """Attach a privileged shell to a running container."""
    BUILD_ROBOT_WORKSPACE = "ros-build"
    """Build ROS packages."""
    GENERATE_INTERFACES = "orm-build"
    """Generate ORM interfaces."""
    CONFIGURATION = "config"
    """Show the resolved Compose configuration."""
    HELP = "help"
    """Show command usage."""
    SHORT_HELP = "-h"
    """Short usage flag."""
    LONG_HELP = "--help"
    """Long usage flag."""


class Executable(StrEnum):
    """Host and container executables invoked by deployment commands."""

    DOCKER = "docker"
    """Container engine executable."""
    GIT = "git"
    """Source checkout executable."""
    BASH = "bash"
    """Container shell executable."""
    PYTHON = "python"
    """Activated container Python executable."""
    ARCHIVE_EXTRACTOR = "tar"
    """Archive extraction executable."""
    DESKTOP_AUTHORIZATION = "xauth"
    """X11 authorization executable."""


class DockerArgument(StrEnum):
    """Docker subcommands and options used by deployment operations."""

    VERSION = "version"
    """Show the Compose version."""
    JSON = "json"
    """Render resolved configuration as JSON."""
    COMPOSE = "compose"
    """Resolve and run the Compose project."""
    ENVIRONMENT_FILE = "--env-file"
    """Select launcher environment settings."""
    FILE = "-f"
    """Select a Compose override file."""
    RUN = "run"
    """Start a temporary container."""
    REMOVE_ON_EXIT = "--rm"
    """Remove a completed container."""
    NO_DEPENDENCIES = "--no-deps"
    """Skip dependent services."""
    BUILD = "build"
    """Build a container image."""
    BUILDER = "buildx"
    """Select the extended image builder."""
    LOAD = "--load"
    """Load an image into the local engine."""
    TAG = "--tag"
    """Assign an image tag."""
    BUILD_CONTEXT = "--build-context"
    """Supply a named image build context."""
    BUILD_ARGUMENT = "--build-arg"
    """Supply an image build argument."""
    CREATE = "create"
    """Create a stopped container."""
    COPY = "cp"
    """Copy files from a container."""
    REMOVE = "rm"
    """Remove a container."""
    LIST = "ps"
    """List running containers."""
    FILTER = "--filter"
    """Restrict container selection."""
    FORMAT = "--format"
    """Select output formatting."""
    EXECUTE = "exec"
    """Execute in an existing container."""
    INTERACTIVE = "--interactive"
    """Keep standard input attached."""
    TERMINAL = "--tty"
    """Allocate a terminal."""
    USER = "--user"
    """Select a container user."""
    CONFIGURATION = "config"
    """Show Compose configuration."""


class DeploymentName(StrEnum):
    """Container names and paths shared by deployment operations."""

    BASE_IMAGE = "cramployer-base:local"
    """ROS base image tag."""
    IMAGE = "cramployer:local"
    """Development image tag."""
    SERVICE = "cram"
    """Development service name."""
    ROOT = "root"
    """Privileged container account."""
    ENVIRONMENT = "/deployment/environment.sh"
    """ROS and Python shell startup file."""
    ROBOT_WORKSPACE_SCRIPT = "/deployment/build-ros.sh"
    """ROS workspace build script."""
    INTERFACE_GENERATOR = "scripts/regenerate_all_orm.py"
    """ORM interface generator."""
    NATIVE_TEST = "/deployment/container_tests/test_native_extension.py"
    """Native extension smoke checks."""
    SOURCE_ARCHIVE = "/opt/ros/overlay_ws/src/."
    """ROS package sources within the base image."""
    CONTAINER_NAMES = "{{.Names}}"
    """Docker container-name output template."""
    PROJECT_LABEL = "label=com.docker.compose.project="
    """Compose project selection filter."""
    SERVICE_LABEL = "label=com.docker.compose.service=cram"
    """Development service selection filter."""
    REPOSITORY_CONTEXT = "repository="
    """Named source checkout build context."""


class BuildArgument(StrEnum):
    """Image configuration arguments supplied by setup."""

    USER_IDENTIFIER = "HOST_UID"
    """Host user identifier for image ownership."""
    GROUP_IDENTIFIER = "HOST_GID"
    """Host group identifier for image ownership."""
    DEPENDENCY_SOURCE = "TORCH_MODE"
    """PyTorch dependency source."""
    ADDITIONAL_PACKAGES = "CI_PACKAGES"
    """Optional package-specific dependencies."""
    WORKSPACE_EXTRAS = "WORKSPACE_EXTRAS"
    """Optional workspace dependency groups."""


# %% External process options
class GitArgument(StrEnum):
    """Git arguments for creating an editable checkout."""

    CLONE = "clone"
    """Create a source checkout."""
    BRANCH = "--branch"
    """Select its initial branch."""
    SUBMODULES = "--recurse-submodules"
    """Initialize dependent source repositories."""


class ArchiveArgument(StrEnum):
    """Archive extraction arguments for host source initialization."""

    EXTRACT = "--extract"
    """Extract archive members."""
    HOST_OWNERSHIP = "--no-same-owner"
    """Give extracted files to the invoking host user."""
    FILE = "--file"
    """Select the archive input."""
    STANDARD_INPUT = "-"
    """Read an archive from standard input."""
    DIRECTORY = "--directory"
    """Select the extraction destination."""


class AuthorizationArgument(StrEnum):
    """Arguments accepted by the desktop authorization executable."""

    LIST = "nlist"
    """Read the desktop's authorization records."""
    FILE = "-f"
    """Select the authorization file."""
    MERGE = "nmerge"
    """Import authorization records."""
    STANDARD_INPUT = "-"
    """Read records from standard input."""


class ContainerArgument(StrEnum):
    """Shell and smoke-test arguments executed inside development containers."""

    STARTUP_FILE = "--rcfile"
    """Select the shell startup file."""
    INTERACTIVE = "-i"
    """Open an interactive shell."""
    MODULE = "-m"
    """Execute a Python module."""
    PYTEST = "pytest"
    """Run the native smoke checks."""
    CONFIGURATION = "-c"
    """Select the pytest configuration."""
    EMPTY_CONFIGURATION = "/dev/null"
    """Disable repository-wide pytest configuration."""
    PLUGIN = "-p"
    """Select a pytest plugin."""
    NO_CACHE = "no:cacheprovider"
    """Disable writes into the read-only deployment directory."""
    QUIET = "-q"
    """Limit smoke-test output."""
