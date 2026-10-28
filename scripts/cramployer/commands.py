"""
Deployment commands with a shared execution contract.
"""

from __future__ import annotations

# %% Imports
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.cramployer.launcher_vocabulary import (
    ContainerArgument,
    GitArgument,
    BuildArgument,
    DeploymentName,
    DockerArgument,
    Executable,
    ExitStatus,
)
from scripts.cramployer.source_container import SourceContainer
from scripts.cramployer.vocabulary import FileName

if TYPE_CHECKING:
    from scripts.cramployer.settings import DeploymentSettings


# %% Command interface
@dataclass
class DeploymentCommand(ABC):
    """
    Execute one deployment action with the configured host environment.
    """

    configuration: DeploymentSettings
    """Source paths, process environment, and selected container settings."""
    authorize_desktop: bool = field(default=False, init=False)
    """
    Whether this command needs active desktop authorization.
    """

    @abstractmethod
    def execute(self, arguments: tuple[str, ...]) -> int:
        """
        Run the action and return its process exit status.
        """


# %% Temporary container commands
@dataclass
class ContainerCommand(DeploymentCommand, ABC):
    """
    Run a command in an independent development container.
    """

    authorize_desktop: bool = field(default=True, init=False)
    """
    Authorize desktop forwarding before creating a container.
    """

    @abstractmethod
    def build_arguments(self, arguments: tuple[str, ...]) -> list[str]:
        """
        Build the command arguments executed inside the container.
        """

    def execute(self, arguments: tuple[str, ...]) -> int:
        command = self.configuration.build_compose_command()
        command.extend(
            [
                DockerArgument.RUN,
                DockerArgument.REMOVE_ON_EXIT,
                DockerArgument.NO_DEPENDENCIES,
                DeploymentName.SERVICE,
                *self.build_arguments(arguments),
            ]
        )
        return self.configuration.run(command).returncode


@dataclass
class DevelopmentShell(ContainerCommand):
    """
    Open an interactive shell or execute supplied application arguments.
    """

    def build_arguments(self, arguments: tuple[str, ...]) -> list[str]:
        return list(arguments) or [Executable.BASH]


@dataclass
class RobotWorkspaceBuild(ContainerCommand):
    """
    Build the shared ROS development workspace.
    """

    def build_arguments(self, arguments: tuple[str, ...]) -> list[str]:
        return [Executable.BASH, DeploymentName.ROBOT_WORKSPACE_SCRIPT]


@dataclass
class InterfaceGeneration(ContainerCommand):
    """
    Regenerate the checkout's ORM interfaces.
    """

    def build_arguments(self, arguments: tuple[str, ...]) -> list[str]:
        return [Executable.PYTHON, DeploymentName.INTERFACE_GENERATOR]


# %% Environment setup
@dataclass
class EnvironmentSetup(DeploymentCommand):
    """
    Build images, prepare sources, and initialize the development workspace.
    """

    authorize_desktop: bool = field(default=True, init=False)
    """
    Authorize desktop forwarding before running setup containers.
    """

    base_image_directory: Path = Path(".github/docker")
    """
    ROS image build context relative to the editable checkout.
    """

    workflow_path: Path = Path(".github/workflows/ci_reusable.yml")
    """
    Installation workflow required before building images.
    """

    def execute(self, arguments: tuple[str, ...]) -> int:
        self.prepare_checkout()
        self.build_base_image()
        self.prepare_sources()
        self.build_development_image()
        self.check_native_extension()
        RobotWorkspaceBuild(self.configuration).execute(())
        InterfaceGeneration(self.configuration).execute(())
        print(
            "Setup complete. Open the development environment with ./cramployer shell"
        )
        return ExitStatus.SUCCESS

    def prepare_checkout(self) -> None:
        """
        Clone a missing checkout without changing an existing one.
        """
        configuration = self.configuration
        if (configuration.source_checkout / FileName.GIT_METADATA).exists():
            return
        configuration.source_checkout.parent.mkdir(parents=True, exist_ok=True)
        configuration.run(
            [
                Executable.GIT,
                GitArgument.CLONE,
                GitArgument.BRANCH,
                configuration.repository_branch,
                GitArgument.SUBMODULES,
                configuration.repository_address,
                str(configuration.source_checkout),
            ]
        )

    def build_base_image(self) -> None:
        """
        Build the shared ROS base image from the selected checkout.
        """
        configuration = self.configuration
        for required_path in (
            configuration.source_checkout
            / self.base_image_directory
            / FileName.DOCKERFILE,
            configuration.source_checkout / self.workflow_path,
        ):
            required_path.resolve(strict=True)
        configuration.run(
            [
                Executable.DOCKER,
                DockerArgument.BUILD,
                DockerArgument.TAG,
                DeploymentName.BASE_IMAGE,
                str(configuration.source_checkout / self.base_image_directory),
            ]
        )

    def prepare_sources(self) -> None:
        """
        Seed empty host ROS sources from the base image.
        """
        self.configuration.robot_workspace_sources.mkdir(parents=True, exist_ok=True)
        if any(self.configuration.robot_workspace_sources.iterdir()):
            return
        with SourceContainer(self.configuration) as container:
            container.copy_sources()

    def build_development_image(self) -> None:
        """
        Build the development image with host ownership and dependency selections.
        """
        configuration = self.configuration
        selections = {
            BuildArgument.USER_IDENTIFIER: str(configuration.user_identifier),
            BuildArgument.GROUP_IDENTIFIER: str(configuration.group_identifier),
            BuildArgument.DEPENDENCY_SOURCE: configuration.dependency_source,
            BuildArgument.ADDITIONAL_PACKAGES: configuration.additional_packages,
            BuildArgument.WORKSPACE_EXTRAS: configuration.workspace_extras,
        }
        command = [
            Executable.DOCKER,
            DockerArgument.BUILDER,
            DockerArgument.BUILD,
            DockerArgument.LOAD,
            DockerArgument.TAG,
            DeploymentName.IMAGE,
            DockerArgument.BUILD_CONTEXT,
            f"{DeploymentName.REPOSITORY_CONTEXT}{configuration.source_checkout}",
        ]
        for selection, value in selections.items():
            command.extend([DockerArgument.BUILD_ARGUMENT, f"{selection}={value}"])
        command.append(str(configuration.deployment_directory))
        configuration.run(command)

    def check_native_extension(self) -> None:
        """
        Verify native extension availability with editable source mounted.
        """
        command = self.configuration.build_compose_command()
        command.extend(
            [
                DockerArgument.RUN,
                DockerArgument.REMOVE_ON_EXIT,
                DockerArgument.NO_DEPENDENCIES,
                DeploymentName.SERVICE,
                Executable.PYTHON,
                ContainerArgument.MODULE,
                ContainerArgument.PYTEST,
                ContainerArgument.CONFIGURATION,
                ContainerArgument.EMPTY_CONFIGURATION,
                ContainerArgument.PLUGIN,
                ContainerArgument.NO_CACHE,
                DeploymentName.NATIVE_TEST,
                ContainerArgument.QUIET,
            ]
        )
        self.configuration.run(command)


# %% Existing-container access
@dataclass
class RootShell(DeploymentCommand):
    """
    Attach a privileged shell to a selected running development container.
    """

    def execute(self, arguments: tuple[str, ...]) -> int:
        if len(arguments) > 1:
            print("Usage: ./cramployer root-shell [CONTAINER]", file=sys.stderr)
            return ExitStatus.INVALID_SELECTION
        configuration = self.configuration
        result = configuration.run(
            [
                Executable.DOCKER,
                DockerArgument.LIST,
                DockerArgument.FILTER,
                f"{DeploymentName.PROJECT_LABEL}{configuration.project_name}",
                DockerArgument.FILTER,
                DeploymentName.SERVICE_LABEL,
                DockerArgument.FORMAT,
                DeploymentName.CONTAINER_NAMES,
            ],
            capture_output=True,
        )
        container_names = result.stdout.strip().splitlines()
        if not container_names:
            print(
                "No running development container. Open ./cramployer shell first.",
                file=sys.stderr,
            )
            return ExitStatus.UNAVAILABLE
        if not arguments and len(container_names) != 1:
            print(
                "Several development containers are running. Choose ./cramployer root-shell CONTAINER:\n"
                + "\n".join(container_names),
                file=sys.stderr,
            )
            return ExitStatus.INVALID_SELECTION
        selected_container = next(iter(arguments), container_names[0])
        if selected_container not in container_names:
            print(
                f"Not a running development container: {selected_container}\nAvailable containers:\n"
                + "\n".join(container_names),
                file=sys.stderr,
            )
            return ExitStatus.INVALID_SELECTION
        return configuration.run(
            [
                Executable.DOCKER,
                DockerArgument.EXECUTE,
                DockerArgument.INTERACTIVE,
                DockerArgument.TERMINAL,
                DockerArgument.USER,
                DeploymentName.ROOT,
                selected_container,
                Executable.BASH,
                ContainerArgument.STARTUP_FILE,
                DeploymentName.ENVIRONMENT,
                ContainerArgument.INTERACTIVE,
            ]
        ).returncode


# %% Configuration and usage
@dataclass
class ConfigurationDisplay(DeploymentCommand):
    """
    Show the resolved development service configuration.
    """

    def execute(self, arguments: tuple[str, ...]) -> int:
        return self.configuration.run(
            [*self.configuration.build_compose_command(), DockerArgument.CONFIGURATION]
        ).returncode


@dataclass
class UsageDisplay(DeploymentCommand):
    """
    Show the launcher's public command usage.
    """

    help_file: Path = Path("command-help.txt")
    """
    Usage document relative to the deployment directory.
    """

    def execute(self, arguments: tuple[str, ...]) -> int:
        print(
            (self.configuration.deployment_directory / self.help_file).read_text(),
            end="",
        )
        return ExitStatus.SUCCESS
