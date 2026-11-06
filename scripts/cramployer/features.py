"""
Independent hardware features composed into a development environment.
"""

from __future__ import annotations

# %% Imports
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.cramployer.exceptions import DesktopUnavailable
from scripts.cramployer.launcher_vocabulary import (
    AuthorizationArgument,
    EnvironmentVariable,
    Executable,
)

if TYPE_CHECKING:
    from scripts.cramployer.settings import DeploymentSettings


# %% Container feature interface
@dataclass
class ContainerFeature(ABC):
    """
    Add container configuration and prepare its required host resources.
    """

    def configure(
        self, configuration: DeploymentSettings, authorize_desktop: bool
    ) -> None:
        """
        Select the feature's Compose override and prepare its host environment.
        """
        configuration.compose_files.append(self.select_compose_file())
        self.prepare_environment(configuration, authorize_desktop)

    @abstractmethod
    def select_compose_file(self) -> Path:
        """
        Select the Compose override supplying this feature.
        """

    @abstractmethod
    def prepare_environment(
        self, configuration: DeploymentSettings, authorize_desktop: bool
    ) -> None:
        """
        Prepare host resources required by this feature and command.
        """


# %% Hardware implementations
@dataclass
class NvidiaAccess(ContainerFeature):
    """
    Expose NVIDIA compute and graphics capabilities to development containers.
    """

    compose_file: Path = Path("compose.nvidia.yaml")
    """
    Compose override exposing NVIDIA devices.
    """

    def select_compose_file(self) -> Path:
        return self.compose_file

    def prepare_environment(
        self, configuration: DeploymentSettings, authorize_desktop: bool
    ) -> None:
        pass


@dataclass
class GraphicalAccess(ContainerFeature):
    """
    Forward X11 windows with a private copy of the desktop authorization cookie.
    """

    compose_file: Path = Path("compose.gui.yaml")
    """
    Compose override forwarding graphical applications.
    """

    display_file: Path = Path(".state/display")
    """
    Saved display selection for subsequent commands.
    """

    authority_file: Path = Path(".state/Xauthority")
    """
    Private desktop authorization file mounted into containers.
    """

    file_permissions: int = 0o600
    """
    Permissions limiting authorization files to the host user.
    """

    authorization_family: str = "ffff"
    """
    Address-family prefix for container desktop authorization cookies.
    """

    def select_compose_file(self) -> Path:
        return self.compose_file

    def prepare_environment(
        self, configuration: DeploymentSettings, authorize_desktop: bool
    ) -> None:
        display_file = configuration.deployment_directory / self.display_file
        environment = configuration.process_environment
        if not environment.get(EnvironmentVariable.DISPLAY) and display_file.is_file():
            environment[EnvironmentVariable.DISPLAY] = display_file.read_text().strip()
        if not authorize_desktop:
            return
        self.authorize_display(configuration)

    def authorize_display(self, configuration: DeploymentSettings) -> None:
        """
        Copy the active desktop cookie into a private container authorization file.
        """
        display = configuration.process_environment.get(EnvironmentVariable.DISPLAY, "")
        if (
            not display
            or shutil.which(
                Executable.DESKTOP_AUTHORIZATION,
                path=configuration.process_environment.get(
                    EnvironmentVariable.EXECUTABLE_PATH
                ),
            )
            is None
        ):
            raise DesktopUnavailable(
                "GUI mode requires DISPLAY and host xauth. Use a desktop terminal."
            )
        authorization = configuration.run(
            [Executable.DESKTOP_AUTHORIZATION, AuthorizationArgument.LIST, display],
            capture_output=True,
        ).stdout
        if not authorization.strip():
            raise DesktopUnavailable(
                "No X11 cookie found. Check DISPLAY and XAUTHORITY."
            )
        authority_file = configuration.deployment_directory / self.authority_file
        authority_file.parent.mkdir(parents=True, exist_ok=True)
        authority_file.touch(mode=self.file_permissions, exist_ok=True)
        authority_file.chmod(self.file_permissions)
        cookies = (
            "\n".join(
                self.authorization_family + line[len(self.authorization_family) :]
                for line in authorization.splitlines()
            )
            + "\n"
        )
        configuration.run(
            [
                Executable.DESKTOP_AUTHORIZATION,
                AuthorizationArgument.FILE,
                str(authority_file),
                AuthorizationArgument.MERGE,
                AuthorizationArgument.STANDARD_INPUT,
            ],
            input_text=cookies,
        )
        display_file = configuration.deployment_directory / self.display_file
        display_file.touch(mode=self.file_permissions, exist_ok=True)
        display_file.chmod(self.file_permissions)
        display_file.write_text(display + "\n")
