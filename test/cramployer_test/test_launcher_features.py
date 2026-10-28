"""
Compose hardware features and prepare private graphical authorization.
"""

from __future__ import annotations

# %% Imports
import stat

import pytest
from typing_extensions import TYPE_CHECKING

from scripts.cramployer.exceptions import DesktopUnavailable
from scripts.cramployer.features import GraphicalAccess, NvidiaAccess
from scripts.cramployer.launch import DeploymentLauncher
from scripts.cramployer.launcher_vocabulary import (
    CommandName,
    EnvironmentVariable,
    ExitStatus,
    FeatureSetting,
)
from scripts.cramployer.settings import DeploymentSettings

from .dataset.docker import EnvironmentVariable as RecordingEnvironment
from .dataset.paths import FixturePath

if TYPE_CHECKING:
    from pathlib import Path

    from .dataset.deployment import DevelopmentEnvironment


# %% Settings fixture
@pytest.fixture
def settings(deployment: DevelopmentEnvironment, tmp_path: Path) -> DeploymentSettings:
    """
    Provide host settings that run Docker through the command recorder.
    """
    return DeploymentSettings(
        deployment.deployment_directory,
        tmp_path / FixturePath.CHECKOUT,
        tmp_path / FixturePath.ROBOT_WORKSPACE_SOURCES,
    )


# %% Independent hardware selection
@pytest.mark.parametrize(
    "features",
    [
        (),
        (EnvironmentVariable.NVIDIA_ACCESS,),
        (EnvironmentVariable.GRAPHICAL_ACCESS,),
        (EnvironmentVariable.NVIDIA_ACCESS, EnvironmentVariable.GRAPHICAL_ACCESS),
    ],
)
def test_hardware_features_compose_independently(
    settings: DeploymentSettings, features: tuple[EnvironmentVariable, ...]
) -> None:
    """
    Apply every enabled hardware override to the same Compose invocation.
    """
    expected_files = settings.compose_files.copy()
    available = {
        EnvironmentVariable.NVIDIA_ACCESS: NvidiaAccess(),
        EnvironmentVariable.GRAPHICAL_ACCESS: GraphicalAccess(),
    }
    for feature in features:
        settings.process_environment[feature] = FeatureSetting.ENABLED
        expected_files.append(available[feature].compose_file)
    assert (
        DeploymentLauncher(settings).run((CommandName.CONFIGURATION,))
        == ExitStatus.SUCCESS
    )
    assert settings.compose_files == expected_files


# %% Desktop authorization
def test_graphical_access_restores_saved_display(settings: DeploymentSettings) -> None:
    """
    Restore the previous display for configuration commands.
    """
    feature = GraphicalAccess()
    display_file = settings.deployment_directory / feature.display_file
    display_file.parent.mkdir(parents=True)
    display = ":12"
    display_file.write_text(display)
    settings.process_environment.pop(EnvironmentVariable.DISPLAY, None)
    feature.configure(settings, authorize_desktop=False)
    assert settings.process_environment[EnvironmentVariable.DISPLAY] == display


def test_graphical_access_requires_a_display(settings: DeploymentSettings) -> None:
    """
    Reject graphical container creation when no desktop display is available.
    """
    settings.process_environment.pop(EnvironmentVariable.DISPLAY, None)
    with pytest.raises(DesktopUnavailable):
        GraphicalAccess().configure(settings, authorize_desktop=True)


def test_graphical_access_copies_private_authorization(
    settings: DeploymentSettings, tmp_path: Path
) -> None:
    """
    Copy the desktop cookie with the configured address family and permissions.
    """
    executable = (
        tmp_path / FixturePath.EXECUTABLES / FixturePath.AUTHORIZATION_EXECUTABLE
    )
    executable.write_bytes(FixturePath.AUTHORIZATION_MIMIC.read_bytes())
    executable.chmod(0o755)
    settings.process_environment[EnvironmentVariable.DISPLAY] = ":12"
    feature = GraphicalAccess()
    feature.configure(settings, authorize_desktop=True)
    authority_file = settings.deployment_directory / feature.authority_file
    original_cookie = FixturePath.DESKTOP_COOKIE.read_text()
    assert (
        authority_file.read_text()
        == feature.authorization_family
        + original_cookie[len(feature.authorization_family) :]
    )
    assert stat.S_IMODE(authority_file.stat().st_mode) == feature.file_permissions


def test_graphical_access_rejects_an_empty_cookie(
    settings: DeploymentSettings, tmp_path: Path
) -> None:
    """
    Reject desktop authorization when the selected display has no cookie.
    """
    executable = (
        tmp_path / FixturePath.EXECUTABLES / FixturePath.AUTHORIZATION_EXECUTABLE
    )
    executable.write_bytes(FixturePath.AUTHORIZATION_MIMIC.read_bytes())
    executable.chmod(0o755)
    empty_cookie = tmp_path / "empty-cookie"
    empty_cookie.touch()
    settings.process_environment[EnvironmentVariable.DISPLAY] = ":12"
    settings.process_environment[RecordingEnvironment.DESKTOP_COOKIE] = str(
        empty_cookie
    )
    with pytest.raises(DesktopUnavailable):
        GraphicalAccess().configure(settings, authorize_desktop=True)
