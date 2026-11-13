"""
Stream source archives and release temporary source containers.
"""

from __future__ import annotations

# %% Imports
from enum import StrEnum

import pytest
from typing_extensions import TYPE_CHECKING

from scripts.cramployer.settings import DeploymentSettings
from scripts.cramployer.commands import EnvironmentSetup
from scripts.cramployer.source_container import SourceContainer
from scripts.cramployer.launcher_vocabulary import DockerArgument

from .dataset.docker import FixtureEnvironment
from .dataset.paths import FixturePath

if TYPE_CHECKING:
    from pathlib import Path

    from .dataset.deployment import DevelopmentEnvironment


# %% Source archive lifetime
def test_source_container_copies_sources(
    deployment: DevelopmentEnvironment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Copy an archive into host sources and remove the temporary container.
    """
    archive_source = tmp_path / "image-sources"
    archive_source.mkdir()
    source_file = archive_source / FixturePath.EXISTING_PACKAGE
    source_file.write_text(str(archive_source))
    monkeypatch.setenv(FixtureEnvironment.SOURCE_DIRECTORY, str(archive_source))
    destination = tmp_path / "extracted-sources"
    destination.mkdir()
    settings = DeploymentSettings(
        deployment.deployment_directory, tmp_path / FixturePath.CHECKOUT, destination
    )
    with SourceContainer(settings) as container:
        container.copy_sources()
    assert (destination / source_file.name).read_text() == source_file.read_text()


def test_source_container_is_removed_after_extraction_failure(
    deployment: DevelopmentEnvironment, tmp_path: Path
) -> None:
    """
    Remove the temporary container when extraction raises an error.
    """
    settings = DeploymentSettings(
        deployment.deployment_directory,
        tmp_path / FixturePath.CHECKOUT,
        tmp_path / "missing-destination",
    )
    with pytest.raises(FileNotFoundError):
        with SourceContainer(settings) as container:
            raise FileNotFoundError(settings.robot_workspace_sources)
    assert deployment.read_commands()[-1] == [
        DockerArgument.REMOVE,
        container.container_name,
    ]


# %% Preserved source comparison
class SourceChange(StrEnum):
    """Content differences between image and preserved host sources."""

    ADDED = "added"
    """A new repository or package appears in the image."""
    REMOVED = "removed"
    """A former source file no longer exists in the image."""
    EDITED = "edited"
    """An image source file has different contents."""


@pytest.mark.parametrize("change", list(SourceChange))
def test_setup_warns_when_image_sources_differ_without_overwriting_host(
    deployment: DevelopmentEnvironment,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    change: SourceChange,
) -> None:
    """Expose source drift while preserving locally editable ROS files."""
    image_sources = tmp_path / FixturePath.IMAGE_SOURCES
    image_sources.mkdir()
    image_file = image_sources / FixturePath.EXISTING_PACKAGE
    image_file.touch()
    if change == SourceChange.ADDED:
        (image_sources / SourceChange.ADDED).touch()
    elif change == SourceChange.REMOVED:
        image_file.unlink()
    else:
        image_file.write_text(str(image_sources))
    monkeypatch.setenv(FixtureEnvironment.SOURCE_DIRECTORY, str(image_sources))
    host_sources = tmp_path / FixturePath.ROBOT_WORKSPACE_SOURCES
    settings = DeploymentSettings(
        deployment.deployment_directory, tmp_path / FixturePath.CHECKOUT, host_sources
    )
    EnvironmentSetup(settings).prepare_sources()
    assert str(host_sources) in capsys.readouterr().err
    assert {path.name: path.read_bytes() for path in host_sources.iterdir()} == {
        FixturePath.EXISTING_PACKAGE.name: b""
    }


def test_setup_does_not_warn_for_identical_sources(
    deployment: DevelopmentEnvironment,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Accept identical source content without a drift warning."""
    host_sources = tmp_path / FixturePath.ROBOT_WORKSPACE_SOURCES
    monkeypatch.setenv(FixtureEnvironment.SOURCE_DIRECTORY, str(host_sources))
    settings = DeploymentSettings(
        deployment.deployment_directory, tmp_path / FixturePath.CHECKOUT, host_sources
    )
    EnvironmentSetup(settings).prepare_sources()
    assert capsys.readouterr().err == ""
