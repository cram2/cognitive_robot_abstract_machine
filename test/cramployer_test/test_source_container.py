"""
Stream source archives and release temporary source containers.
"""

from __future__ import annotations

# %% Imports

import pytest
from typing_extensions import TYPE_CHECKING

from scripts.cramployer.settings import DeploymentSettings
from scripts.cramployer.source_container import SourceContainer
from scripts.cramployer.launcher_vocabulary import DockerArgument

from .dataset.docker import EnvironmentVariable
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
    monkeypatch.setenv(EnvironmentVariable.SOURCE_DIRECTORY, str(archive_source))
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
