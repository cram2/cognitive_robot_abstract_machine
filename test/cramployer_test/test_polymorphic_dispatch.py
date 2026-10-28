"""
Extend deployment behavior through explicit command and feature interfaces.
"""

from __future__ import annotations

# %% Imports
from typing_extensions import TYPE_CHECKING

from scripts.cramployer.launch import DeploymentLauncher
from scripts.cramployer.settings import DeploymentSettings

from .dataset.dispatch import (
    DispatchName,
    ComposeFileRecording,
    SelectionRecording,
    AdditionalComposeFile,
)
from scripts.cramployer.launcher_vocabulary import ExitStatus

if TYPE_CHECKING:
    from pathlib import Path


# %% Command extension
def test_launcher_executes_an_added_command(tmp_path: Path) -> None:
    """
    Run a command implementation registered by the caller.
    """
    settings = DeploymentSettings(tmp_path, tmp_path, tmp_path)
    launcher = DeploymentLauncher(
        settings, command_types={DispatchName.RECORD: SelectionRecording}
    )
    arguments = (DispatchName.RECORD, str(tmp_path))
    assert launcher.run(arguments) == ExitStatus.SUCCESS
    assert (tmp_path / DispatchName.OUTPUT).read_text() == str(tmp_path)


# %% Feature extension
def test_launcher_composes_an_added_container_feature(tmp_path: Path) -> None:
    """
    Apply a registered feature through the common feature contract.
    """
    settings = DeploymentSettings(
        tmp_path,
        tmp_path,
        tmp_path,
        process_environment={DispatchName.ENABLED_FEATURE: DispatchName.ENABLED},
    )
    original_files = settings.compose_files.copy()
    feature = AdditionalComposeFile()
    launcher = DeploymentLauncher(
        settings,
        command_types={DispatchName.RECORD: ComposeFileRecording},
        feature_types={DispatchName.ENABLED_FEATURE: AdditionalComposeFile},
    )
    assert launcher.run((DispatchName.RECORD,)) == ExitStatus.SUCCESS
    assert (tmp_path / DispatchName.OUTPUT).read_text().splitlines() == [
        str(path) for path in [*original_files, feature.compose_file]
    ]
