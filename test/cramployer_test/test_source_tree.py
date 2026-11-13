"""Compare ROS source contents independently of local repository metadata."""

from __future__ import annotations

# %% Imports
from enum import StrEnum

import pytest
from typing_extensions import TYPE_CHECKING

from scripts.cramployer.source_tree import IgnoredDirectory, SourceFile, SourceTree

from .dataset.paths import FixturePath

if TYPE_CHECKING:
    from pathlib import Path


# %% Source entries
class SourcePath(StrEnum):
    """File names in source comparison fixtures."""

    LINK = "source-link"
    """A source symlink whose target can differ."""
    OTHER = "other-source"
    """An alternate source or symlink target."""


# %% Content comparison
@pytest.mark.parametrize("ignored_directory", list(IgnoredDirectory))
def test_source_tree_ignores_metadata(
    tmp_path: Path, ignored_directory: IgnoredDirectory
) -> None:
    """Exclude Git metadata and bytecode caches from source contents."""
    ignored = tmp_path / ignored_directory
    ignored.mkdir()
    (ignored / FixturePath.EXISTING_PACKAGE).write_text(str(ignored))
    assert SourceTree(tmp_path).read_files() == {}


def test_source_tree_records_symlink_target_without_following_it(
    tmp_path: Path,
) -> None:
    """Compare link targets even when the destination does not exist."""
    link = tmp_path / SourcePath.LINK
    link.symlink_to(SourcePath.OTHER)
    assert SourceTree(tmp_path).read_files() == {
        link.relative_to(tmp_path): SourceFile(link_target=link.readlink())
    }


def test_source_tree_detects_equal_size_content_edits(tmp_path: Path) -> None:
    """Detect changed source bytes even when file sizes are unchanged."""
    source_file = tmp_path / FixturePath.EXISTING_PACKAGE
    source_file.write_text(SourcePath.LINK)
    original = SourceTree(tmp_path).read_files()
    source_file.write_text(SourcePath.LINK[::-1])
    assert SourceTree(tmp_path).read_files() != original
