"""Content comparison of editable ROS sources and image sources."""

from __future__ import annotations

# %% Imports
import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


# %% Source content
class IgnoredDirectory(StrEnum):
    """Metadata excluded from ROS source comparisons."""

    GIT = ".git"
    """Repository history and local Git state."""
    PYTHON_CACHE = "__pycache__"
    """Generated Python bytecode."""


@dataclass
class SourceFile:
    """Content or symbolic link target of a source file."""

    content_digest: str | None = None
    """SHA-256 digest of a regular file's contents."""
    link_target: Path | None = None
    """Target of a symbolic link without following it."""


@dataclass
class SourceTree:
    """A ROS source directory whose content can be compared independently of metadata."""

    directory: Path
    """Root of the source files to inspect."""

    def read_files(self) -> dict[Path, SourceFile]:
        """Describe regular files and symbolic links, excluding repository metadata."""
        files = {}
        for path in self.directory.rglob("*"):
            relative_path = path.relative_to(self.directory)
            if any(part in IgnoredDirectory for part in relative_path.parts):
                continue
            if path.is_symlink():
                files[relative_path] = SourceFile(link_target=path.readlink())
            elif path.is_file():
                with path.open("rb") as stream:
                    files[relative_path] = SourceFile(
                        content_digest=hashlib.file_digest(
                            stream, hashlib.sha256
                        ).hexdigest()
                    )
        return files
