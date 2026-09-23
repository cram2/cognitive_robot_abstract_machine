"""
Standing an executable in for the real one, on a test subprocess's ``PATH``.

Two suites need the same two moves - put a stub where a script will find it, and take a
real executable out of reach - so both live here rather than in whichever suite happened
to need them first.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .constants import STUBS_DIRECTORY


@dataclass(frozen=True)
class ExecutableStubDirectory:
    """
    A directory of stubbed executables, meant to be placed first on a subprocess's
    ``PATH`` so the script under test finds the stub instead of the real thing.
    """

    path: Path
    """
    The directory itself, which is what goes on ``PATH``.
    """

    @classmethod
    def create(cls, parent: Path) -> ExecutableStubDirectory:
        """
        :param parent: Where to create the directory, typically pytest's ``tmp_path``.
        :return: An empty stub directory, ready for :meth:`install`.
        """
        path = parent / "stub-bin"
        path.mkdir()
        return cls(path)

    def install(self, executable_name: str) -> None:
        """
        Copy the stub backing *executable_name*, from ``dataset/stubs/<name>.sh``.

        :param executable_name: The executable to stand in for, e.g. ``"gh"``.
        """
        destination = self.path / executable_name
        shutil.copy(STUBS_DIRECTORY / f"{executable_name}.sh", destination)
        destination.chmod(0o755)

    def ahead_of(self, search_path: str) -> str:
        """
        :param search_path: A ``PATH`` string to fall back to.
        :return: *search_path* with this directory first, so its stubs win.
        """
        return os.pathsep.join([str(self.path), search_path])


def path_hiding_executable(executable_name: str, mirror_parent: Path) -> str:
    """
    Build a ``PATH`` string equivalent to the current one but with *executable_name*
    unfindable through it.

    Mirrors (via symlinks) any directory that provides *executable_name* into a copy
    missing just that one file, rather than dropping the whole directory from `PATH` -
    the directory providing it (typically ``/usr/bin``) also provides ``bash``, ``git``
    and ``python3``, which the script under test still needs to run at all.

    :param executable_name: The executable to hide, e.g. ``"gh"``.
    :param mirror_parent: Where to create mirror directories.
    :return: The adjusted ``PATH`` string.
    """
    entries = []
    mirror_index = 0
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        directory = Path(entry)
        if not directory.is_dir() or not (directory / executable_name).exists():
            entries.append(entry)
            continue
        mirror = mirror_parent / f"hide-{executable_name}-{mirror_index}"
        mirror_index += 1
        mirror.mkdir()
        for item in directory.iterdir():
            if item.name == executable_name:
                continue
            try:
                (mirror / item.name).symlink_to(item)
            except OSError:
                continue
        entries.append(str(mirror))
    return os.pathsep.join(entries)
