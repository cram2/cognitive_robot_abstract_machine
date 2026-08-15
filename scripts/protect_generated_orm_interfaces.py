"""
Mark tracked ormatic_interface.py files with git's skip-worktree bit.

Once skip-worktree is set on a path, git treats its working-tree content as always equal
to what is committed: the file never shows up in ``git status``/``git diff``, and ``git
add`` -- explicit or via ``-A``/``.`` -- cannot stage it. This lets a locally
regenerated ORM interface (see ``scripts/regenerate_all_orm.py``) stay on disk for
database work without ever risking an accidental commit of its real, unreviewed content.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing_extensions import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def tracked_ormatic_interfaces() -> Sequence[str]:
    """
    List every ormatic_interface.py path tracked by git, relative to the repository
    root.

    :return: Repository-relative paths of tracked ORM interface files.
    """
    result = subprocess.run(
        ["git", "ls-files", "--", "*/ormatic_interface.py"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.splitlines()


def mark_skip_worktree(path: str) -> None:
    """
    Set the skip-worktree bit on ``path`` so git ignores local changes to it.

    :param path: Repository-relative path of a tracked ORM interface file.
    """
    subprocess.run(
        ["git", "update-index", "--skip-worktree", path],
        cwd=REPOSITORY_ROOT,
        check=True,
    )


def main() -> None:
    """
    Set the skip-worktree bit on every tracked ORM interface file.
    """
    for path in tracked_ormatic_interfaces():
        mark_skip_worktree(path)


if __name__ == "__main__":
    sys.exit(main())
