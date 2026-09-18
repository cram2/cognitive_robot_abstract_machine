"""
The paths and names more than one suite in this directory needs.

Two rules decide what belongs here. Anything that *is* a Python module of
:mod:`basstler` is not written down at all - a suite imports the module and reads its
``__name__`` or ``__file__``, so a rename moves with the code instead of leaving a
literal behind. Everything else has no import to derive it from: the bash entry points,
the skill directories, the notes-branch paths and this directory's own dataset. Those
are here, grouped into a :class:`~enum.StrEnum` where they form a set and left as plain
constants where they do not.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from basstler.package_layout import PACKAGE_DIRECTORY, REPOSITORY_ROOT

__all__ = [
    "DATASET_DIRECTORY",
    "NOTES_BRANCH",
    "PACKAGE_DIRECTORY",
    "REPOSITORY_ROOT",
    "SCRUBBED_ENVIRONMENT_PREFIXES",
    "SET_UP_CLONE_DATASET",
    "STUBS_DIRECTORY",
    "UPSTREAM_REVIEW_RESPONSE_DIRECTORY",
    "WORK_BRANCH",
    "PersonalNotesPath",
    "ToolingDirectory",
]

DATASET_DIRECTORY = Path(__file__).parent / "dataset"
"""
This suite's own test data, next to the tests that read it.
"""

STUBS_DIRECTORY = DATASET_DIRECTORY / "stubs"
"""
Executables copied onto a scratch ``PATH`` to stand in for a real ``gh``, ``curl`` or
hook script.
"""

UPSTREAM_REVIEW_RESPONSE_DIRECTORY = DATASET_DIRECTORY / "upstream-review-responses"
"""
The recorded GraphQL responses the upstream review reader is replayed against.
"""

SET_UP_CLONE_DATASET = DATASET_DIRECTORY / "set-up-clone"
"""
A committed tree of everything ``check-setup.sh`` requires of a set-up clone, copied over
a scratch project root rather than written out file by file.
"""


class ToolingDirectory(StrEnum):
    """
    Where this repository keeps what did not move into the package, relative to the
    project root.

    Literals because there is nothing to derive them from: Claude Code finds a hook and a
    skill by path, so the path *is* the interface, and no import knows it.
    """

    HOOKS = ".claude/hooks"
    """
    The bash entry points, which stay put because Claude Code runs them by path.
    """

    PLAN_DASHBOARD_SKILL = ".claude/skills/plan-dashboard"
    """
    The dashboard skill: its instructions, its worked example and its shell entry point.
    """

    STACKED_PULL_REQUEST_MAINTENANCE_SKILL = ".claude/skills/stacked-pr-maintenance"
    """
    The maintenance pass's own instructions.
    """

    @property
    def path(self) -> Path:
        """:return: This directory inside the repository under test."""
        return REPOSITORY_ROOT / self


NOTES_BRANCH = "claude/personal-notes"
"""
The personal-notes branch name the hooks resolve to by default.
"""

WORK_BRANCH = "some-work-branch"
"""
The throwaway branch a scratch repository is left checked out on.
"""


class PersonalNotesPath(StrEnum):
    """
    What the personal-notes branch holds, and where the hooks write its contents into a
    clone - relative to the project root in both cases.

    Literals for the same reason as :class:`ToolingDirectory`: these paths are the
    interface between the shell hooks and the branch, so nothing imports them.
    """

    NOTES_FILE = ".claude/personal/cram-notes.md"
    """
    The notes themselves, which session-start.sh writes into CLAUDE.local.md.
    """

    GIT_IDENTITY = ".claude/personal/git-identity"
    """
    The recorded git identity a clone with none of its own is given.
    """

    SETTINGS_ON_NOTES_BRANCH = ".claude/personal/settings.local.json"
    """
    The Claude Code settings the branch carries.
    """

    LOCAL_SETTINGS = ".claude/settings.local.json"
    """
    Where those settings are synced to in the clone - the file Claude Code itself reads,
    and writes its own permission grants into.
    """

    PLANS = ".claude/personal/plans"
    """
    Where a plan's manifest and roadmap live.
    """

    BRANCH_INDEX = ".claude/personal/plans/_generated/branch-index.tsv"
    """
    The generated reverse index mapping an item's branch to the plan tracking it.
    """


SCRUBBED_ENVIRONMENT_PREFIXES = (
    "CLAUDE_PERSONAL_NOTES_",
    "GIT_AUTHOR_",
    "GIT_COMMITTER_",
)
"""
Variables a scratch run must not inherit, by the prefix of their name.

The session running this suite legitimately has all of them set, and every one of them
changes what a hook resolves - so a test asserting the default resolution has to be run
without them rather than around them.
"""
