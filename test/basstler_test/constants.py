"""
The paths and names more than one suite in this directory needs.

Anything :mod:`basstler` already names - a module, or a location in
:mod:`basstler.locations` - is not written down here: a suite imports it, so a rename
moves with the code instead of leaving a literal behind. What remains has no import to
derive it from: this directory's own dataset, the skill directories, the notes-branch
paths only the shell hooks name, and the names a scratch repository is built with.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from basstler.locations import PathEnumeration, ProjectLocation


class DatasetLocation(PathEnumeration):
    """
    This suite's own test data, next to the tests that read it.
    """

    DIRECTORY = Path(__file__).parent / "dataset"
    """
    The dataset directory itself.
    """

    STUBS = DIRECTORY / "stubs"
    """
    Executables copied onto a scratch ``PATH`` to stand in for a real ``gh``, ``curl`` or
    hook script.
    """

    UPSTREAM_REVIEW_RESPONSES = DIRECTORY / "upstream-review-responses"
    """
    The recorded GraphQL responses the upstream review reader is replayed against.
    """

    INSTALLED_DISTRIBUTIONS = DIRECTORY / "installed-distributions"
    """
    A directory that reads as installed distributions once it is on ``sys.path``, holding one
    whose name is spelled with every separator a distribution name may use.
    """

    SET_UP_CLONE = DIRECTORY / "set-up-clone"
    """
    A committed tree of everything ``check-setup.sh`` requires of a set-up clone, copied over
    a scratch project root rather than written out file by file.

    Its ``basstler/`` stands in for the real package: the ``tooling_files`` check tests only
    that each path exists, so nothing there is ever imported or run.
    """


class StackBranch(StrEnum):
    """
    The branches a stack under test is built from, named once for every suite that
    builds one.

    A suite names several of them per test, in a board entry, a git command and an
    assertion at once, so a spelling that differs anywhere is a test quietly about a
    branch its stack does not contain.
    """

    PARENT = "a-parent"
    """
    The bottom of the chain, cut from the base.
    """

    CHILD = "a-child"
    """
    Stacked directly on the parent.
    """


class StackLabel(StrEnum):
    """
    The labels the workflow under test reads and writes, named once for every suite that
    puts one on a pull request.

    A label is written into a board entry, handed to a command and read back in an
    assertion, so a suite that spells it is holding the code to a name nothing else in
    the suite has to agree with.
    """

    IN_REVIEW = "in-review"
    """
    Carried by a branch that has reached the upstream review queue.
    """

    REBASE = "rebase"
    """
    Authorises rewriting a branch's published history rather than merging into it.
    """

    NEEDS_RESOLUTION = "needs-resolution"
    """
    Put on a branch whose owner has been asked to resolve a conflict.
    """

    BUG = "bug"
    """
    Carried by a fix, and never acted on by this tooling - a label it reads past.
    """


class SkillDirectory(PathEnumeration):
    """
    The skills the suites read, relative to the project root.

    Claude Code finds a skill by its path, so the path *is* the interface, and the package
    does not name it.
    """

    PLAN_DASHBOARD = ProjectLocation.CLAUDE_CODE_DIRECTORY / "skills" / "plan-dashboard"
    """
    The dashboard skill: its instructions, its worked example and its shell entry point.
    """

    STACKED_PULL_REQUEST_MAINTENANCE = (
        ProjectLocation.CLAUDE_CODE_DIRECTORY / "skills" / "stacked-pr-maintenance"
    )
    """
    The maintenance pass's own instructions.
    """


class ScratchBranch(StrEnum):
    """
    The branches a scratch repository is given besides a stack's own.
    """

    PERSONAL_NOTES = "claude/personal-notes"
    """
    The personal-notes branch name the hooks resolve to by default.
    """

    WORK = "some-work-branch"
    """
    The throwaway branch a scratch repository is left checked out on.
    """


class PersonalNotesPath(PathEnumeration):
    """
    The files the shell hooks read from the personal-notes branch or write into a clone,
    relative to the project root, that the package itself does not name.
    """

    GIT_IDENTITY = ProjectLocation.PERSONAL_NOTES / "git-identity"
    """
    The recorded git identity a clone with none of its own is given.
    """

    SETTINGS_ON_NOTES_BRANCH = ProjectLocation.PERSONAL_NOTES / "settings.local.json"
    """
    The Claude Code settings the branch carries.
    """

    LOCAL_SETTINGS = ProjectLocation.CLAUDE_CODE_DIRECTORY / "settings.local.json"
    """
    Where those settings are synced to in the clone - the file Claude Code itself reads,
    and writes its own permission grants into.
    """

    BRANCH_INDEX = ProjectLocation.PLANS / "_generated" / "branch-index.tsv"
    """
    The generated reverse index mapping an item's branch to the plan tracking it.
    """


class ScrubbedEnvironmentPrefix(StrEnum):
    """
    Variables a scratch run must not inherit, by the prefix of their name.

    The session running this suite legitimately has all of them set, and every one of them
    changes what a hook resolves - so a test asserting the default resolution has to be run
    without them rather than around them.
    """

    PERSONAL_NOTES = "CLAUDE_PERSONAL_NOTES_"
    """
    The personal-notes remote, branch and path overrides.
    """

    GIT_AUTHOR = "GIT_AUTHOR_"
    """
    The commit author git would otherwise take from the configuration.
    """

    GIT_COMMITTER = "GIT_COMMITTER_"
    """
    The committer git would otherwise take from the configuration.
    """
