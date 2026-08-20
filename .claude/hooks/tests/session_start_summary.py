"""
Reading session-start.sh's summary report back in a test.

The hook's report is its only observable output, so more than one test module asserts
against it; both halves of doing so live here rather than being written out again in
each of them - parsing a line out of the report, and rendering the line the report is
expected to carry from the same definitions the hook prints it from.

.. note::
   Rendering an expectation rather than restating it means a reword moves both sides at
   once, so wording is deliberately not what these tests pin.
"""

from __future__ import annotations

import subprocess
from enum import StrEnum

from scratch_repository import HOOKS_SOURCE_DIRECTORY

MESSAGES_SCRIPT = HOOKS_SOURCE_DIRECTORY / "session-start-messages.sh"
"""
The shell file defining the wording of every summary line.
"""


class SummaryMessage(StrEnum):
    """
    The situations session-start.sh reports, each carrying the session-start-messages.sh
    function that renders its line.

    Named for the situation rather than for that function: a test reads as the case it
    exercises, while the shell file keeps grouping its own definitions by which summary
    line they belong to.
    """

    PLAN_NOT_APPLICABLE = "plan_line_not_applicable"
    """
    The branch is one no plan item could ever track.
    """

    NO_PLANS_TRACKED = "plan_line_no_plans_tracked"
    """
    The notes branch tracks no plans at all.
    """

    NO_PLAN_ITEM_TRACKS_BRANCH = "plan_line_no_item_tracks_branch"
    """
    Plans are tracked, and none holds an item for this branch.
    """

    PLAN_MANIFEST_MISSING = "plan_line_manifest_missing"
    """
    The index names a plan whose manifest is not on the notes branch.
    """

    BRANCH_TRACKED_IN_PLAN = "plan_line_tracked"
    """
    The branch is a tracked item of a plan that resolved.
    """

    NO_GIT_IDENTITY_RECORDED = "git_identity_line_not_recorded"
    """
    The notes branch carries no git identity at all.
    """

    GIT_IDENTITY_INCOMPLETE = "git_identity_line_incomplete"
    """
    A git identity is recorded with only one of its two halves.
    """

    CLONE_HAS_ITS_OWN_GIT_IDENTITY = "git_identity_line_already_set"
    """
    The clone already has a repository-local identity, which is left alone.
    """

    GIT_IDENTITY_WRITTEN = "git_identity_line_written"
    """
    The recorded identity was written into this clone's repository-local config.
    """

    SETUP_SCRIPT_MISSING = "setup_line_not_checked"
    """
    The check-setup.sh script is not in this checkout, so there is no verdict.
    """

    SETUP_OK = "setup_line_ok"
    """
    Every setup check passed.
    """

    CHECKS_NEED_SETUP = "setup_line_needs_setup"
    """
    The heading above the indented rows naming each check that needs setup.
    """


def summary_value(output: str, label: str) -> str:
    """
    Extract one line's value from the summary report.

    :param output: session-start.sh's standard output.
    :param label: The summary line's label, such as ``plan``.
    :return: Everything after the label, stripped.
    :raises AssertionError: If the report has no such line.
    """
    prefix = f"  {label}:"
    for line in output.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    raise AssertionError(f"no '{label}' line in this summary report:\n{output}")


def summary_message(message: SummaryMessage, *arguments: str) -> str:
    """
    Render one summary line from the same definitions session-start.sh prints it from,
    so an assertion is never a second copy of the wording.

    :param message: The summary line to render.
    :param arguments: That message's arguments, in order.
    :return: The rendered line.
    """
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; shift; "$@"',
            "_",
            str(MESSAGES_SCRIPT),
            message,
            *arguments,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout
