#!/usr/bin/env python3
"""
Deterministic executor for the stacked-PR maintenance pass.

``stack.py`` derives what a pass should do and prints it; nothing in it moves a commit.
Every fetch, merge, rebase and push in the workflow was therefore performed by a session
following prose, and ``board.json`` was hand-assembled from whatever the caller happened
to fetch - the same class of hand-assembled input that let a dropped ``merged_at`` field
read as a legitimate value.

This module is the command line onto the modules that perform those steps::

    python .claude/stack/maintenance.py board --write     # export the fork's open pull requests
    python .claude/stack/maintenance.py fast-forward      # move the fork's base onto the upstream
    python .claude/stack/maintenance.py restack           # integrate every moved parent, report every conflict
    python .claude/stack/maintenance.py promote           # record the upstream link on every ready branch
    python .claude/stack/maintenance.py run-report --json # the whole pass as one document

It executes an already-derived plan: structure still comes from ``stack.py`` and from
GitHub's own stack object. Retargeting a pull request's **base branch** is the one write
GitHub refuses to the credential this runs on - probed directly, alongside the label,
comment and description writes it does allow - so that step alone is reported for the
caller to perform through the GitHub MCP server.

The exit status is the result. ``run-report --json`` is the machine-readable form, so a
scheduled job with no model in the loop can emit it directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from maintenance_board import MissingPullRequestFieldError
from maintenance_commands import COMMANDS, MaintenancePass
from maintenance_git_commands import GitCommandFailed, GitCommandRunner
from maintenance_github import GitHubCredentialUnavailableError, GitHubRequestFailed
from maintenance_report import MaintenanceExitCode
from stack import (
    AmbiguousForkRemoteError,
    BoardUnavailable,
    ContradictoryLabelWriteError,
    ForkRemoteNotFoundError,
    PromotionLinkTooLongError,
    load_configuration,
)


def _argument_parser() -> argparse.ArgumentParser:
    """:return: The parser, built from the commands rather than from a list of them."""
    parser = argparse.ArgumentParser(
        prog="maintenance.py",
        description="Stacked-PR maintenance: perform the pass, report what happened.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        command.declare_arguments(
            subparsers.add_parser(command.invoked_as, help=command.description)
        )
    return parser


def main() -> MaintenanceExitCode:
    """
    Run the command line and say, in words, what its status means.

    The status itself can only be a number, so the name accompanies it on stderr for
    anything other than a clean run - success stays silent, since announcing it would
    make every run noisy.

    :return: The process exit code.
    """
    status = _dispatch()
    if status is not MaintenanceExitCode.SUCCESS:
        print(
            f"maintenance.py: {status.name_for_a_caller} ({int(status)})",
            file=sys.stderr,
        )
    return status


def _dispatch() -> MaintenanceExitCode:
    """
    Run the requested command, mapping every refusal to its own status.

    :return: The process exit code.
    """
    arguments = _argument_parser().parse_args()
    requested = next(
        entry for entry in COMMANDS if entry.invoked_as == arguments.command
    )
    try:
        maintenance = MaintenancePass(
            configuration=load_configuration(),
            git=GitCommandRunner(working_directory=Path.cwd()),
        )
        return requested.run(maintenance, arguments)
    except (ForkRemoteNotFoundError, AmbiguousForkRemoteError) as error:
        print(f"{error}", file=sys.stderr)
        return MaintenanceExitCode.REMOTES_UNRESOLVED
    except BoardUnavailable as error:
        print(f"{error}", file=sys.stderr)
        return MaintenanceExitCode.BOARD_UNAVAILABLE
    except GitHubCredentialUnavailableError as error:
        print(f"{error}", file=sys.stderr)
        return MaintenanceExitCode.CREDENTIAL_UNAVAILABLE
    except (
        MissingPullRequestFieldError,
        ContradictoryLabelWriteError,
        PromotionLinkTooLongError,
    ) as error:
        print(f"{error}", file=sys.stderr)
        return MaintenanceExitCode.USAGE
    except GitCommandFailed as error:
        print(f"{error}", file=sys.stderr)
        return MaintenanceExitCode.GIT_COMMAND_FAILED
    except GitHubRequestFailed as error:
        print(f"{error}", file=sys.stderr)
        return MaintenanceExitCode.GITHUB_REQUEST_FAILED


if __name__ == "__main__":
    sys.exit(main())
