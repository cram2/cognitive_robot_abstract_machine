#!/usr/bin/env python3
"""
Shared support for plan-updates-since.sh: the CLI option name, every user-facing message
string, and the tracking-issue-comment JSON shape.

Kept as a real, testable module rather than inline ``python3 -c`` snippets in the
shell script (mirrors plan_manifest_tools.py's own reasoning for save-plan.sh), and
defined here exactly once so the shell script prints these strings by calling into
this module instead of embedding its own copy, and the test suite imports the same
constants instead of hardcoding a second copy of them.

Usage:
    python3 plan_updates_since_support.py print-comments < comments.json
    python3 plan_updates_since_support.py print-no-changes-message
    python3 plan_updates_since_support.py print-no-tracking-issue-message
    python3 plan_updates_since_support.py print-no-default-repository-message \\
        <plan-id> <tracking-issue-number>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from enum import StrEnum


class PlanUpdatesSinceOption(StrEnum):
    """
    Command-line options plan-updates-since.sh accepts.
    """

    SINCE = "--since"
    """
    Overrides the recorded stamp with an explicit baseline commit SHA.
    """


class IssueCommentField(StrEnum):
    """
    Field names read from one element of GitHub's issue-comments API response.
    """

    USER = "user"
    LOGIN = "login"
    CREATED_AT = "created_at"
    BODY = "body"


NO_CHANGES_MESSAGE = "No changes."
"""
Printed when the plan directory has no diff since the baseline commit.
"""

NO_NEW_COMMENTS_MESSAGE = "No new comments."
"""
Printed when the tracking issue has no comments newer than the baseline.
"""

NO_TRACKING_ISSUE_MESSAGE = (
    "This plan has no tracking_issue set - nothing to check for new comments."
)
"""
Printed when the plan's manifest has no tracking_issue field.
"""


def no_default_repository_message(plan_id: str, tracking_issue: str) -> str:
    """
    Build the error printed when a plan has a tracking_issue but no default_repository
    to resolve it against.

    :param plan_id: The plan's id.
    :param tracking_issue: The plan's tracking_issue number, as a string.
    :return: The error message.
    """
    return (
        f"Plan '{plan_id}' has tracking_issue set (#{tracking_issue}) but no\n"
        "default_repository - cannot tell which GitHub repository to query."
    )


@dataclass
class IssueComment:
    """
    One comment on a GitHub issue or pull request.
    """

    author_login: str
    """
    The GitHub login of whoever posted the comment.
    """

    created_at: str
    """
    The comment's creation timestamp, as GitHub's API returns it.
    """

    body: str
    """
    The comment's raw text body.
    """

    @classmethod
    def from_api_response(cls, comment: dict) -> IssueComment:
        """
        Build an IssueComment from one element of the GitHub issue-comments API
        response.

        :param comment: One comment object as returned by GitHub's REST API.
        :return: The parsed comment.
        """
        return cls(
            author_login=comment[IssueCommentField.USER][IssueCommentField.LOGIN],
            created_at=comment[IssueCommentField.CREATED_AT],
            body=comment[IssueCommentField.BODY],
        )

    def to_api_response(self) -> dict:
        """
        Render this comment back into the shape GitHub's issue-comments API returns.

        The inverse of from_api_response - used by tests to build stub API responses
        from the same field names, instead of a hand-typed second copy of them.

        :return: The API-shaped comment object.
        """
        return {
            IssueCommentField.USER: {IssueCommentField.LOGIN: self.author_login},
            IssueCommentField.CREATED_AT: self.created_at,
            IssueCommentField.BODY: self.body,
        }

    def formatted(self) -> str:
        """
        Render this comment as a "[<login> @ <created_at>]"-then-body block.

        :return: The formatted text, without a trailing separator.
        """
        return f"[{self.author_login} @ {self.created_at}]\n{self.body}"


def format_issue_comments(comments_json: str) -> str:
    """
    Render a GitHub issue-comments API response as human-readable text.

    :param comments_json: The raw JSON array GitHub's comments endpoint returned.
    :return: One formatted block per comment separated by "---" lines, or
        NO_NEW_COMMENTS_MESSAGE if the array is empty.
    """
    comments = [
        IssueComment.from_api_response(comment) for comment in json.loads(comments_json)
    ]
    if not comments:
        return NO_NEW_COMMENTS_MESSAGE
    return "\n---\n".join(comment.formatted() for comment in comments) + "\n---"


def main() -> int:
    """
    Parse arguments and dispatch to the requested subcommand.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subcommands = parser.add_subparsers(dest="subcommand", required=True)

    subcommands.add_parser(
        "print-comments",
        help="Read a GitHub issue-comments JSON response from stdin and print it",
    )
    subcommands.add_parser("print-no-changes-message")
    subcommands.add_parser("print-no-tracking-issue-message")
    no_default_repository_parser = subcommands.add_parser(
        "print-no-default-repository-message"
    )
    no_default_repository_parser.add_argument("plan_id")
    no_default_repository_parser.add_argument("tracking_issue")

    arguments = parser.parse_args()

    if arguments.subcommand == "print-comments":
        print(format_issue_comments(sys.stdin.read()))
    elif arguments.subcommand == "print-no-changes-message":
        print(NO_CHANGES_MESSAGE)
    elif arguments.subcommand == "print-no-tracking-issue-message":
        print(NO_TRACKING_ISSUE_MESSAGE)
    elif arguments.subcommand == "print-no-default-repository-message":
        print(
            no_default_repository_message(arguments.plan_id, arguments.tracking_issue),
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
