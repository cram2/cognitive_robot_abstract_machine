"""
Recording the upstream link on every branch ready to be promoted.

The upstream pull request is not opened here - the credential has no write access there.
What is written is the link that opens it prefilled, into the fork pull request's own
description, plus the label that stops a later pass rebuilding it.
"""

from __future__ import annotations

from dataclasses import dataclass

from maintenance_constants import PROMOTION_HEADING, PROMOTION_LINK_LABEL
from maintenance_board import PullRequestField
from maintenance_github import ForkPullRequests, PullRequestWriter
from stack import (
    BranchStatus,
    LabelWrite,
    PromotionLink,
    Stack,
    promotion_order,
)


@dataclass(frozen=True)
class Promotion:
    """
    One branch's compare-and-create link, and where it was recorded.
    """

    branch: str
    """
    The branch promoted.
    """

    pull_request_number: int
    """
    Its fork pull request.
    """

    url: str
    """
    The compare-and-create link opening the upstream pull request.
    """

    body_was_truncated: bool
    """
    Whether the prefilled description had to be shortened to fit the URL limit.
    """


def description_with_promotion_link(description: str, url: str) -> str:
    """
    Put a promotion link into a description, replacing any already there.

    :param description: The pull request's current description.
    :param url: The link to record.
    :return: The description to write back.
    """
    before, _, _ = description.partition(PROMOTION_HEADING)
    return f"{before.rstrip()}\n\n{PROMOTION_HEADING}\n\n{url}\n"


def promotion_summary(description: str) -> str:
    """
    Take the one paragraph of a description that prefills the upstream pull request.

    A compare URL discards an over-long prefill silently, so the whole description is
    never sent - the link back to the fork pull request carries the rest.

    :param description: The fork pull request's description.
    :return: Its first paragraph, empty if it has none.
    """
    before, _, _ = description.partition(PROMOTION_HEADING)
    paragraphs = [block.strip() for block in before.split("\n\n") if block.strip()]
    return paragraphs[0] if paragraphs else ""


def promote(stack: Stack, fork: ForkPullRequests) -> list[Promotion]:
    """
    Build and record the upstream link for every branch ready to be promoted.

    The upstream pull request is not opened here - the app has no write access there, so
    that call fails every time. What is written is the link that opens it prefilled, into
    the fork pull request's own description, plus the label stopping a later pass
    rebuilding it. The ``in-review`` label stays the developer's to add, since the
    upstream pull request does not exist until they click Create.

    Both the decision and the label write read the labels the branch carries *now*, not
    the ones the board was exported with: the restack runs between those two moments and
    withholds a branch by labelling it, so a snapshot is already out of date here.

    :param stack: The derived stack.
    :param fork: The fork to read descriptions from and write links back to.
    :return: One entry per branch promoted, in dependency order.
    """
    promoted: list[Promotion] = []
    withheld = stack.configuration.needs_resolution_label
    for branch in promotion_order(stack):
        number = branch.pull_request_number
        pull_request = fork.pull_request(number)
        labels = PullRequestField.LABELS.read(pull_request, number)
        if PROMOTION_LINK_LABEL in labels or withheld in labels:
            continue
        description = str(PullRequestField.BODY.read(pull_request, number) or "")
        link = PromotionLink.build(
            stack.configuration,
            branch.name,
            str(PullRequestField.TITLE.read(pull_request, number) or branch.name),
            _prefilled_description(description, number, stack),
        )
        fork.set_description(
            number,
            description_with_promotion_link(description, link.url),
        )
        fork.replace_labels(
            number,
            LabelWrite.replacing(labels, added=[PROMOTION_LINK_LABEL]).labels,
        )
        promoted.append(
            Promotion(
                branch=branch.name,
                pull_request_number=branch.pull_request_number,
                url=link.url,
                body_was_truncated=link.body_was_truncated,
            )
        )
    return promoted


def _prefilled_description(
    description: str, pull_request_number: int, stack: Stack
) -> str:
    """
    Build what the upstream pull request opens with.

    :param description: The fork pull request's description.
    :param pull_request_number: The fork pull request, to link back to.
    :param stack: The derived stack, naming the fork.
    :return: One paragraph plus a link back to the full detail.
    """
    summary = promotion_summary(description)
    detail = (
        f"Full detail: https://github.com/{stack.configuration.fork_repository}"
        f"/pull/{pull_request_number}"
    )
    return f"{summary}\n\n{detail}" if summary else detail


def clear_spent_promotion_labels(
    stack: Stack, fork: PullRequestWriter
) -> tuple[str, ...]:
    """
    Drop the link label from every branch whose link has already been acted on.

    :param stack: The derived stack.
    :param fork: The fork to write to.
    :return: The branches whose label was cleared.
    """
    spent = [
        branch
        for branch in stack.branches
        if PROMOTION_LINK_LABEL in branch.labels
        and branch.status in {BranchStatus.IN_REVIEW, BranchStatus.MERGED}
    ]
    for branch in spent:
        fork.replace_labels(
            branch.pull_request_number,
            LabelWrite.replacing(branch.labels, removed=[PROMOTION_LINK_LABEL]).labels,
        )
    return tuple(branch.name for branch in spent)
