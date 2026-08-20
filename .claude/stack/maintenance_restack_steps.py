"""
What a restack does to one branch, one step at a time.

A step either concludes the branch - returning the outcome its owner acts on - or lets
the next one run. The order they run in is the procedure itself, and lives with the pass
that runs them rather than here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from maintenance_constants import (
    CONFLICT_COMMENT_PREFIX,
    MERGEABLE_STATE_WITH_CONFLICTS,
)
from maintenance_board import PullRequestField
from maintenance_git_commands import GitCommandRunner, ProposedPush
from maintenance_github import ForkPullRequests
from stack import (
    Branch,
    CommitMoveAction,
    CommitMoveChecks,
    Configuration,
    IntegrationStrategy,
    LabelWrite,
    ProposedCommitMove,
    RefusalReason,
    Stack,
    resolve_ref,
)

# %% what became of one branch


class RestackOutcome(StrEnum):
    """
    What became of one branch during a restack.
    """

    PUSHED = "pushed"
    """
    Its parent was integrated and the result published.
    """

    UP_TO_DATE = "up-to-date"
    """
    Its parent's tip was already contained in it.
    """

    CONFLICT = "conflict"
    """
    Its parent could not be integrated cleanly; nothing was published.
    """

    INTEGRATION_FAILED = "integration-failed"
    """
    Integrating its parent failed without conflicting on anything, so the branch is not
    the thing to fix and its owner was not told; nothing was published.
    """

    REFUSED = "refused"
    """
    Move check refused the push; nothing was published.
    """

    PUSH_REJECTED = "push-rejected"
    """
    The fork rejected the push, so the branch moved under this pass; nothing was
    published, and nothing was forced over whatever moved it.
    """

    WITHHELD = "withheld"
    """
    It is still conflicted against its base from a previous pass, so it was left
    untouched rather than re-reported.
    """


@dataclass(frozen=True)
class BranchOutcome:
    """
    What became of one branch, in terms its owner can act on.
    """

    branch: str
    """
    The branch this is about.
    """

    parent: str
    """
    The branch whose tip was to be integrated into it.
    """

    strategy: IntegrationStrategy
    """
    How the parent was to be integrated.
    """

    outcome: RestackOutcome
    """
    What became of it.
    """

    conflicting_paths: tuple[str, ...] = ()
    """
    The paths that conflicted, empty unless the outcome is a conflict.
    """

    refusals: tuple[RefusalReason, ...] = ()
    """
    Why the push was refused, empty unless the outcome is a refusal.
    """

    pushed_commit: str | None = None
    """
    The commit published, absent unless the outcome is a push.
    """

    explanation: str | None = None
    """
    Why this outcome happened in words its owner can act on, absent unless the outcome
    carries one.
    """

    reported_at: str | None = None
    """
    URL of the comment telling this branch's owner about it, absent unless one was
    posted.
    """


def conflict_report(
    branch: Branch, conflicting_paths: Sequence[str], parent: str
) -> str:
    """
    Write the comment telling a branch's owner that their branch needs them.

    :param branch: The branch that could not be integrated.
    :param conflicting_paths: The paths that conflicted.
    :param parent: The branch whose tip was being integrated.
    :return: The comment body.
    """
    files = "\n".join(f"- `{path}`" for path in conflicting_paths)
    addressed = (
        f"\n\n{branch.session}"
        if branch.session
        else "\n\nThis pull request's description names no session to address."
    )
    return (
        f"{CONFLICT_COMMENT_PREFIX} integrating `{parent}` into `{branch.name}` "
        f"conflicts, so this branch was left untouched and skipped.\n\n"
        f"Conflicting files:\n{files}\n\n"
        f"Please resolve and push. This branch is labelled "
        f"`needs-resolution` so later passes skip it rather than re-reporting the same "
        f"conflict; the label is cleared automatically once it merges cleanly again, "
        f"and the branch rejoins the pass.{addressed}"
    )


@dataclass(frozen=True)
class BranchUnderRestack:
    """
    One branch's restack, and everything a step needs to carry it out.
    """

    branch: Branch
    """
    The branch being restacked.
    """

    parent: str
    """
    The branch whose tip is to be integrated into it.
    """

    strategy: IntegrationStrategy
    """
    How that parent is to be integrated, which is also what authorises a rewrite.
    """

    stack: Stack
    """
    The derived stack it belongs to.
    """

    git: GitCommandRunner
    """
    The runner to execute through.
    """

    fork: ForkPullRequests
    """
    The fork, read for conflict state and written to when reporting.
    """

    checks: CommitMoveChecks
    """
    The checks its push is put through.
    """

    @property
    def configuration(self) -> Configuration:
        """:return: The resolved configuration."""
        return self.stack.configuration

    @property
    def branch_reference(self) -> str:
        """:return: The fork's copy of this branch, which every step starts from."""
        return resolve_ref(self.configuration, self.branch.name)

    @property
    def parent_reference(self) -> str:
        """:return: The fork's copy of the parent being integrated."""
        return resolve_ref(self.configuration, self.parent)

    def concluded(self, outcome: RestackOutcome, **detail: Any) -> BranchOutcome:
        """
        Finish this branch with an outcome its owner can act on.

        :param outcome: What became of it.
        :param detail: Whatever that outcome carries.
        :return: The outcome, naming this branch and its parent.
        """
        return BranchOutcome(
            self.branch.name, self.parent, self.strategy, outcome, **detail
        )


# %% the steps themselves


@dataclass(frozen=True)
class RestackStep(ABC):
    """
    One step of a branch's restack.

    A step either concludes the branch - returning the outcome its owner acts on - or
    returns nothing and lets the next step run. Adding a step is writing a subclass and
    placing it in :data:`maintenance_restack_procedure.RESTACK_STEPS`, whose order is the procedure.
    """

    @abstractmethod
    def attempt(self, restacking: BranchUnderRestack) -> BranchOutcome | None:
        """
        Carry out this step.

        :param restacking: The branch being restacked.
        :return: The outcome concluding the branch, or ``None`` to continue.
        """


@dataclass(frozen=True)
class WithholdBranchStillConflicting(RestackStep):
    """
    Leaves a branch alone while it is still conflicted from an earlier pass.

    Clears the label as a side effect when it is not, since that is what lets the branch
    rejoin the pass without anybody remembering to remove it by hand.
    """

    def attempt(self, restacking: BranchUnderRestack) -> BranchOutcome | None:
        """:param restacking: The branch being restacked.
        :return: A withheld outcome while it still conflicts, otherwise ``None``."""
        branch = restacking.branch
        label = restacking.configuration.needs_resolution_label
        if label not in branch.labels:
            return None
        state = PullRequestField.MERGEABLE_STATE.read(
            restacking.fork.pull_request(branch.pull_request_number),
            branch.pull_request_number,
        )
        if state == MERGEABLE_STATE_WITH_CONFLICTS:
            return restacking.concluded(
                RestackOutcome.WITHHELD,
                explanation="still conflicted against its base since a previous pass",
            )
        restacking.fork.replace_labels(
            branch.pull_request_number,
            LabelWrite.replacing(branch.labels, removed=[label]).labels,
        )
        return None


@dataclass(frozen=True)
class SkipBranchAlreadyCurrent(RestackStep):
    """
    Leaves a branch alone when its parent's tip is already contained in it.
    """

    def attempt(self, restacking: BranchUnderRestack) -> BranchOutcome | None:
        """:param restacking: The branch being restacked.
        :return: An up-to-date outcome when nothing has to move, otherwise ``None``."""
        if restacking.git.contains(
            restacking.parent_reference, restacking.branch_reference
        ):
            return restacking.concluded(RestackOutcome.UP_TO_DATE)
        return None


@dataclass(frozen=True)
class IntegrateParent(RestackStep):
    """
    Integrates the parent's tip, reporting a conflict to the branch's owner.

    A conflict is never resolved here - that is a change to somebody else's branch. It
    is labelled and commented on, so the next pass withholds the branch rather than
    re-reporting it.

    Unmerged paths are what make a failed integration a conflict, not its exit status:
    a merge also refuses when an untracked file is in the way, when the histories are
    unrelated, or when a reference does not resolve. Labelling those would name a
    branch that merges perfectly well, and the branch's owner would have nothing to
    fix - so they are reported as a plain failure of the pass instead.
    """

    def attempt(self, restacking: BranchUnderRestack) -> BranchOutcome | None:
        """:param restacking: The branch being restacked.
        :return: A conflict outcome when the parent left unmerged paths, a failure
            outcome when the integration failed without any, otherwise ``None``."""
        git = restacking.git
        git.checkout(restacking.branch.name, restacking.branch_reference)
        integration = (
            git.rebase(restacking.parent_reference)
            if restacking.strategy is IntegrationStrategy.REBASE
            else git.merge(restacking.parent_reference)
        )
        if integration.succeeded:
            return None
        conflicting = git.unmerged_paths()
        git.abandon(restacking.strategy)
        if not conflicting:
            return restacking.concluded(
                RestackOutcome.INTEGRATION_FAILED,
                explanation=integration.error_output,
            )
        return restacking.concluded(
            RestackOutcome.CONFLICT,
            conflicting_paths=conflicting,
            reported_at=self._report(restacking, conflicting),
        )

    @staticmethod
    def _report(
        restacking: BranchUnderRestack, conflicting_paths: Sequence[str]
    ) -> str:
        """
        Tell the branch's owner, and label it so the next pass withholds it.

        :param restacking: The branch being restacked.
        :param conflicting_paths: The paths that conflicted.
        :return: The URL of the comment posted.
        """
        branch = restacking.branch
        restacking.fork.replace_labels(
            branch.pull_request_number,
            LabelWrite.replacing(
                branch.labels,
                added=[restacking.configuration.needs_resolution_label],
            ).labels,
        )
        return restacking.fork.add_comment(
            branch.pull_request_number,
            conflict_report(branch, conflicting_paths, restacking.parent),
        )


@dataclass(frozen=True)
class RefuseAnUnsafeMove(RestackStep):
    """
    Puts the push through the checks before it is made, without exception.
    """

    def attempt(self, restacking: BranchUnderRestack) -> BranchOutcome | None:
        """:param restacking: The branch being restacked.
        :return: A refused outcome carrying every reason, otherwise ``None``."""
        checks = CommitMoveChecks(
            stack=restacking.checks.stack,
            checked_out_branch=restacking.git.checked_out_branch(),
            is_ancestor=restacking.checks.is_ancestor,
        )
        refusals = tuple(
            refusal.reason
            for refusal in checks.refusals(
                ProposedCommitMove(
                    action=CommitMoveAction.RESTACK,
                    source=restacking.branch.name,
                    destination=restacking.branch.name,
                    destination_remote=restacking.configuration.fork_remote,
                )
            )
        )
        if refusals:
            return restacking.concluded(RestackOutcome.REFUSED, refusals=refusals)
        return None


@dataclass(frozen=True)
class PublishBranch(RestackStep):
    """
    Publishes the integrated branch, reporting rather than forcing a rejection.
    """

    def attempt(self, restacking: BranchUnderRestack) -> BranchOutcome:
        """:param restacking: The branch being restacked.
        :return: What became of the push - this step always concludes the branch."""
        git = restacking.git
        push = git.push(
            ProposedPush.publishing(
                restacking.configuration, restacking.branch.name, restacking.strategy
            )
        )
        if not push.succeeded:
            return restacking.concluded(
                RestackOutcome.PUSH_REJECTED, explanation=push.error_output
            )
        git.fetch(restacking.configuration.fork_remote, restacking.branch.name)
        return restacking.concluded(
            RestackOutcome.PUSHED, pushed_commit=git.commit_at("HEAD")
        )
