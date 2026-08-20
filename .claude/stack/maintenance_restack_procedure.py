"""
The restack itself: which steps a branch is put through, in which order, and where.

The order is the procedure, so it is stated here rather than discovered - a branch is
published only once its move has been checked. The steps run in a worktree of their own
so that switching branches never takes the pass's own tooling away from it.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from maintenance_git_commands import BranchAncestry, GitCommandRunner
from maintenance_github import ForkPullRequests
from maintenance_restack_steps import (
    BranchOutcome,
    BranchUnderRestack,
    IntegrateParent,
    PublishBranch,
    RefuseAnUnsafeMove,
    RestackStep,
    SkipBranchAlreadyCurrent,
    WithholdBranchStillConflicting,
)
from stack import CommitMoveChecks, IntegrationStrategy, Stack, restack_plan

if TYPE_CHECKING:
    from types import TracebackType


# %% the order that is the procedure

RESTACK_STEPS: tuple[RestackStep, ...] = (
    WithholdBranchStillConflicting(),
    SkipBranchAlreadyCurrent(),
    IntegrateParent(),
    RefuseAnUnsafeMove(),
    PublishBranch(),
)
"""
Every step a branch is put through, in the order that is the procedure.

Unlike :data:`maintenance_commands.COMMANDS`, these are listed rather than found from
their own subclasses: a branch is published only once its move has been checked, so this
order is a decision about what a pass does, not bookkeeping. Stating it here keeps it
where it is read, rather than making it a consequence of where the classes happen to be
defined.
"""


@dataclass
class RestackConcludedNothingError(RuntimeError):
    """
    Raised when no step concluded a branch, which the last step always must.
    """

    branch: str
    """
    The branch left without an outcome.
    """

    def __str__(self) -> str:
        """:return: Which branch was left unconcluded."""
        return f"no restack step concluded '{self.branch}'"


# %% where the branch switching happens


@dataclass(frozen=True)
class DetachedCheckout:
    """
    The invoking checkout, detached so its branch can be restacked elsewhere.

    git refuses to check one branch out in two worktrees at once, and the caller of a
    pass is usually sitting on a branch of the stack. Detaching releases the name while
    changing nothing else - same commit, same files, same work in progress - and the
    branch is checked out again afterwards, which is also how the caller picks up a
    restack of their own branch.
    """

    git: GitCommandRunner
    """
    The invoking checkout.
    """

    branch: str
    """
    The branch it was on, empty when it was already detached.
    """

    @classmethod
    def of(cls, git: GitCommandRunner) -> DetachedCheckout:
        """:param git: The checkout to detach.
        :return: The detachment, to be used as a context manager so it is undone."""
        return cls(git, git.checked_out_branch())

    def __enter__(self) -> DetachedCheckout:
        """:return: This detachment, once the checkout is off its branch."""
        if self.branch:
            self.git.run("checkout", "--quiet", "--detach")
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Put the checkout back on the branch it was on.

        Attempted rather than depended on, so failing to restore never replaces the
        exception on its way out; the checkout is then left detached at the commit it
        started from, with everything it was carrying.
        """
        if self.branch:
            self.git.attempt("checkout", "--quiet", self.branch)


@dataclass(frozen=True)
class RestackWorktree:
    """
    A checkout of its own for the branch switching a restack does.

    Every step of the pass shells out to this file, which is tracked content in the
    checkout the pass is invoked from. Most branches in a stack were cut before that
    tooling landed, so checking one out there deletes the tooling the rest of the pass
    needs and leaves the caller on a branch that is not theirs. This worktree is added
    outside the project instead, out of reach of the branches a restack switches to.

    Its refs are the same refs, so a branch it moves is moved for the whole repository.
    """

    git: GitCommandRunner
    """
    The runner every branch switch of a restack goes through.
    """

    origin: GitCommandRunner
    """
    The invoking checkout, which the worktree is added to and removed from.
    """

    @classmethod
    def added_to(cls, origin: GitCommandRunner) -> RestackWorktree:
        """
        Add a worktree, detached at whatever the invoking checkout has.

        :param origin: The checkout to add it to.
        :return: The worktree, to be used as a context manager so it is removed again.
        """
        path = Path(tempfile.mkdtemp(prefix="stack-restack-"))
        origin.run("worktree", "add", "--quiet", "--detach", str(path), "HEAD")
        return cls(GitCommandRunner(working_directory=path), origin)

    def __enter__(self) -> GitCommandRunner:
        """:return: The runner to restack through."""
        return self.git

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Remove the worktree, whether the restack finished or was abandoned.

        Removal is attempted rather than depended on, so a failure to tidy up never
        replaces the exception that is on its way out.
        """
        self.origin.attempt(
            "worktree", "remove", "--force", str(self.git.working_directory)
        )


# %% the restack itself


def restack(
    stack: Stack, git: GitCommandRunner, fork: ForkPullRequests
) -> list[BranchOutcome]:
    """
    Put every branch whose parent moved through :data:`RESTACK_STEPS`, bottom up.

    The steps run in a :class:`RestackWorktree` rather than in the invoking checkout,
    which lends its branch through a :class:`DetachedCheckout` and gets it back with its
    own files still in place. The worktree goes first so it is gone before the branch is
    wanted again.

    :param stack: The derived stack, whose plan this executes.
    :param git: The runner naming the checkout to add the worktree to.
    :param fork: The fork, read for conflict state and written to when reporting.
    :return: One outcome per branch in the plan, parent before child.
    """
    with DetachedCheckout.of(git), RestackWorktree.added_to(git) as switching:
        checks = CommitMoveChecks(
            stack=stack,
            checked_out_branch="",
            is_ancestor=BranchAncestry(stack.configuration, switching).is_ancestor,
        )
        by_name = {branch.name: branch for branch in stack.branches}
        return [
            _restack_branch(
                BranchUnderRestack(
                    branch=by_name[entry["branch"]],
                    parent=entry["parent"],
                    strategy=IntegrationStrategy(entry["strategy"]),
                    stack=stack,
                    git=switching,
                    fork=fork,
                    checks=checks,
                )
            )
            for entry in restack_plan(stack)
        ]


def _restack_branch(restacking: BranchUnderRestack) -> BranchOutcome:
    """:param restacking: The branch to restack.
    :return: The outcome of the first step that concluded it.
    :raises RestackConcludedNothingError: If no step did."""
    for step in RESTACK_STEPS:
        outcome = step.attempt(restacking)
        if outcome is not None:
            return outcome
    raise RestackConcludedNothingError(restacking.branch.name)
