"""
Running git for a pass, and deciding what a pass is allowed to publish.

Every command a pass runs goes through :class:`GitCommandRunner`, and every push it
proposes is built as a :class:`ProposedPush` - so whether published history may be
rewritten is decided in one place rather than at each call site.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from maintenance_errors import ExternalCallFailed
from stack import Configuration, IntegrationStrategy, resolve_ref

# %% running git


@dataclass
class GitCommandFailed(ExternalCallFailed):
    """
    Raised when a git command this module depends on the result of fails.
    """

    arguments: tuple[str, ...] = ()
    """
    The git subcommand and its arguments, as invoked.
    """

    @property
    def call(self) -> str:
        """:return: The git command line, as invoked."""
        return f"git {' '.join(self.arguments)}"


@dataclass(frozen=True)
class GitCommandResult:
    """
    One finished git command, whether or not it succeeded.
    """

    arguments: tuple[str, ...]
    """
    The git subcommand and its arguments, as invoked.
    """

    exit_status: int
    """
    The status git exited with.
    """

    output: str
    """
    Git's stripped stdout.
    """

    error_output: str
    """
    Git's stripped stderr.
    """

    @property
    def succeeded(self) -> bool:
        """:return: Whether git exited zero."""
        return self.exit_status == 0

    def raise_if_failed(self) -> GitCommandResult:
        """:return: This result, when the command succeeded.
        :raises GitCommandFailed: When it did not."""
        if not self.succeeded:
            raise GitCommandFailed(
                status=self.exit_status,
                detail=self.error_output,
                arguments=self.arguments,
            )
        return self


# %% what a pass is allowed to publish


@dataclass(frozen=True)
class ProposedPush:
    """
    One publication, and whether it is authorised to overwrite what is published.

    Every push the executor makes is built here, so whether history may be rewritten is
    decided once rather than at each call.
    """

    remote: str
    """
    The remote to publish to.
    """

    refspec: str
    """
    What to publish, as ``<source>:<destination>``.
    """

    with_lease: bool = False
    """
    Whether published history may be overwritten, and then only if the remote is where
    this checkout last saw it.
    """

    @classmethod
    def publishing(
        cls, configuration: Configuration, branch: str, strategy: IntegrationStrategy
    ) -> ProposedPush:
        """
        Build the push that publishes a restacked branch.

        :param configuration: The resolved configuration.
        :param branch: The branch to publish.
        :param strategy: How its parent was integrated, which is what authorises a
            rewrite - and which ``build_stack`` sets to rebase only from the label.
        :return: The push.
        """
        return cls(
            remote=configuration.fork_remote,
            refspec=f"{branch}:{branch}",
            with_lease=strategy is IntegrationStrategy.REBASE,
        )


@dataclass(frozen=True)
class GitCommandRunner:
    """
    Runs git in one checkout, reporting failures rather than swallowing them.

    ``stack.py`` reads git through a helper that returns an empty string when a command
    fails. That is right for derivation, where a missing ref simply means "no answer",
    and wrong here: a push that silently did nothing would be indistinguishable from one
    that worked.
    """

    working_directory: Path
    """
    The checkout every command runs in.
    """

    def attempt(self, *arguments: str) -> GitCommandResult:
        """
        Run a command whose failure is an expected outcome.

        :param arguments: The git subcommand and its arguments.
        :return: The finished command.
        """
        completed = subprocess.run(
            ["git", *arguments],
            cwd=self.working_directory,
            capture_output=True,
            text=True,
        )
        return GitCommandResult(
            arguments=arguments,
            exit_status=completed.returncode,
            output=completed.stdout.strip(),
            error_output=completed.stderr.strip(),
        )

    def run(self, *arguments: str) -> str:
        """
        Run a command this module depends on the result of.

        :param arguments: The git subcommand and its arguments.
        :return: Git's stripped stdout.
        :raises GitCommandFailed: If git exits non-zero.
        """
        return self.attempt(*arguments).raise_if_failed().output

    def fetch(self, remote: str, *references: str) -> None:
        """
        Refresh what this checkout knows about a remote.

        :param remote: The remote to fetch from.
        :param references: The branches to fetch, all of them when none is named.
        """
        self.run("fetch", "--quiet", remote, *references)

    def commit_at(self, reference: str) -> str:
        """:param reference: Any reference git can resolve.
        :return: The commit it names."""
        return self.run("rev-parse", reference)

    def checkout(self, branch: str, start_point: str) -> None:
        """
        Put a branch at a starting point and check it out.

        :param branch: The branch to move and check out.
        :param start_point: What to point it at.
        """
        self.run("checkout", "--quiet", "-B", branch, start_point)

    def checked_out_branch(self) -> str:
        """:return: The branch whose content a push would move."""
        return self.run("branch", "--show-current")

    def merge(self, reference: str) -> GitCommandResult:
        """:param reference: The reference to merge in.
        :return: The finished merge, whose failure is a conflict only when it left
            unmerged paths behind."""
        return self.attempt("merge", "--no-edit", reference)

    def rebase(self, reference: str) -> GitCommandResult:
        """:param reference: The reference to rebase onto.
        :return: The finished rebase, whose failure is a conflict only when it left
            unmerged paths behind."""
        return self.attempt("rebase", reference)

    def abandon(self, strategy: IntegrationStrategy) -> None:
        """
        Undo whichever integration just failed.

        :param strategy: The integration that was attempted.
        """
        self.attempt(
            "rebase" if strategy is IntegrationStrategy.REBASE else "merge", "--abort"
        )

    def unmerged_paths(self) -> tuple[str, ...]:
        """:return: The paths the integration that just failed left conflicted."""
        unmerged = self.attempt("diff", "--name-only", "--diff-filter=U")
        return tuple(path for path in unmerged.output.splitlines() if path)

    def push(self, proposed: ProposedPush) -> GitCommandResult:
        """
        Publish a refspec, forcing only where the push itself says it is authorised.

        :param proposed: What to publish, and whether a rewrite is authorised.
        :return: The finished push, whose failure the caller reports rather than forces.
        """
        lease = ["--force-with-lease"] if proposed.with_lease else []
        return self.attempt(
            "push", "--quiet", *lease, proposed.remote, proposed.refspec
        )

    def contains(self, candidate: str, descendant: str) -> bool:
        """:param candidate: The reference that may be contained.
        :param descendant: The reference that may contain it.
        :return: Whether *candidate* is an ancestor of *descendant*."""
        return self.attempt(
            "merge-base", "--is-ancestor", candidate, descendant
        ).succeeded


# %% asking git what contains what


@dataclass(frozen=True)
class BranchAncestry:
    """
    Answers containment questions about the fork's branches.

    :class:`CommitMoveChecks` asks its false-merge question through this, so the
    question is asked of git rather than of anything this module remembers.
    """

    configuration: Configuration
    """
    The resolved configuration naming the fork remote.
    """

    git: GitCommandRunner
    """
    The runner to ask git through.
    """

    def is_ancestor(self, candidate: str, descendant: str) -> bool:
        """:param candidate: A fork branch that may be contained.
        :param descendant: A local branch that may contain it.
        :return: Whether the fork's copy of *candidate* is contained in *descendant*."""
        return self.git.contains(resolve_ref(self.configuration, candidate), descendant)
