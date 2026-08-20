"""
Moving the fork's copy of the upstream base onto the upstream's tip.

This is the step that closes the pull requests whose work has landed, since GitHub marks
one merged the moment its head becomes an ancestor of its base.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from maintenance_git_commands import GitCommandRunner, ProposedPush
from stack import Configuration, resolve_ref


class FastForwardOutcome(StrEnum):
    """
    What became of the fork's base branch.
    """

    PUSHED = "pushed"
    """
    It was moved onto the upstream's tip.
    """

    ALREADY_CURRENT = "already-current"
    """
    It already pointed at the upstream's tip.
    """

    REFUSED_NOT_FAST_FORWARD = "refused-not-fast-forward"
    """
    It carries commits the upstream does not, so moving it would discard them.
    """


@dataclass(frozen=True)
class FastForwardReport:
    """
    What the fast-forward did, and to what.
    """

    outcome: FastForwardOutcome
    """
    What became of the fork's base branch.
    """

    upstream_reference: str
    """
    The upstream ref the fork's base was compared against.
    """

    fork_reference: str
    """
    The fork ref that was to be moved.
    """

    commit: str
    """
    The commit the fork's base points at now.
    """

    explanation: str | None = None
    """
    Why a refusal was refused, absent when nothing was refused.
    """


def fast_forward(
    configuration: Configuration, git: GitCommandRunner
) -> FastForwardReport:
    """
    Move the fork's copy of the upstream base onto the upstream's tip.

    This is what closes the pull requests whose work has landed: GitHub marks one merged
    the moment its head becomes an ancestor of its base. A move that is not a
    fast-forward is refused rather than forced - the fork's base is a mirror of the
    upstream trunk, and anything else on it would flow into every branch above.

    :param configuration: The resolved configuration.
    :param git: The runner to execute through.
    :return: What was done.
    """
    upstream_reference = (
        f"{configuration.upstream_remote}/{configuration.upstream_base}"
    )
    fork_reference = resolve_ref(configuration, configuration.upstream_base)
    git.fetch(configuration.upstream_remote, configuration.upstream_base)
    git.fetch(configuration.fork_remote, configuration.upstream_base)
    upstream_commit = git.commit_at(upstream_reference)
    fork_commit = git.commit_at(fork_reference)

    if upstream_commit == fork_commit:
        return FastForwardReport(
            FastForwardOutcome.ALREADY_CURRENT,
            upstream_reference,
            fork_reference,
            fork_commit,
        )
    if not git.contains(fork_commit, upstream_commit):
        return FastForwardReport(
            FastForwardOutcome.REFUSED_NOT_FAST_FORWARD,
            upstream_reference,
            fork_reference,
            fork_commit,
            explanation=(
                f"'{fork_reference}' is not contained in '{upstream_reference}', so "
                f"moving it would discard commits; resolve this by hand rather than "
                f"forcing"
            ),
        )
    git.push(
        ProposedPush(
            remote=configuration.fork_remote,
            refspec=f"{upstream_commit}:refs/heads/{configuration.upstream_base}",
        )
    ).raise_if_failed()
    git.fetch(configuration.fork_remote, configuration.upstream_base)
    return FastForwardReport(
        FastForwardOutcome.PUSHED,
        upstream_reference,
        fork_reference,
        upstream_commit,
    )
