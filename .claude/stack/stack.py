#!/usr/bin/env python3
"""Stacked-PR helper for the fork-staging / cram2-review workflow.

GitHub is the single source of truth. The stack is **not** declared in a ledger: it is read from a
``board.json`` export of the fork's pull requests, combined with plain ``git``:

  * dependency tree = each fork PR's **base branch** (base = parent);
  * ``draft`` <-> ``ready`` = the fork PR's draft flag;
  * ``in-review`` = the ``in_review_label`` on the fork PR (cram2 is not readable from the cloud);
  * ``merged`` = the branch is an ancestor of ``<upstream_remote>/<upstream_base>``.

``stack.toml`` carries the committed defaults (label names, the upstream repository); a
``.claude/personal/stack.toml`` on the personal-notes branch, if present, layers per-user overrides on
top of them (see :func:`load_configuration`).

Commands (run from the repo root; ``--help`` on any of them for its flags)::

    python .claude/stack/stack.py status         # the whole stack: parent, state, drift
    python .claude/stack/stack.py check          # would each branch merge cleanly onto its parent now?
    python .claude/stack/stack.py next           # which branches to submit upstream next
    python .claude/stack/stack.py next --porcelain    # machine-readable: one 'name<TAB>pr' line per branch
    python .claude/stack/stack.py restack-plan   # bottom-up restack plan as JSON
    python .claude/stack/stack.py configuration  # every resolved setting, including the remotes
    python .claude/stack/stack.py labels         # the complete label set a write must send
    python .claude/stack/stack.py preflight      # may these commits move onto that branch?
    python .claude/stack/stack.py promotion-link # the upstream compare-and-create URL for a branch
    python .claude/stack/stack.py reparents      # children whose base has landed, and the base they need
    python .claude/stack/stack.py landed         # open fork pull requests whose branch has landed

The last five exist so the steps most easily got wrong by hand are computed rather than recalled: a
label write replaces the whole set, a push whose two sides name different branches moves the wrong
commits, an unencoded compare URL loses its prefill, and a landed parent is decided by git ancestry
rather than by pull-request state. ``landed`` reports only - GitHub closes a pull request as merged
by itself once its head is contained in its base, so nothing here has to close one.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import ClassVar
from urllib.parse import quote

# %% configuration

CONFIGURATION_PATH = Path(__file__).with_name("stack.toml")
BOARD_PATH = Path(__file__).with_name("board.json")

PERSONAL_STACK_CONFIGURATION_PATH = ".claude/personal/stack.toml"
"""Path, relative to the project root, of the per-user configuration override file on the personal-notes
branch (see :func:`_personal_configuration_overrides`)."""


@dataclass
class MalformedRepositoryError(ValueError):
    """Raised when a repository reference is not in ``owner/name`` form."""

    text: str
    """The value that could not be parsed."""

    def __str__(self) -> str:
        """:return: What was expected and what arrived instead."""
        return f"expected a repository as 'owner/name', got {self.text!r}"


@dataclass(frozen=True)
class Repository:
    """A GitHub repository, identified the way GitHub itself writes it."""

    owner: str
    """The user or organization the repository belongs to."""

    name: str
    """The repository's own name."""

    @classmethod
    def parse(cls, text: str) -> Repository:
        """Parse an ``owner/name`` repository reference.

        :param text: The reference to parse.
        :return: The parsed repository.
        :raises MalformedRepositoryError: If *text* is not ``owner/name``.
        """
        owner, separator, name = text.partition("/")
        if not (owner and separator and name):
            raise MalformedRepositoryError(text)
        return cls(owner, name)

    @staticmethod
    def _remote_url_segments(url: str) -> list[str]:
        """Split a remote URL into its path segments, discarding scheme and host.

        :param url: The remote URL to split.
        :return: The path segments, which name a repository when there are two or more.
        """
        reference = url.removesuffix(".git").rstrip("/")
        if "://" in reference:
            _, _, host_and_path = reference.partition("://")
            _, _, path = host_and_path.partition("/")
        elif ":" in reference:
            _, _, path = reference.rpartition(":")
        else:
            return []
        return [segment for segment in path.split("/") if segment]

    @classmethod
    def names_a_repository(cls, url: str) -> bool:
        """Test whether a remote URL points at a repository at all.

        :param url: The remote URL to test.
        :return: Whether it names an ``owner/name`` pair.
        """
        return len(cls._remote_url_segments(url)) >= 2

    @classmethod
    def from_remote_url(cls, url: str) -> Repository:
        """Read the repository a git remote URL points at.

        Accepts every form a fork remote takes - HTTPS, SSH, and the local proxy a cloud
        session is given - by discarding the host and taking the last two path segments.

        :param url: The remote URL to read.
        :return: The repository it names.
        :raises MalformedRepositoryError: If *url* names no ``owner/name`` pair.
        """
        segments = cls._remote_url_segments(url)
        if len(segments) < 2:
            raise MalformedRepositoryError(url)
        return cls.parse("/".join(segments[-2:]))

    def __str__(self) -> str:
        """:return: The ``owner/name`` form GitHub uses."""
        return f"{self.owner}/{self.name}"


@dataclass(frozen=True)
class Remote:
    """A git remote, identified by the repository its URL names rather than by its name."""

    name: str
    """What this checkout calls the remote."""

    repository: Repository
    """The repository the remote points at."""


@dataclass
class ForkRemoteNotFoundError(LookupError):
    """Raised when no remote points at a repository other than the upstream."""

    upstream_repository: Repository
    """The upstream every candidate turned out to be."""

    def __str__(self) -> str:
        """:return: What was searched for and why nothing qualified."""
        return (
            f"no remote points at a fork: every remote is {self.upstream_repository}. "
            f"Add a remote for your fork, or set fork_repository in stack.toml."
        )


@dataclass
class AmbiguousForkRemoteError(LookupError):
    """Raised when several remotes could each be the fork."""

    candidates: tuple[Remote, ...]
    """The remotes that are not the upstream, in the order git reported them."""

    def __str__(self) -> str:
        """:return: The candidates and how to disambiguate them."""
        listed = ", ".join(
            f"{remote.name} -> {remote.repository}" for remote in self.candidates
        )
        return (
            f"several remotes could be the fork ({listed}). "
            f"Set fork_repository in stack.toml to say which."
        )


@dataclass(frozen=True)
class RemoteResolution:
    """Which remote is the fork and which is the upstream, decided without trusting names."""

    fork: Remote
    """The remote holding the stack."""

    upstream: Remote | None
    """The remote for the upstream review repository, absent if the checkout has none."""

    upstream_repository: Repository
    """The upstream, whether or not a remote points at it yet."""

    preferred_upstream_name: str
    """What to call the upstream remote when one has to be added."""

    @property
    def upstream_name(self) -> str:
        """:return: The upstream remote's name, or the name it will get when added."""
        return self.upstream.name if self.upstream else self.preferred_upstream_name

    @property
    def upstream_setup_command(self) -> str | None:
        """:return: The command adding the missing upstream remote, or ``None`` if present."""
        if self.upstream:
            return None
        return (
            f"git remote add {self.preferred_upstream_name} "
            f"https://github.com/{self.upstream_repository}.git"
        )


def resolve_remotes(
    remote_urls: Mapping[str, str],
    upstream_repository: Repository,
    preferred_upstream_name: str,
    fork_repository: Repository | None = None,
) -> RemoteResolution:
    """Decide which remote is the fork and which is the upstream.

    Remotes are matched by the repository their URL names, so a checkout whose remotes are
    called anything at all resolves the same way.

    :param remote_urls: Remote name to URL, as git reports them.
    :param upstream_repository: The repository every fork is forked from.
    :param preferred_upstream_name: What to call the upstream remote if one must be added.
    :param fork_repository: The fork, when configuration names it outright.
    :return: The resolved remotes.
    :raises ForkRemoteNotFoundError: If no remote points at a fork.
    :raises AmbiguousForkRemoteError: If several do and configuration does not disambiguate.
    """
    remotes = [
        Remote(name, Repository.from_remote_url(url))
        for name, url in remote_urls.items()
        if Repository.names_a_repository(url)
    ]
    upstream = next(
        (remote for remote in remotes if remote.repository == upstream_repository), None
    )
    candidates = tuple(
        remote for remote in remotes if remote.repository != upstream_repository
    )
    return RemoteResolution(
        fork=_select_fork(candidates, fork_repository, upstream_repository),
        upstream=upstream,
        upstream_repository=upstream_repository,
        preferred_upstream_name=preferred_upstream_name,
    )


def _select_fork(
    candidates: tuple[Remote, ...],
    fork_repository: Repository | None,
    upstream_repository: Repository,
) -> Remote:
    """Pick the fork from the remotes that are not the upstream.

    :param candidates: The non-upstream remotes.
    :param fork_repository: The fork, when configuration names it outright.
    :param upstream_repository: The upstream, for reporting when nothing qualifies.
    :return: The fork's remote.
    :raises ForkRemoteNotFoundError: If no candidate qualifies.
    :raises AmbiguousForkRemoteError: If several do and configuration does not disambiguate.
    """
    if fork_repository:
        named = [
            remote for remote in candidates if remote.repository == fork_repository
        ]
        if not named:
            raise ForkRemoteNotFoundError(upstream_repository)
        return named[0]
    if not candidates:
        raise ForkRemoteNotFoundError(upstream_repository)
    if len(candidates) > 1:
        raise AmbiguousForkRemoteError(candidates)
    return candidates[0]


@dataclass
class Configuration:
    """Everything this checkout runs on: the layered settings and the remotes they resolve to."""

    in_review_label: str
    """Fork-PR label marking a branch as promoted to the upstream and under review."""

    rebase_label: str
    """Fork-PR label opting a branch into the rebase strategy instead of the default merge."""

    needs_resolution_label: str
    """Fork-PR label marking a branch withheld from promotion pending conflict resolution."""

    fork_repository: Repository
    """The fork that holds the full stack, as GitHub names it."""

    fork_remote: str
    """Git remote for the fork that holds the full stack."""

    upstream_repository: Repository
    """The repository every fork is forked from, and the only one constant across contributors."""

    upstream_remote: str
    """Git remote for the upstream review repository."""

    upstream_base: str
    """The upstream base branch every stack ultimately targets."""

    upstream_setup_command: str | None
    """The command adding the upstream remote, or ``None`` once this checkout has one."""


def load_configuration(
    path: Path = CONFIGURATION_PATH,
    fork_repository: Repository | None = None,
    upstream_repository: Repository | None = None,
) -> Configuration:
    """Parse the layered configuration into a :class:`Configuration`.

    Values from the committed *path* are the defaults; any key present in
    ``.claude/personal/stack.toml`` on the personal-notes branch overrides them, so a user's own
    remotes/labels never have to be hand-edited into the checked-in file. A repository passed
    here outranks both, so a caller that has been told which repository is which never has one
    inferred for it.

    :param path: The committed defaults file.
    :param fork_repository: The fork, when the caller already knows it.
    :param upstream_repository: The upstream, when the caller already knows it.
    :return: The layered configuration.
    """
    values = _configuration_values(path)
    upstream_repository = upstream_repository or Repository.parse(
        values["upstream_repository"]
    )
    resolution = resolved_remotes(path, fork_repository, upstream_repository)
    return Configuration(
        in_review_label=values.get("in_review_label", "in-review"),
        rebase_label=values.get("rebase_label", "rebase"),
        needs_resolution_label=values.get("needs_resolution_label", "needs-resolution"),
        fork_repository=resolution.fork.repository,
        fork_remote=resolution.fork.name,
        upstream_repository=upstream_repository,
        upstream_remote=resolution.upstream_name,
        upstream_base=values.get("upstream_base", "main"),
        upstream_setup_command=resolution.upstream_setup_command,
    )


def _configuration_values(path: Path) -> dict[str, str]:
    """Read the committed defaults with any personal-notes overrides layered on top.

    :param path: The committed defaults file.
    :return: The layered values.
    """
    values = tomllib.loads(path.read_text())
    values.update(_personal_configuration_overrides())
    return values


def resolved_remotes(
    path: Path = CONFIGURATION_PATH,
    fork_repository: Repository | None = None,
    upstream_repository: Repository | None = None,
) -> RemoteResolution:
    """Resolve this checkout's fork and upstream remotes.

    :param path: The committed defaults file.
    :param fork_repository: The fork, when the caller already knows it.
    :param upstream_repository: The upstream, when the caller already knows it.
    :return: The resolved remotes.
    :raises ForkRemoteNotFoundError: If no remote points at a fork.
    :raises AmbiguousForkRemoteError: If several do and nothing names which one it is.
    """
    values = _configuration_values(path)
    configured_fork = values.get("fork_repository")
    return resolve_remotes(
        _remote_urls(),
        upstream_repository or Repository.parse(values["upstream_repository"]),
        values.get("upstream_remote", "cram2"),
        fork_repository
        or (Repository.parse(configured_fork) if configured_fork else None),
    )


def _remote_urls() -> dict[str, str]:
    """:return: Every remote in this checkout, mapped to its fetch URL."""
    listed = _git("remote").splitlines()
    return {name: _git("remote", "get-url", name) for name in listed if name}


def _resolve_personal_notes_remote() -> str:
    """:return: the personal-notes remote, by the same precedence as
    ``resolve-personal-notes-config.sh``: git config, then an environment variable, then a default.
    """
    return (
        _git("config", "--get", "claude.personalNotesRemote")
        or os.environ.get("CLAUDE_PERSONAL_NOTES_REMOTE")
        or "origin"
    )


def _resolve_personal_notes_branch() -> str:
    """:return: the personal-notes branch name, by the same precedence as
    :func:`_resolve_personal_notes_remote`."""
    return (
        _git("config", "--get", "claude.personalNotesBranch")
        or os.environ.get("CLAUDE_PERSONAL_NOTES_BRANCH")
        or "claude/personal-notes"
    )


def _personal_configuration_overrides() -> dict[str, object]:
    """Fetch the personal-notes branch and parse its configuration override file, if any.

    :return: The parsed contents of ``.claude/personal/stack.toml`` on the personal-notes branch, or
        an empty mapping if the branch or the file doesn't exist (e.g. before it has ever been
        written).
    """
    remote = _resolve_personal_notes_remote()
    branch = _resolve_personal_notes_branch()
    if not _git_succeeds("fetch", remote, branch, "--quiet"):
        return {}
    if not _git_succeeds(
        "cat-file", "-e", f"FETCH_HEAD:{PERSONAL_STACK_CONFIGURATION_PATH}"
    ):
        return {}
    return tomllib.loads(
        _git("show", f"FETCH_HEAD:{PERSONAL_STACK_CONFIGURATION_PATH}")
    )


# %% domain model


class BranchStatus(StrEnum):
    """A stack node's lifecycle position."""

    DRAFT = "draft"
    READY = "ready"
    IN_REVIEW = "in-review"
    MERGED = "merged"


class IntegrationStrategy(StrEnum):
    """How a branch integrates its parent's moved tip during a restack."""

    MERGE = "merge"
    REBASE = "rebase"


@dataclass
class PullRequest:
    """One fork pull request as exported into ``board.json``."""

    number: int
    """The pull request number on the fork."""

    head: str
    """The PR's head branch - the branch this stack node names."""

    base: str
    """The PR's base branch - its parent in the stack (``base = parent``)."""

    draft: bool
    """Whether the PR is a draft (not yet approved for review)."""

    labels: list[str] = field(default_factory=list)
    """Labels currently on the PR."""

    ci: str | None = None
    """Latest CI conclusion on the PR head: ``success`` / ``failure`` / ``pending`` / None."""

    session: str | None = None
    """URL of the Claude session working this PR, parsed from the PR body (None if none)."""


@dataclass
class Branch:
    """A stack node derived from a fork PR plus git state."""

    name: str
    """The branch name (the PR head)."""

    parent: str
    """The parent branch (the PR base)."""

    pull_request_number: int
    """The fork PR number."""

    status: BranchStatus
    """Lifecycle status."""

    strategy: IntegrationStrategy
    """Integration strategy onto the parent."""

    labels: list[str]
    """Labels carried by the PR."""

    ci: str | None = None
    """Latest CI conclusion on the PR head."""

    session: str | None = None
    """URL of the Claude session working this PR, if any."""


@dataclass
class Stack:
    """The whole stack: configuration plus the branches derived from GitHub and git."""

    configuration: Configuration
    """The static configuration."""

    branches: list[Branch]
    """The derived stack nodes."""

    is_merged: Callable[[str], bool]
    """Maps any branch name - tracked by this stack or not - to whether it has landed
    upstream."""

    def needs_resolution(self, branch: Branch) -> bool:
        """:param branch: The branch to check.
        :return: Whether the branch is withheld from promotion pending conflict resolution.
        """
        return self.configuration.needs_resolution_label in branch.labels

    def has_landed_upstream(self, branch_name: str) -> bool:
        """Whether a branch's commits are already in the upstream base.

        Answered from git ancestry, so it holds for any branch name - including one no open
        pull request describes, which the board therefore never mentions.

        :param branch_name: The branch to check.
        :return: Whether its commits are in the upstream base.
        """
        return branch_name == self.configuration.upstream_base or self.is_merged(
            branch_name
        )


class BoardUnavailable(RuntimeError):
    """Raised when ``board.json`` is missing."""


def load_board(path: Path = BOARD_PATH) -> list[PullRequest]:
    """Parse ``board.json`` into the list of fork pull requests.

    :param path: The board export file.
    :return: The exported pull requests.
    :raises BoardUnavailable: If *path* does not exist.
    """
    if not path.exists():
        raise BoardUnavailable(f"{path.name} not found")
    data = json.loads(path.read_text())
    return [
        PullRequest(
            number=pr["number"],
            head=pr["head"],
            base=pr["base"],
            draft=bool(pr["draft"]),
            labels=list(pr.get("labels", [])),
            ci=pr.get("ci"),
            session=pr.get("session"),
        )
        for pr in data["pull_requests"]
    ]


def derive_status(draft: bool, merged: bool, in_review: bool) -> BranchStatus:
    """Map a PR's raw facts to a lifecycle status.

    Precedence: a merged branch is ``merged``; an ``in-review``-labelled branch is ``in-review``; an
    un-drafted branch is ``ready`` (self-approved for promotion); otherwise ``draft``.

    :param draft: Whether the PR is still a draft.
    :param merged: Whether the branch has landed upstream (git ancestry).
    :param in_review: Whether the PR carries the in-review label.
    :return: The derived status.
    """
    if merged:
        return BranchStatus.MERGED
    if in_review:
        return BranchStatus.IN_REVIEW
    return BranchStatus.DRAFT if draft else BranchStatus.READY


def build_stack(
    configuration: Configuration,
    prs: list[PullRequest],
    is_merged: Callable[[str], bool],
) -> Stack:
    """Assemble the :class:`Stack` from the PR export and a merged-branch predicate.

    :param configuration: The static configuration.
    :param prs: The exported pull requests.
    :param is_merged: Maps a branch name to whether it has landed upstream; injected so the pure
        assembly logic can be tested without git.
    :return: The assembled stack.
    """
    branches = [
        Branch(
            name=pr.head,
            parent=pr.base,
            pull_request_number=pr.number,
            status=derive_status(
                pr.draft, is_merged(pr.head), configuration.in_review_label in pr.labels
            ),
            strategy=(
                IntegrationStrategy.REBASE
                if configuration.rebase_label in pr.labels
                else IntegrationStrategy.MERGE
            ),
            labels=pr.labels,
            ci=pr.ci,
            session=pr.session,
        )
        for pr in prs
    ]
    return Stack(configuration=configuration, branches=branches, is_merged=is_merged)


# %% label writes


@dataclass
class ContradictoryLabelWriteError(ValueError):
    """Raised when one write is asked both to carry a label and to drop it."""

    labels: tuple[str, ...]
    """The labels asked for in both directions."""

    def __str__(self) -> str:
        """:return: Which labels contradict, and why no outcome is safe."""
        return (
            f"cannot both add and remove {', '.join(self.labels)}: "
            f"either outcome would be a guess at what was meant"
        )


@dataclass(frozen=True)
class LabelWrite:
    """The complete label set a pull request has to be written back with.

    GitHub's label write replaces the whole set rather than adding to it, so the labels a
    change leaves untouched still have to be sent. Computing them from the intended change
    alone is what silently strips the rest.
    """

    labels: tuple[str, ...]
    """Every label the pull request will carry once written, in the order it will carry them."""

    @classmethod
    def replacing(
        cls,
        current: Iterable[str],
        added: Iterable[str] = (),
        removed: Iterable[str] = (),
    ) -> LabelWrite:
        """Work out the full set that adding and removing labels leaves behind.

        :param current: The labels the pull request carries now.
        :param added: Labels it must carry afterwards.
        :param removed: Labels it must not carry afterwards.
        :return: The complete set to write.
        :raises ContradictoryLabelWriteError: If a label is both added and removed.
        """
        added, removed = tuple(added), tuple(removed)
        contradictory = tuple(label for label in added if label in removed)
        if contradictory:
            raise ContradictoryLabelWriteError(contradictory)
        kept = [label for label in current if label not in removed]
        return cls(tuple(kept + [label for label in added if label not in kept]))


# %% promotion links


@dataclass
class PromotionLinkTooLongError(ValueError):
    """Raised when a link cannot fit even with its whole description dropped."""

    branch: str
    """The branch whose link was being built."""

    length_without_a_description: int
    """How long the URL already is before any description is added."""

    def __str__(self) -> str:
        """:return: Which link overflowed, and by how much before the body counts."""
        return (
            f"the compare link for {self.branch} is {self.length_without_a_description} "
            f"characters before its description, over the "
            f"{PromotionLink.URL_CHARACTER_LIMIT}-character limit: shorten the title"
        )


@dataclass(frozen=True)
class PromotionLink:
    """A one-click compare-and-create link opening a fork branch's pull request upstream.

    The prefill travels in the query string, so an unencoded character truncates it and an
    oversized one is discarded whole by the server - both silently.
    """

    URL_CHARACTER_LIMIT: ClassVar[int] = 8192
    """Longest URL to build, the conventional 8 KiB request-line limit servers accept."""

    TRUNCATION_MARKER: ClassVar[str] = "..."
    """Appended to a body that had to be shortened, so the cut is visible in the prefill."""

    url: str
    """The compare-and-create URL, ready to open."""

    body_was_truncated: bool
    """Whether the body had to be shortened to fit :attr:`URL_CHARACTER_LIMIT`."""

    @classmethod
    def build(
        cls, configuration: Configuration, branch: str, title: str, body: str
    ) -> PromotionLink:
        """Build the link promoting a fork branch to the upstream base.

        :param configuration: The resolved configuration naming both repositories.
        :param branch: The fork branch to promote.
        :param title: Title to prefill.
        :param body: Description to prefill, shortened if the URL would not otherwise fit.
        :return: The link, flagged if the body was shortened.
        """

        def url_for(text: str) -> str:
            return (
                f"https://github.com/{configuration.upstream_repository}/compare/"
                f"{configuration.upstream_base}..."
                f"{configuration.fork_repository.owner}:{branch}"
                f"?expand=1&title={quote(title)}&body={quote(text)}"
            )

        if len(url_for(body)) <= cls.URL_CHARACTER_LIMIT:
            return cls(url_for(body), body_was_truncated=False)
        if len(url_for(cls.TRUNCATION_MARKER)) > cls.URL_CHARACTER_LIMIT:
            raise PromotionLinkTooLongError(branch, len(url_for("")))
        shortest, longest = 0, len(body)
        while shortest < longest:
            midpoint = (shortest + longest + 1) // 2
            candidate = url_for(body[:midpoint] + cls.TRUNCATION_MARKER)
            shortest, longest = (
                (midpoint, longest)
                if len(candidate) <= cls.URL_CHARACTER_LIMIT
                else (shortest, midpoint - 1)
            )
        return cls(
            url_for(body[:shortest] + cls.TRUNCATION_MARKER), body_was_truncated=True
        )


# %% git plumbing


def _git(*args: str) -> str:
    """Run a git command and return its stripped stdout (empty string on failure).

    :param args: The git subcommand and its arguments.
    :return: The command's stripped stdout.
    """
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=Path.cwd()
    )
    return result.stdout.strip()


def _git_succeeds(*args: str) -> bool:
    """Run a git command, discarding its output.

    :param args: The git subcommand and its arguments.
    :return: Whether the command exited successfully.
    """
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=Path.cwd()
    )
    return result.returncode == 0


def _merged_predicate(configuration: Configuration):
    """:param configuration: The static configuration.
    :return: A predicate testing whether a fork branch is an ancestor of the upstream base.
    """
    upstream = f"{configuration.upstream_remote}/{configuration.upstream_base}"

    def is_merged(name: str) -> bool:
        ref = f"{configuration.fork_remote}/{name}"
        return _git_succeeds("merge-base", "--is-ancestor", ref, upstream)

    return is_merged


def load_stack() -> Stack:
    """:return: the full live stack: configuration + board export + git merged-detection."""
    configuration = load_configuration()
    prs = load_board()
    fetch(configuration, [pr.head for pr in prs])
    return build_stack(configuration, prs, _merged_predicate(configuration))


def resolve_ref(configuration: Configuration, name: str) -> str:
    """:param configuration: The static configuration.
    :param name: A branch or parent name.
    :return: Its ref on the fork remote."""
    return f"{configuration.fork_remote}/{name}"


def fetch(configuration: Configuration, branches: list[str]) -> None:
    """Refresh the refs the stack references so drift and merged-detection are current.

    :param configuration: The static configuration.
    :param branches: The fork branch names to fetch.
    """
    _git("fetch", configuration.upstream_remote, configuration.upstream_base, "-q")
    _git("fetch", configuration.fork_remote, "-q", *branches)


def _count(rev_range: str) -> int | None:
    """:param rev_range: A git rev-range expression.
    :return: The number of commits in it, or ``None`` if a ref is missing."""
    out = _git("rev-list", "--count", rev_range)
    return int(out) if out.isdigit() else None


# %% stack assembly


def order(stack: Stack) -> list[Branch]:
    """:param stack: The stack to order.
    :return: Its branches, topologically ordered so a parent always precedes its children.
    """
    by_name = {b.name: b for b in stack.branches}
    ordered: list[Branch] = []
    seen: set[str] = set()

    def visit(branch: Branch) -> None:
        if branch.name in seen:
            return
        seen.add(branch.name)
        parent = by_name.get(branch.parent)
        if parent is not None:
            visit(parent)
        ordered.append(branch)

    for branch in stack.branches:
        visit(branch)
    return ordered


def parent_landed(stack: Stack, branch: Branch, by_name: dict[str, Branch]) -> bool:
    """Whether a branch's parent has reached the upstream (merged or in-review), so it can promote.

    A parent no open pull request describes cannot carry an in-review label, so git ancestry is the
    only evidence available for it - and absence from the board is not itself evidence of a root
    branch.

    :param stack: The stack the branch belongs to.
    :param branch: The branch to check.
    :param by_name: Every branch in the stack, keyed by name.
    :return: Whether the branch's parent has landed.
    """
    parent = by_name.get(branch.parent)
    if parent is None:
        return stack.has_landed_upstream(branch.parent)
    return parent.status in {
        BranchStatus.IN_REVIEW,
        BranchStatus.MERGED,
    }


# %% promotion policy


def promotion_order(stack: Stack) -> list[Branch]:
    """The branches to submit to the upstream next, in dependency order.

    A branch is a candidate when it is approved (``ready``), its parent has landed, and it has not
    been withheld pending conflict resolution. Every such branch promotes together - there is no
    admission cap or per-stack slot limit (see the module's history for why: a ``wip_cap`` large
    enough to never bind made that machinery a no-op, so it was removed rather than fixed).

    :param stack: The stack to evaluate.
    :return: The promotable branches, in dependency order.
    """
    by_name = {b.name: b for b in stack.branches}
    return [
        branch
        for branch in order(stack)
        if branch.status == BranchStatus.READY
        and parent_landed(stack, branch, by_name)
        and not stack.needs_resolution(branch)
    ]


def next_to_promote(stack: Stack) -> Branch | None:
    """:param stack: The stack to evaluate.
    :return: The first branch to submit to the upstream next, or ``None`` if nothing is ready.
    """
    ordered = promotion_order(stack)
    return ordered[0] if ordered else None


def restack_plan(stack: Stack) -> list[dict[str, str]]:
    """The bottom-up restack plan the ``restack`` workflow consumes as its ``args``.

    One entry per branch not yet ``merged``, in parent-before-child order. In-review branches are
    included so they pick up a moved parent; their ``merge`` strategy keeps that update conflict-free
    and force-push-free, so an open review is never disrupted.

    When a branch's parent has **merged** into the upstream, its commits are already in the base, so
    the child is reparented onto the upstream base: the restack rebases it there and it stops
    depending on a landed (and about-to-be-closed) branch. The routine mirrors this by retargeting
    the child PR's base to the upstream base on GitHub. This holds however the parent landed -
    including when its own pull request was closed rather than merged, leaving the board with no
    entry for it at all.

    :param stack: The stack to plan.
    :return: The restack plan, one entry per not-yet-merged branch.
    """
    plan: list[dict[str, str]] = []
    for branch in order(stack):
        if branch.status == BranchStatus.MERGED:
            continue
        effective_parent = (
            stack.configuration.upstream_base
            if stack.has_landed_upstream(branch.parent)
            else branch.parent
        )
        plan.append(
            {
                "branch": branch.name,
                "parent": effective_parent,
                "strategy": branch.strategy,
            }
        )
    return plan


# %% landed parents


@dataclass(frozen=True)
class Reparent:
    """A pull request whose base has landed and must be retargeted at the upstream base.

    A child left on a landed base cannot reach the upstream, and is closed outright the
    moment that base branch is deleted.
    """

    branch: str
    """The child branch that has to move."""

    pull_request_number: int
    """The child's fork pull request."""

    current_base: str
    """The landed branch it is still targeting."""

    target_base: str
    """The base it must be retargeted at."""


def reparents(stack: Stack) -> list[Reparent]:
    """Every open pull request whose base has already landed upstream.

    Decided by git ancestry rather than by the base's own pull-request state, so it also
    covers a parent whose pull request was closed rather than merged - which leaves the
    board with no entry for it at all.

    :param stack: The stack to sweep.
    :return: The children to retarget, parent before child.
    """
    base = stack.configuration.upstream_base
    return [
        Reparent(
            branch=branch.name,
            pull_request_number=branch.pull_request_number,
            current_base=branch.parent,
            target_base=base,
        )
        for branch in order(stack)
        if branch.status != BranchStatus.MERGED
        and branch.parent != base
        and stack.has_landed_upstream(branch.parent)
    ]


def landed_branches(stack: Stack) -> list[Branch]:
    """Every open fork pull request whose own branch is already in the upstream base.

    These are the ones to label as landed and close; their children are named separately
    by :func:`reparents`, which must be acted on first so none is ever orphaned.

    :param stack: The stack to sweep.
    :return: The landed branches, parent before child.
    """
    return [branch for branch in order(stack) if branch.status == BranchStatus.MERGED]


# %% pre-flight


class CommitMoveAction(StrEnum):
    """What a proposed move would do to the destination branch."""

    PUSH = "push"
    MERGE = "merge"
    RESTACK = "restack"


@dataclass(frozen=True)
class ProposedCommitMove:
    """Commits a caller proposes to move onto a branch, before anything is run."""

    action: CommitMoveAction
    """What the move would do."""

    source: str
    """The branch whose commits would move."""

    destination: str
    """The branch they would land on."""

    destination_remote: str
    """The remote holding the destination branch."""


class RefusalReason(StrEnum):
    """Why a proposed move must not be made.

    Each names a move that has gone wrong in practice, so a caller can act on which one
    it hit without reading the sentence explaining it.
    """

    NOT_CHECKED_OUT = "not-checked-out"
    """The source is not the branch whose content a push would actually move."""

    MISMATCHED_BRANCH_NAMES = "mismatched-branch-names"
    """The source and the destination of the push are different branches."""

    NOT_THE_FORK = "not-the-fork"
    """The destination remote is not the fork the stack lives on."""

    FALSE_MERGE = "false-merge"
    """A child would become an ancestor of its own parent, which GitHub reads as merged."""


@dataclass(frozen=True)
class PreFlightRefusal:
    """One reason a proposed move must not be made."""

    reason: RefusalReason
    """Which refusal this is, for a caller deciding what to do about it."""

    explanation: str
    """What is wrong, in terms of the branches the caller named."""


@dataclass(frozen=True)
class PreFlight:
    """Checks a proposed move against the checkout it would run in.

    Every refusal describes a move that has gone wrong in practice - see
    :class:`RefusalReason`.
    """

    stack: Stack
    """The stack the branches belong to."""

    checked_out_branch: str
    """The branch actually checked out, whose content a push would move."""

    is_ancestor: Callable[[str, str], bool]
    """Maps a candidate ancestor branch and a descendant branch to whether the first is
    contained in the second; injected so the checks are testable without git."""

    def refusals(self, move: ProposedCommitMove) -> list[PreFlightRefusal]:
        """Every reason the move must not be made.

        All of them are reported together: fixing one and re-running to discover the next
        is how a move gets half made.

        :param move: The proposed move.
        :return: The refusals, empty when the move is safe.
        """
        found = []
        if move.source != self.checked_out_branch:
            found.append(
                PreFlightRefusal(
                    RefusalReason.NOT_CHECKED_OUT,
                    f"'{move.source}' is not checked out; '{self.checked_out_branch}' is, "
                    f"and its content is what would move",
                )
            )
        if move.source != move.destination:
            found.append(
                PreFlightRefusal(
                    RefusalReason.MISMATCHED_BRANCH_NAMES,
                    f"this would move '{move.source}' onto '{move.destination}'; the "
                    f"source and the destination must be the same branch",
                )
            )
        if move.destination_remote != self.stack.configuration.fork_remote:
            found.append(
                PreFlightRefusal(
                    RefusalReason.NOT_THE_FORK,
                    f"'{move.destination_remote}' is not the fork remote "
                    f"'{self.stack.configuration.fork_remote}'",
                )
            )
        found.extend(self._false_merges(move))
        return found

    def _false_merges(self, move: ProposedCommitMove) -> list[PreFlightRefusal]:
        """The children of the destination this move would make GitHub call merged.

        :param move: The proposed move.
        :return: One refusal per child already contained in the move's source.
        """
        return [
            PreFlightRefusal(
                RefusalReason.FALSE_MERGE,
                f"'{child.name}' is already contained in '{move.source}', so this would "
                f"make it an ancestor of its own parent '{move.destination}' and GitHub "
                f"would close pull request #{child.pull_request_number} as merged",
            )
            for child in self.stack.branches
            if child.parent == move.destination
            and self.is_ancestor(child.name, move.source)
        ]


# %% commands


def print_status(stack: Stack) -> None:
    """Print the whole stack: parent, state, and drift versus the upstream.

    :param stack: The stack to report.
    """
    configuration = stack.configuration
    upstream = f"{configuration.upstream_remote}/{configuration.upstream_base}"
    print(f"Stack ({len(stack.branches)} branches) vs {upstream}\n")
    print(f"{'branch':<38} {'state':<10} {'PR':>4}  ahead/behind parent   behind base")
    print("-" * 92)
    for branch in order(stack):
        ref = resolve_ref(configuration, branch.name)
        parent_ref = resolve_ref(configuration, branch.parent)
        ahead = _count(f"{parent_ref}..{ref}")
        behind_parent = _count(f"{ref}..{parent_ref}")
        behind_base = _count(f"{ref}..{upstream}")
        drift = f"+{ahead}/-{behind_parent} ({branch.strategy} onto {branch.parent})"
        print(
            f"{branch.name:<38} {branch.status:<10} #{branch.pull_request_number:<3}  {drift:<28} {behind_base}"
        )


def print_check(stack: Stack) -> None:
    """Print whether each branch would merge cleanly onto its parent right now.

    :param stack: The stack to probe.
    """
    configuration = stack.configuration
    print(
        "Integration probe - would each branch merge cleanly onto its parent right now?\n"
    )
    for branch in order(stack):
        ref = resolve_ref(configuration, branch.name)
        parent_ref = resolve_ref(configuration, branch.parent)
        result = subprocess.run(
            ["git", "merge-tree", "--write-tree", parent_ref, ref],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            verdict = "CLEAN"
        elif result.returncode == 1:
            verdict = f"CONFLICTS onto {branch.parent}"
        else:
            verdict = f"UNKNOWN (ref missing: {parent_ref} / {ref})"
        print(f"  {branch.name:<40} {verdict}")


def print_next(stack: Stack) -> None:
    """Print which branch(es) are ready to submit to the upstream next.

    :param stack: The stack to report.
    """
    configuration = stack.configuration
    by_name = {b.name: b for b in stack.branches}
    promotable = promotion_order(stack)
    withheld = [
        b
        for b in stack.branches
        if b.status == BranchStatus.READY
        and parent_landed(stack, b, by_name)
        and stack.needs_resolution(b)
    ]

    def report_withheld() -> None:
        if withheld:
            print(
                f"  Withheld (delegated, needs-resolution): {', '.join(b.name for b in withheld)}"
            )

    if promotable:
        plural = "es" if len(promotable) != 1 else ""
        print(
            f"NEXT to submit to {configuration.upstream_remote} ({len(promotable)} branch{plural}):"
        )
        for branch in promotable:
            print(
                f"  {branch.name} (PR #{branch.pull_request_number}) - approved, parent '{branch.parent}' landed"
            )
        report_withheld()
        return

    ready_blocked = [
        b
        for b in stack.branches
        if b.status == BranchStatus.READY and not parent_landed(stack, b, by_name)
    ]
    draft_candidates = [b for b in order(stack) if b.status == BranchStatus.DRAFT]

    print("Nothing to promote - no branch is both approved and unblocked.")
    if ready_blocked:
        print(
            f"  Approved but waiting on a parent to land: {', '.join(b.name for b in ready_blocked)}"
        )
    report_withheld()
    if draft_candidates:
        print(
            "  The gate: self-review a fork PR, then un-draft it (or set its status ready). "
            f"Draft candidates: {draft_candidates[0].name}"
        )


def print_next_porcelain(stack: Stack) -> None:
    """Print machine-readable :func:`print_next`: one ``name<TAB>pr`` line per promotable branch.

    :param stack: The stack to report.
    """
    for branch in promotion_order(stack):
        print(f"{branch.name}\t{branch.pull_request_number}")


def print_restack_plan(stack: Stack) -> None:
    """Print the bottom-up restack plan as JSON, one object per not-yet-landed branch.

    :param stack: The stack to plan.
    """
    print(json.dumps(restack_plan(stack), indent=2))


def print_label_write(write: LabelWrite) -> None:
    """Print the complete label set, one label per line.

    One per line rather than a separator-joined list, because a label may contain any
    character a separator could be.

    :param write: The computed set.
    """
    for label in write.labels:
        print(label)


def print_promotion_link(link: PromotionLink) -> None:
    """Print the compare-and-create URL, reporting on stderr if the body was shortened.

    :param link: The built link.
    """
    print(link.url)
    if link.body_was_truncated:
        print(
            "the description was shortened to fit the URL length limit",
            file=sys.stderr,
        )


def print_reparents(stack: Stack) -> None:
    """Print one ``branch<TAB>pr<TAB>current base<TAB>target base`` line per orphaned child.

    :param stack: The stack to sweep.
    """
    for reparent in reparents(stack):
        print(
            f"{reparent.branch}\t{reparent.pull_request_number}\t"
            f"{reparent.current_base}\t{reparent.target_base}"
        )


def print_landed(stack: Stack) -> None:
    """Print one ``branch<TAB>pr`` line per open pull request whose branch has landed.

    :param stack: The stack to sweep.
    """
    for branch in landed_branches(stack):
        print(f"{branch.name}\t{branch.pull_request_number}")


def print_preflight(pre_flight: PreFlight, move: ProposedCommitMove) -> ExitCode:
    """Print whether a move may be made, and every reason it may not.

    :param pre_flight: The checks to run.
    :param move: The proposed move.
    :return: Success when the move is clear, refusal otherwise.
    """
    refusals = pre_flight.refusals(move)
    if not refusals:
        print(
            f"{move.action} {move.source} onto "
            f"{move.destination_remote}/{move.destination}: clear"
        )
        return ExitCode.SUCCESS
    for refusal in refusals:
        print(f"{refusal.reason}: {refusal.explanation}", file=sys.stderr)
    return ExitCode.PREFLIGHT_REFUSED


def print_configuration(configuration: Configuration) -> None:
    """Print the resolved configuration as one ``field<TAB>value`` line per setting.

    Keys are :class:`Configuration`'s own field names, so a caller reading one by name cannot
    be reading a name this module never prints. A setting with no value is omitted rather than
    printed empty, which is what keeps ``upstream_setup_command`` readable as "run this".

    :param configuration: The configuration to report.
    """
    for name, value in vars(configuration).items():
        if value is None:
            continue
        print(f"{name}\t{value}")


class Command(StrEnum):
    """Every command this tool answers, named once so no caller spells one out."""

    STATUS = "status"
    CHECK = "check"
    NEXT = "next"
    RESTACK_PLAN = "restack-plan"
    REPARENTS = "reparents"
    LANDED = "landed"
    PREFLIGHT = "preflight"
    CONFIGURATION = "configuration"
    LABELS = "labels"
    PROMOTION_LINK = "promotion-link"

    @property
    def needs_a_board(self) -> bool:
        """Whether answering this command means deriving the stack.

        The ones that do not are answerable from git and configuration alone, so they
        run before a board has ever been exported.

        :return: Whether ``board.json`` must exist.
        """
        return self not in {
            Command.CONFIGURATION,
            Command.LABELS,
            Command.PROMOTION_LINK,
        }


# %% entry point


class ExitCode(IntEnum):
    """What this tool's exit status tells a caller.

    A distinct status per failure lets a caller - a shell script, or a session acting on
    what it gets back - tell "you asked for something that does not exist" from "the
    checkout is not in a state I can read", without parsing stderr.
    """

    SUCCESS = 0
    """The command ran and printed its result."""

    USAGE = 2
    """No such command, or the wrong arguments; the conventional status for a usage
    error, as `argparse` also uses."""

    BOARD_UNAVAILABLE = 3
    """`board.json` is missing or unreadable, so the stack cannot be derived."""

    REMOTES_UNRESOLVED = 4
    """The fork could not be identified from this checkout's remotes."""

    PREFLIGHT_REFUSED = 5
    """The proposed move must not be made; the reasons are on stderr."""


def _argument_parser() -> argparse.ArgumentParser:
    """:return: The parser for every command and its own flags."""
    parser = argparse.ArgumentParser(
        prog="stack.py",
        description="Stacked-PR helper: read state, compute the writes.",
    )
    parser.set_defaults(porcelain=False)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser(Command.STATUS, help="the whole stack: parent, state, drift")
    commands.add_parser(
        Command.CHECK, help="would each branch merge cleanly onto its parent"
    )
    promote = commands.add_parser(
        Command.NEXT, help="which branches to submit upstream next"
    )
    promote.add_argument(
        "--porcelain",
        action="store_true",
        help="print only 'name<TAB>pr' per branch to promote",
    )
    commands.add_parser(Command.RESTACK_PLAN, help="the bottom-up restack plan as JSON")
    commands.add_parser(
        Command.REPARENTS, help="children whose base has landed, and the base they need"
    )
    commands.add_parser(
        Command.LANDED, help="open pull requests whose branch has landed"
    )

    configuration = commands.add_parser(
        Command.CONFIGURATION, help="every resolved setting, including the remotes"
    )
    configuration.add_argument(
        "--fork",
        type=Repository.parse,
        help="the fork as 'owner/name', when you already know it",
    )
    configuration.add_argument(
        "--upstream",
        type=Repository.parse,
        help="the upstream as 'owner/name', when you already know it",
    )

    labels = commands.add_parser(
        Command.LABELS, help="the complete label set a label write must send"
    )
    labels.add_argument(
        "--current",
        action="append",
        default=[],
        metavar="LABEL",
        help="a label the pull request carries now; repeat per label",
    )
    labels.add_argument(
        "--add",
        action="append",
        default=[],
        metavar="LABEL",
        help="a label to end up with",
    )
    labels.add_argument(
        "--remove",
        action="append",
        default=[],
        metavar="LABEL",
        help="a label to end up without",
    )

    link = commands.add_parser(
        Command.PROMOTION_LINK, help="the upstream compare-and-create URL for a branch"
    )
    link.add_argument("--branch", required=True, help="the fork branch to promote")
    link.add_argument("--title", required=True, help="title to prefill")
    link.add_argument("--body", default="", help="description to prefill")

    preflight = commands.add_parser(
        Command.PREFLIGHT, help="may these commits move onto that branch?"
    )
    preflight.add_argument(
        "--action",
        required=True,
        type=CommitMoveAction,
        choices=list(CommitMoveAction),
        help="what the move would do",
    )
    preflight.add_argument(
        "--source", required=True, help="the branch whose commits move"
    )
    preflight.add_argument(
        "--destination", required=True, help="the branch they would land on"
    )
    preflight.add_argument(
        "--destination-remote", required=True, help="the remote holding the destination"
    )
    return parser


def _run_without_a_board(command: Command, arguments: argparse.Namespace) -> ExitCode:
    """Run a command that needs no ``board.json``.

    :param command: The command to run.
    :param arguments: The parsed command line.
    :return: The process exit code.
    """
    if command is Command.LABELS:
        print_label_write(
            LabelWrite.replacing(arguments.current, arguments.add, arguments.remove)
        )
        return ExitCode.SUCCESS
    if command is Command.CONFIGURATION:
        print_configuration(
            load_configuration(
                fork_repository=arguments.fork,
                upstream_repository=arguments.upstream,
            )
        )
        return ExitCode.SUCCESS
    print_promotion_link(
        PromotionLink.build(
            load_configuration(), arguments.branch, arguments.title, arguments.body
        )
    )
    return ExitCode.SUCCESS


def _run_against_the_board(
    command: Command, arguments: argparse.Namespace, stack: Stack
) -> ExitCode:
    """Run a command that reads the derived stack.

    :param command: The command to run.
    :param arguments: The parsed command line.
    :param stack: The derived stack.
    :return: The process exit code.
    """
    if command is Command.PREFLIGHT:
        return print_preflight(
            PreFlight(
                stack=stack,
                checked_out_branch=_git("branch", "--show-current"),
                is_ancestor=_ancestry_predicate(stack.configuration),
            ),
            ProposedCommitMove(
                action=arguments.action,
                source=arguments.source,
                destination=arguments.destination,
                destination_remote=arguments.destination_remote,
            ),
        )
    reporters = {
        Command.STATUS: print_status,
        Command.CHECK: print_check,
        Command.NEXT: print_next_porcelain if arguments.porcelain else print_next,
        Command.RESTACK_PLAN: print_restack_plan,
        Command.REPARENTS: print_reparents,
        Command.LANDED: print_landed,
    }
    reporters[command](stack)
    return ExitCode.SUCCESS


def _ancestry_predicate(configuration: Configuration) -> Callable[[str, str], bool]:
    """:param configuration: The resolved configuration naming the fork remote.
    :return: A predicate testing whether a fork branch is contained in a local branch.
    """

    def is_ancestor(candidate: str, descendant: str) -> bool:
        return _git_succeeds(
            "merge-base",
            "--is-ancestor",
            resolve_ref(configuration, candidate),
            descendant,
        )

    return is_ancestor


def main() -> ExitCode:
    """Dispatch the command-line invocation, mapping every refusal to its own status.

    :return: The process exit code.
    """
    arguments = _argument_parser().parse_args()
    command = Command(arguments.command)
    try:
        if not command.needs_a_board:
            return _run_without_a_board(command, arguments)
        return _run_against_the_board(command, arguments, load_stack())
    except (ForkRemoteNotFoundError, AmbiguousForkRemoteError) as error:
        print(f"{error}", file=sys.stderr)
        return ExitCode.REMOTES_UNRESOLVED
    except BoardUnavailable as error:
        print(f"{error}", file=sys.stderr)
        return ExitCode.BOARD_UNAVAILABLE
    except (ContradictoryLabelWriteError, PromotionLinkTooLongError) as error:
        print(f"{error}", file=sys.stderr)
        return ExitCode.USAGE


if __name__ == "__main__":
    raise SystemExit(main())
