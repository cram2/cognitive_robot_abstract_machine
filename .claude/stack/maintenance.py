#!/usr/bin/env python3
"""Deterministic executor for the stacked-PR maintenance pass.

``stack.py`` derives what a pass should do and prints it; nothing in it moves a commit.
Every fetch, merge, rebase and push in the workflow was therefore performed by a session
following prose, and ``board.json`` was hand-assembled from whatever the caller happened
to fetch - the same class of hand-assembled input that let a dropped ``merged_at`` field
read as a legitimate value.

This module performs those steps instead, and reports what it did::

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
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
import dataclasses
from dataclasses import asdict, dataclass
from enum import Enum, IntEnum, StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

if TYPE_CHECKING:
    from types import TracebackType

from stack import (
    BOARD_PATH,
    AmbiguousForkRemoteError,
    BoardUnavailable,
    Branch,
    BranchStatus,
    CommitMoveAction,
    Configuration,
    ContradictoryLabelWriteError,
    ForkRemoteNotFoundError,
    IntegrationStrategy,
    LabelWrite,
    CommitMoveChecks,
    PromotionLink,
    PromotionLinkTooLongError,
    ProposedCommitMove,
    PullRequest,
    RefusalReason,
    Reparent,
    Repository,
    Stack,
    landed_branches,
    load_configuration,
    load_stack,
    promotion_order,
    reparents,
    resolve_ref,
    restack_plan,
)

GITHUB_API_ROOT = "https://api.github.com"
"""Base URL every REST call in this module is built on."""

CREDENTIAL_VARIABLES = ("GH_TOKEN", "GITHUB_TOKEN")
"""Environment variables read, in order, for the token the API calls authenticate with."""

SESSION_LINK_PATTERN = re.compile(r"https://claude\.ai/code/session_[A-Za-z0-9_-]+")
"""Matches the session link a pull request description carries, which is the only
channel for telling a branch's owner that their branch needs them."""

PullRequestRecord = Mapping[str, Any]
"""One pull request as the REST API answers it, before any field is read."""


# %% when something this pass depends on refuses


@dataclass
class ExternalCallFailed(RuntimeError):
    """Base for a call to git or GitHub that this pass depended on and did not get.

    Both carry the same three things under different names - what was called, the
    status it came back with, and what it said - so they say so once here and differ
    only in how they name the call. Mirrors ``krrood``'s dataclass-exception idiom
    (typed context fields, an abstract message composed by the base) without importing
    it, since this module is deliberately dependency-free.
    """

    status: int
    """The status the call came back with."""

    detail: str
    """What the far side said about it."""

    @property
    def call(self) -> str:
        """:return: The call that failed, named the way its own caller named it."""
        raise NotImplementedError

    def __str__(self) -> str:
        """:return: The call, its status, and the reason given."""
        return f"{self.call} failed with {self.status}: {self.detail}"


@dataclass
class GitCommandFailed(ExternalCallFailed):
    """Raised when a git command this module depends on the result of fails."""

    arguments: tuple[str, ...] = ()
    """The git subcommand and its arguments, as invoked."""

    @property
    def call(self) -> str:
        """:return: The git command line, as invoked."""
        return f"git {' '.join(self.arguments)}"


@dataclass(frozen=True)
class GitCommandResult:
    """One finished git command, whether or not it succeeded."""

    arguments: tuple[str, ...]
    """The git subcommand and its arguments, as invoked."""

    exit_status: int
    """The status git exited with."""

    output: str
    """Git's stripped stdout."""

    error_output: str
    """Git's stripped stderr."""

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


@dataclass(frozen=True)
class GitCommandRunner:
    """Runs git in one checkout, reporting failures rather than swallowing them.

    ``stack.py`` reads git through a helper that returns an empty string when a command
    fails. That is right for derivation, where a missing ref simply means "no answer",
    and wrong here: a push that silently did nothing would be indistinguishable from one
    that worked.
    """

    working_directory: Path
    """The checkout every command runs in."""

    def attempt(self, *arguments: str) -> GitCommandResult:
        """Run a command whose failure is an expected outcome.

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
        """Run a command this module depends on the result of.

        :param arguments: The git subcommand and its arguments.
        :return: Git's stripped stdout.
        :raises GitCommandFailed: If git exits non-zero.
        """
        return self.attempt(*arguments).raise_if_failed().output

    def fetch(self, remote: str, *references: str) -> None:
        """Refresh what this checkout knows about a remote.

        :param remote: The remote to fetch from.
        :param references: The branches to fetch, all of them when none is named.
        """
        self.run("fetch", "--quiet", remote, *references)

    def commit_at(self, reference: str) -> str:
        """:param reference: Any reference git can resolve.
        :return: The commit it names."""
        return self.run("rev-parse", reference)

    def checkout(self, branch: str, start_point: str) -> None:
        """Put a branch at a starting point and check it out.

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
        """Undo whichever integration just failed.

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
        """Publish a refspec, forcing only where the push itself says it is authorised.

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


@dataclass(frozen=True)
class BranchAncestry:
    """Answers containment questions about the fork's branches.

    :class:`CommitMoveChecks` asks its false-merge question through this, so the question is
    asked of git rather than of anything this module remembers.
    """

    configuration: Configuration
    """The resolved configuration naming the fork remote."""

    git: GitCommandRunner
    """The runner to ask git through."""

    def is_ancestor(self, candidate: str, descendant: str) -> bool:
        """:param candidate: A fork branch that may be contained.
        :param descendant: A local branch that may contain it.
        :return: Whether the fork's copy of *candidate* is contained in *descendant*."""
        return self.git.contains(resolve_ref(self.configuration, candidate), descendant)


# %% the board export


class PullRequestFieldShape(StrEnum):
    """How one pull-request field's value has to be read.

    The API answers some fields with a nested object where a plain value would do, so
    reading is per-field rather than uniform.
    """

    VALUE = "value"
    """Taken as it comes."""

    BRANCH_REFERENCE = "branch-reference"
    """A branch, given either plainly or as an object carrying a ``ref``."""

    LABEL_NAMES = "label-names"
    """A list of labels, each given either plainly or as an object carrying a
    ``name``."""


@dataclass(frozen=True)
class PullRequestFieldSpecification:
    """What one pull-request field is called, how to read it, and whether it may be
    absent."""

    key: str
    """The key the API answers under."""

    shape: PullRequestFieldShape = PullRequestFieldShape.VALUE
    """How its value has to be read."""

    required: bool = False
    """Whether a record omitting it is rejected rather than read."""


class PullRequestField(PullRequestFieldSpecification, Enum):
    """Every pull-request field this executor reads, and how to read it.

    Each member *is* a specification, so nothing outside this enum knows that ``head``
    arrives nested while ``draft`` does not, or which fields a board cannot be derived
    without.

    A member is written as the specification it carries, and :meth:`__init__` unpacks it
    onto the member itself - so ``PullRequestField.HEAD.key`` reads directly and the
    member is a :class:`PullRequestFieldSpecification` in its own right.
    """

    def __init__(self, specification: PullRequestFieldSpecification) -> None:
        """Carry the specification's values on the member itself.

        Without this the mixin would receive the whole specification as its first
        argument - silently, landing the instance in :attr:`key` - since an enum passes
        a member's value straight to the type it mixes in.

        :param specification: What this field is called and how to read it.
        """
        for field in dataclasses.fields(PullRequestFieldSpecification):
            object.__setattr__(self, field.name, getattr(specification, field.name))

    NUMBER = PullRequestFieldSpecification(key="number", required=True)
    """The pull request's number."""

    HEAD = PullRequestFieldSpecification(
        key="head", shape=PullRequestFieldShape.BRANCH_REFERENCE, required=True
    )
    """The branch the pull request would merge - the stack node it names."""

    BASE = PullRequestFieldSpecification(
        key="base", shape=PullRequestFieldShape.BRANCH_REFERENCE, required=True
    )
    """The branch it would merge into - its parent in the stack."""

    DRAFT = PullRequestFieldSpecification(key="draft", required=True)
    """Whether its author has yet reviewed it themselves."""

    LABELS = PullRequestFieldSpecification(
        key="labels", shape=PullRequestFieldShape.LABEL_NAMES, required=True
    )
    """The labels it carries, which the workflow reads as state."""

    BODY = PullRequestFieldSpecification(key="body")
    """Its description, read for the session link and the promotion prefill."""

    TITLE = PullRequestFieldSpecification(key="title")
    """Its title, which prefills the upstream pull request."""

    MERGEABLE_STATE = PullRequestFieldSpecification(key="mergeable_state")
    """GitHub's own verdict on whether it currently conflicts with its base."""

    def read(self, record: PullRequestRecord, number: int | None = None) -> Any:
        """Read this field out of a fetched pull request.

        :param record: The fetched pull request.
        :param number: The pull request being read, named in any rejection.
        :return: The field's value, read according to its shape.
        :raises MissingPullRequestFieldError: If a required field is absent, or its
            value carries no name where its shape says one belongs.
        """
        value = record.get(self.key)
        if value is None:
            if self.required:
                raise MissingPullRequestFieldError(self, number)
            return None
        match self.shape:
            case PullRequestFieldShape.BRANCH_REFERENCE:
                return self._branch_reference(value, number)
            case PullRequestFieldShape.LABEL_NAMES:
                return [
                    label if isinstance(label, str) else str(label["name"])
                    for label in value
                ]
            case _:
                return value

    def _branch_reference(self, value: Any, number: int | None) -> str:
        """:param value: The field's value, plain or nested.
        :param number: The pull request being read, named in any rejection.
        :return: The branch it names.
        :raises MissingPullRequestFieldError: If it names none."""
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping) and value.get("ref"):
            return str(value["ref"])
        raise MissingPullRequestFieldError(self, number)


@dataclass
class MissingPullRequestFieldError(ValueError):
    """Raised when a fetched pull request omits a field the board is derived from.

    A fetch that drops a field is not partially correct: absent and legitimately empty
    are different facts, and defaulting one to the other is what makes bad board data
    indistinguishable from good.
    """

    field_name: PullRequestField
    """The field that was absent."""

    pull_request_number: int | None
    """The pull request it was absent from, or ``None`` when the number itself is."""

    def __str__(self) -> str:
        """:return: Which field is missing, and from where."""
        subject = (
            f"pull request {self.pull_request_number}"
            if self.pull_request_number is not None
            else "a fetched pull request"
        )
        return (
            f"{subject} has no '{self.field_name}'; the board cannot be derived from a "
            f"fetch that omits it"
        )


def get_session_link_in(body: str | None) -> str | None:
    """Read the session link out of a pull request description.

    :param body: The description to search, which may be absent.
    :return: The first session link, or ``None`` if the description names none.
    """
    if not body:
        return None
    found = SESSION_LINK_PATTERN.search(body)
    return found.group(0) if found else None


@dataclass(frozen=True)
class BoardExport:
    """The fork's open pull requests, in the shape the derived stack is read from."""

    pull_requests: tuple[PullRequest, ...]
    """The exported pull requests."""

    @classmethod
    def from_api_records(cls, records: Iterable[PullRequestRecord]) -> BoardExport:
        """Build the export from what the REST API returned.

        :param records: The fetched pull requests.
        :return: The export.
        :raises MissingPullRequestFieldError: If any record omits a required field.
        """
        return cls(tuple(cls._pull_request(record) for record in records))

    @staticmethod
    def _pull_request(record: PullRequestRecord) -> PullRequest:
        """Read one fetched pull request into a board entry.

        :param record: The fetched pull request.
        :return: The board entry.
        :raises MissingPullRequestFieldError: If a required field is absent.
        """
        number = int(PullRequestField.NUMBER.read(record))
        return PullRequest(
            number=number,
            head=PullRequestField.HEAD.read(record, number),
            base=PullRequestField.BASE.read(record, number),
            draft=bool(PullRequestField.DRAFT.read(record, number)),
            labels=PullRequestField.LABELS.read(record, number),
            ci=record.get("ci"),
            session=get_session_link_in(PullRequestField.BODY.read(record, number)),
        )

    def as_json(self) -> str:
        """:return: The export, in the document :func:`stack.load_board` parses."""
        return json.dumps(
            {"pull_requests": [asdict(entry) for entry in self.pull_requests]},
            indent=2,
        )

    def write(self, path: Path = BOARD_PATH) -> Path:
        """Write the export where the derived stack is read from.

        :param path: Where to write it.
        :return: The path written to.
        """
        path.write_text(self.as_json() + "\n")
        return path


# %% reading and writing the fork's pull requests


@dataclass
class GitHubCredentialUnavailableError(RuntimeError):
    """Raised when no token is available to authenticate the API calls with."""

    variables: tuple[str, ...]
    """The environment variables that were consulted."""

    def __str__(self) -> str:
        """:return: What was looked for, so the caller can supply it."""
        return (
            f"no GitHub token: set one of {', '.join(self.variables)}, or run this "
            f"with a caller that has one"
        )


class PullRequestReader(Protocol):
    """Reading the pull-request state a pass derives from."""

    def open_pull_requests(self) -> list[PullRequestRecord]:
        """:return: Every open pull request on the fork."""

    def pull_request(self, number: int) -> PullRequestRecord:
        """:param number: The pull request to read.
        :return: That pull request."""


class PullRequestWriter(Protocol):
    """The three writes a pass makes, each one probed against the live API first.

    Every one of them is available to the credential a session carries; a pull request's
    *base branch* is the single write that is not, which is why reparenting is the
    caller's job and none of this is.
    """

    def replace_labels(self, number: int, labels: Sequence[str]) -> None:
        """:param number: The pull request to write.
        :param labels: The complete label set it must end up with."""

    def add_comment(self, number: int, body: str) -> str:
        """:param number: The pull request to comment on.
        :param body: The comment.
        :return: The comment's URL."""

    def set_description(self, number: int, body: str) -> None:
        """:param number: The pull request to write.
        :param body: The new description."""


class ForkPullRequests(PullRequestReader, PullRequestWriter, Protocol):
    """Everything a pass does to the fork's pull requests.

    A pass reads state and writes back to the same fork, so the two halves are named
    together wherever both are needed; the board export takes the reading half alone,
    which is what keeps an export from being able to write.
    """


@dataclass
class GitHubRequestFailed(ExternalCallFailed):
    """Raised when the API refuses a call this module depends on."""

    method: str = ""
    """The HTTP method used."""

    path: str = ""
    """The API path called, without the host."""

    @property
    def call(self) -> str:
        """:return: The request line, as issued."""
        return f"{self.method} {self.path}"


@dataclass(frozen=True)
class GitHubRepository:
    """Every pull-request call this executor makes, against one repository.

    ``gh`` is absent from the environment this normally runs in, so the calls are plain
    authenticated requests rather than a CLI wrapper.
    """

    repository: Repository
    """The repository to read and write."""

    token: str
    """The credential the requests authenticate with."""

    page_size: int = 100
    """How many pull requests to ask for per request."""

    @classmethod
    def from_environment(cls, repository: Repository) -> GitHubRepository:
        """Build a client from whichever credential the environment carries.

        :param repository: The repository to read and write.
        :return: The client.
        :raises GitHubCredentialUnavailableError: If no token is set.
        """
        for variable in CREDENTIAL_VARIABLES:
            token = os.environ.get(variable)
            if token:
                return cls(repository, token)
        raise GitHubCredentialUnavailableError(CREDENTIAL_VARIABLES)

    def open_pull_requests(self) -> list[PullRequestRecord]:
        """:return: Every open pull request on the repository, oldest page first."""
        collected: list[PullRequestRecord] = []
        page = 1
        while True:
            query = urllib.parse.urlencode(
                {"state": "open", "per_page": self.page_size, "page": page}
            )
            fetched = self._call("GET", f"/pulls?{query}")
            collected.extend(fetched)
            if len(fetched) < self.page_size:
                return collected
            page += 1

    def pull_request(self, number: int) -> PullRequestRecord:
        """:param number: The pull request to read.
        :return: That pull request."""
        return self._call("GET", f"/pulls/{number}")

    def replace_labels(self, number: int, labels: Sequence[str]) -> None:
        """Write a pull request's complete label set.

        :param number: The pull request to write.
        :param labels: The complete set it must end up with, computed by
            :meth:`stack.LabelWrite.replacing` - this call replaces rather than adds.
        """
        self._call("PUT", f"/issues/{number}/labels", {"labels": list(labels)})

    def add_comment(self, number: int, body: str) -> str:
        """:param number: The pull request to comment on.
        :param body: The comment.
        :return: The comment's URL."""
        created = self._call("POST", f"/issues/{number}/comments", {"body": body})
        return str(created["html_url"])

    def set_description(self, number: int, body: str) -> None:
        """Rewrite a pull request's description and nothing else.

        :param number: The pull request to write.
        :param body: The new description.
        """
        self._call("PATCH", f"/pulls/{number}", {"body": body})

    def _call(
        self, method: str, path: str, payload: Mapping[str, Any] | None = None
    ) -> Any:
        """Make one authenticated API call.

        :param method: The HTTP method.
        :param path: The path below the repository, starting with a slash.
        :param payload: The JSON body, absent for a read.
        :return: The decoded response.
        :raises GitHubRequestFailed: If the API answers with an error status.
        """
        request = urllib.request.Request(
            f"{GITHUB_API_ROOT}/repos/{self.repository}{path}",
            method=method,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as refused:
            raise GitHubRequestFailed(
                status=refused.code,
                detail=refused.read().decode(errors="replace"),
                method=method,
                path=path,
            ) from refused


# %% fast-forwarding the fork's copy of the upstream base


class FastForwardOutcome(StrEnum):
    """What became of the fork's base branch."""

    PUSHED = "pushed"
    """It was moved onto the upstream's tip."""

    ALREADY_CURRENT = "already-current"
    """It already pointed at the upstream's tip."""

    REFUSED_NOT_FAST_FORWARD = "refused-not-fast-forward"
    """It carries commits the upstream does not, so moving it would discard them."""


@dataclass(frozen=True)
class FastForwardReport:
    """What the fast-forward did, and to what."""

    outcome: FastForwardOutcome
    """What became of the fork's base branch."""

    upstream_reference: str
    """The upstream ref the fork's base was compared against."""

    fork_reference: str
    """The fork ref that was to be moved."""

    commit: str
    """The commit the fork's base points at now."""

    explanation: str | None = None
    """Why a refusal was refused, absent when nothing was refused."""


def fast_forward(
    configuration: Configuration, git: GitCommandRunner
) -> FastForwardReport:
    """Move the fork's copy of the upstream base onto the upstream's tip.

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


# %% restacking every branch whose parent moved


class RestackOutcome(StrEnum):
    """What became of one branch during a restack."""

    PUSHED = "pushed"
    """Its parent was integrated and the result published."""

    UP_TO_DATE = "up-to-date"
    """Its parent's tip was already contained in it."""

    CONFLICT = "conflict"
    """Its parent could not be integrated cleanly; nothing was published."""

    INTEGRATION_FAILED = "integration-failed"
    """Integrating its parent failed without conflicting on anything, so the branch is
    not the thing to fix and its owner was not told; nothing was published."""

    REFUSED = "refused"
    """Move check refused the push; nothing was published."""

    PUSH_REJECTED = "push-rejected"
    """The fork rejected the push, so the branch moved under this pass; nothing was
    published, and nothing was forced over whatever moved it."""

    WITHHELD = "withheld"
    """It is still conflicted against its base from a previous pass, so it was left
    untouched rather than re-reported."""


@dataclass(frozen=True)
class BranchOutcome:
    """What became of one branch, in terms its owner can act on."""

    branch: str
    """The branch this is about."""

    parent: str
    """The branch whose tip was to be integrated into it."""

    strategy: IntegrationStrategy
    """How the parent was to be integrated."""

    outcome: RestackOutcome
    """What became of it."""

    conflicting_paths: tuple[str, ...] = ()
    """The paths that conflicted, empty unless the outcome is a conflict."""

    refusals: tuple[RefusalReason, ...] = ()
    """Why the push was refused, empty unless the outcome is a refusal."""

    pushed_commit: str | None = None
    """The commit published, absent unless the outcome is a push."""

    explanation: str | None = None
    """Why this outcome happened in words its owner can act on, absent unless the
    outcome carries one."""

    reported_at: str | None = None
    """URL of the comment telling this branch's owner about it, absent unless one was
    posted."""


CONFLICT_COMMENT_PREFIX = "🔴 ROUTINE - NEEDS RESOLUTION:"
"""Opens the comment a conflict is reported in, so the branch's owner can find every one
of them at a glance."""

MERGEABLE_STATE_WITH_CONFLICTS = "dirty"
"""The one ``mergeable_state`` meaning a branch genuinely conflicts with its base.
Everything else - ``clean``, ``unstable``, ``blocked``, ``behind``, ``has_hooks``,
``unknown`` - means there are no conflicts, whatever else may be true of it."""


def conflict_report(
    branch: Branch, conflicting_paths: Sequence[str], parent: str
) -> str:
    """Write the comment telling a branch's owner that their branch needs them.

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
    """One branch's restack, and everything a step needs to carry it out."""

    branch: Branch
    """The branch being restacked."""

    parent: str
    """The branch whose tip is to be integrated into it."""

    strategy: IntegrationStrategy
    """How that parent is to be integrated, which is also what authorises a rewrite."""

    stack: Stack
    """The derived stack it belongs to."""

    git: GitCommandRunner
    """The runner to execute through."""

    fork: ForkPullRequests
    """The fork, read for conflict state and written to when reporting."""

    checks: CommitMoveChecks
    """The checks its push is put through."""

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
        """Finish this branch with an outcome its owner can act on.

        :param outcome: What became of it.
        :param detail: Whatever that outcome carries.
        :return: The outcome, naming this branch and its parent.
        """
        return BranchOutcome(
            self.branch.name, self.parent, self.strategy, outcome, **detail
        )


class RestackStep:
    """One step of a branch's restack.

    A step either concludes the branch - returning the outcome its owner acts on - or
    returns nothing and lets the next step run. Adding a step is writing a subclass and
    placing it in :data:`RESTACK_STEPS`, whose order is the procedure.
    """

    def attempt(self, restacking: BranchUnderRestack) -> BranchOutcome | None:
        """Carry out this step.

        :param restacking: The branch being restacked.
        :return: The outcome concluding the branch, or ``None`` to continue.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class WithholdBranchStillConflicting(RestackStep):
    """Leaves a branch alone while it is still conflicted from an earlier pass.

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
    """Leaves a branch alone when its parent's tip is already contained in it."""

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
    """Integrates the parent's tip, reporting a conflict to the branch's owner.

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
        """Tell the branch's owner, and label it so the next pass withholds it.

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
    """Puts the push through the checks before it is made, without exception."""

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
    """Publishes the integrated branch, reporting rather than forcing a rejection."""

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


RESTACK_STEPS: tuple[RestackStep, ...] = (
    WithholdBranchStillConflicting(),
    SkipBranchAlreadyCurrent(),
    IntegrateParent(),
    RefuseAnUnsafeMove(),
    PublishBranch(),
)
"""Every step a branch is put through, in the order that is the procedure.

Unlike :data:`COMMANDS`, these are listed rather than found from their own subclasses: a
branch is published only once its move has been checked, so this order is a decision
about what a pass does, not bookkeeping. Stating it here keeps it where it is read,
rather than making it a consequence of where the classes happen to be defined.
"""


@dataclass
class RestackConcludedNothingError(RuntimeError):
    """Raised when no step concluded a branch, which the last step always must."""

    branch: str
    """The branch left without an outcome."""

    def __str__(self) -> str:
        """:return: Which branch was left unconcluded."""
        return f"no restack step concluded '{self.branch}'"


@dataclass(frozen=True)
class DetachedCheckout:
    """The invoking checkout, detached so its branch can be restacked elsewhere.

    git refuses to check one branch out in two worktrees at once, and the caller of a
    pass is usually sitting on a branch of the stack. Detaching releases the name while
    changing nothing else - same commit, same files, same work in progress - and the
    branch is checked out again afterwards, which is also how the caller picks up a
    restack of their own branch.
    """

    git: GitCommandRunner
    """The invoking checkout."""

    branch: str
    """The branch it was on, empty when it was already detached."""

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
        """Put the checkout back on the branch it was on.

        Attempted rather than depended on, so failing to restore never replaces the
        exception on its way out; the checkout is then left detached at the commit it
        started from, with everything it was carrying.
        """
        if self.branch:
            self.git.attempt("checkout", "--quiet", self.branch)


@dataclass(frozen=True)
class RestackWorktree:
    """A checkout of its own for the branch switching a restack does.

    Every step of the pass shells out to this file, which is tracked content in the
    checkout the pass is invoked from. Most branches in a stack were cut before that
    tooling landed, so checking one out there deletes the tooling the rest of the pass
    needs and leaves the caller on a branch that is not theirs. This worktree is added
    outside the project instead, out of reach of the branches a restack switches to.

    Its refs are the same refs, so a branch it moves is moved for the whole repository.
    """

    git: GitCommandRunner
    """The runner every branch switch of a restack goes through."""

    origin: GitCommandRunner
    """The invoking checkout, which the worktree is added to and removed from."""

    @classmethod
    def added_to(cls, origin: GitCommandRunner) -> RestackWorktree:
        """Add a worktree, detached at whatever the invoking checkout has.

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
        """Remove the worktree, whether the restack finished or was abandoned.

        Removal is attempted rather than depended on, so a failure to tidy up never
        replaces the exception that is on its way out.
        """
        self.origin.attempt(
            "worktree", "remove", "--force", str(self.git.working_directory)
        )


def restack(
    stack: Stack, git: GitCommandRunner, fork: ForkPullRequests
) -> list[BranchOutcome]:
    """Put every branch whose parent moved through :data:`RESTACK_STEPS`, bottom up.

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


@dataclass(frozen=True)
class ProposedPush:
    """One publication, and whether it is authorised to overwrite what is published.

    Every push this module makes is built here, so whether history may be rewritten is
    decided once rather than at each call.
    """

    remote: str
    """The remote to publish to."""

    refspec: str
    """What to publish, as ``<source>:<destination>``."""

    with_lease: bool = False
    """Whether published history may be overwritten, and then only if the remote is
    where this checkout last saw it."""

    @classmethod
    def publishing(
        cls, configuration: Configuration, branch: str, strategy: IntegrationStrategy
    ) -> ProposedPush:
        """Build the push that publishes a restacked branch.

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


# %% promoting every approved unblocked branch

PROMOTION_HEADING = "## Promote"
"""Heading the compare-and-create link is written under, in the fork pull request's own
description - the summary that carried it is delivered once and then gone, and the
description is still there a week later."""

PROMOTION_LINK_LABEL = "cram2-link-sent"
"""Marks a branch whose link has been built, so a later pass does not rebuild it."""


@dataclass(frozen=True)
class Promotion:
    """One branch's compare-and-create link, and where it was recorded."""

    branch: str
    """The branch promoted."""

    pull_request_number: int
    """Its fork pull request."""

    url: str
    """The compare-and-create link opening the upstream pull request."""

    body_was_truncated: bool
    """Whether the prefilled description had to be shortened to fit the URL limit."""


def description_with_promotion_link(description: str, url: str) -> str:
    """Put a promotion link into a description, replacing any already there.

    :param description: The pull request's current description.
    :param url: The link to record.
    :return: The description to write back.
    """
    before, _, _ = description.partition(PROMOTION_HEADING)
    return f"{before.rstrip()}\n\n{PROMOTION_HEADING}\n\n{url}\n"


def promotion_summary(description: str) -> str:
    """Take the one paragraph of a description that prefills the upstream pull request.

    A compare URL discards an over-long prefill silently, so the whole description is
    never sent - the link back to the fork pull request carries the rest.

    :param description: The fork pull request's description.
    :return: Its first paragraph, empty if it has none.
    """
    before, _, _ = description.partition(PROMOTION_HEADING)
    paragraphs = [block.strip() for block in before.split("\n\n") if block.strip()]
    return paragraphs[0] if paragraphs else ""


def promote(stack: Stack, fork: ForkPullRequests) -> list[Promotion]:
    """Build and record the upstream link for every branch ready to be promoted.

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
    """Build what the upstream pull request opens with.

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
    """Drop the link label from every branch whose link has already been acted on.

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


# %% the report a caller renders or emits


@dataclass(frozen=True)
class MaintenanceReport:
    """Everything one pass did, and the one thing it leaves for its caller.

    ``reparents`` is that one thing: retargeting a base is the single write GitHub
    refuses to the credential this runs on, so it is reported rather than performed.

    Every field defaults to nothing done, so a single command reports the part of the
    pass it performed through the same object - and therefore through the same exit
    status - as a whole pass does.
    """

    fast_forward: FastForwardReport | None = None
    """What became of the fork's base branch, absent when it was not attempted."""

    restacked: tuple[BranchOutcome, ...] = ()
    """What became of each branch in the restack plan."""

    promoted: tuple[Promotion, ...] = ()
    """The branches whose upstream link was built and recorded this pass."""

    promotion_labels_cleared: tuple[str, ...] = ()
    """The branches whose spent link label was removed this pass."""

    reparents: tuple[Reparent, ...] = ()
    """The children whose base has landed, for the caller to retarget - the one step
    this cannot perform itself."""

    landed: tuple[str, ...] = ()
    """The branches whose own commits are already in the upstream base."""

    promotable: tuple[str, ...] = ()
    """The branches approved and unblocked, whether or not a link was built this pass."""

    def as_json(self) -> str:
        """:return: The report as one machine-readable document."""
        status = exit_code_for(self)
        return json.dumps(
            {
                "status": status.name_for_a_caller,
                "exit_code": int(status),
                **asdict(self),
            },
            indent=2,
        )

    @property
    def branches_left_unpublished(self) -> tuple[BranchOutcome, ...]:
        """:return: Every branch the pass could not leave in the state it wanted."""
        return tuple(
            outcome
            for outcome in self.restacked
            if outcome.outcome not in {RestackOutcome.PUSHED, RestackOutcome.UP_TO_DATE}
        )

    @property
    def fast_forward_was_refused(self) -> bool:
        """:return: Whether the fork's base was left behind the upstream."""
        return (
            self.fast_forward is not None
            and self.fast_forward.outcome is FastForwardOutcome.REFUSED_NOT_FAST_FORWARD
        )


def build_report(
    stack: Stack,
    fast_forward_report: FastForwardReport | None,
    restacked: Sequence[BranchOutcome],
    promoted: Sequence[Promotion] = (),
    promotion_labels_cleared: Sequence[str] = (),
) -> MaintenanceReport:
    """Assemble one pass's outcomes and its leftovers into a single report.

    :param stack: The derived stack, read for what the caller still has to do.
    :param fast_forward_report: What became of the fork's base branch, if attempted.
    :param restacked: What became of each branch in the restack plan.
    :param promoted: The branches whose upstream link was built this pass.
    :param promotion_labels_cleared: The branches whose spent link label was removed.
    :return: The report.
    """
    return MaintenanceReport(
        fast_forward=fast_forward_report,
        restacked=tuple(restacked),
        promoted=tuple(promoted),
        promotion_labels_cleared=tuple(promotion_labels_cleared),
        reparents=tuple(reparents(stack)),
        landed=tuple(branch.name for branch in landed_branches(stack)),
        promotable=tuple(branch.name for branch in promotion_order(stack)),
    )


# %% printing


def print_board_export(export: BoardExport, written_to: Path | None) -> None:
    """Report what the export contains, and where it went.

    :param export: The export.
    :param written_to: Where it was written, or ``None`` when it was only printed.
    """
    if written_to is None:
        print(export.as_json())
        return
    print(f"{len(export.pull_requests)} open pull request(s) -> {written_to}")


def print_fast_forward(report: FastForwardReport) -> None:
    """:param report: What became of the fork's base branch."""
    print(f"{report.fork_reference}\t{report.outcome}\t{report.commit}")
    if report.explanation:
        print(report.explanation, file=sys.stderr)


def print_restack(outcomes: Sequence[BranchOutcome]) -> None:
    """:param outcomes: What became of each branch."""
    for outcome in outcomes:
        detail = (
            ",".join(outcome.conflicting_paths)
            or ",".join(outcome.refusals)
            or outcome.pushed_commit
            or outcome.explanation
            or ""
        )
        print(f"{outcome.branch}\t{outcome.outcome}\t{detail}")


def print_promotions(promoted: Sequence[Promotion], cleared: Sequence[str]) -> None:
    """:param promoted: The branches whose link was built this pass.
    :param cleared: The branches whose spent link label was removed."""
    for promotion in promoted:
        print(f"{promotion.branch}\t#{promotion.pull_request_number}\t{promotion.url}")
        if promotion.body_was_truncated:
            print(
                f"{promotion.branch}: the prefilled description was shortened to fit "
                f"the URL limit",
                file=sys.stderr,
            )
    for branch in cleared:
        print(f"{branch}\tlink-label-cleared\t")


# %% entry point


class MaintenanceExitCode(IntEnum):
    """What this executor's exit status tells a caller.

    The first five match :class:`stack.ExitCode` value for value and meaning, so a
    caller acting on the two tools' statuses never has to remember which produced one.
    """

    SUCCESS = 0
    """The command ran and did what it reports."""

    USAGE = 2
    """No such command, or the wrong arguments."""

    BOARD_UNAVAILABLE = 3
    """``board.json`` is missing, so the stack cannot be derived."""

    REMOTES_UNRESOLVED = 4
    """The fork could not be identified from this checkout's remotes."""

    MOVE_REFUSED = 5
    """A push was refused; the reasons are in the report."""

    GIT_COMMAND_FAILED = 6
    """A git command the run depended on failed; nothing further was attempted."""

    NOT_FAST_FORWARD = 7
    """The fork's base carries commits the upstream does not."""

    CREDENTIAL_UNAVAILABLE = 8
    """No GitHub token is set, so the fork cannot be read or written."""

    GITHUB_REQUEST_FAILED = 9
    """The API refused a call this pass depends on; its status and reason are on
    stderr."""

    BRANCH_NEEDS_ATTENTION = 10
    """The pass itself ran, but left at least one branch unpublished for somebody to
    act on - a conflict, a withheld branch, or a push the fork rejected. Distinct from
    a move check refusal, which is a fault in the move rather than in the branch."""

    @property
    def name_for_a_caller(self) -> str:
        """What this status means, in words rather than as a number to be looked up.

        A process exit status can only ever be an integer, so this accompanies the
        number rather than replacing it. Derived from the member itself, so a status
        can never end up carrying a name that belongs to a different one.

        :return: The status's name, in the form a caller reads or matches on.
        """
        return self.name.lower().replace("_", "-")


def exit_code_for(report: MaintenanceReport) -> MaintenanceExitCode:
    """Decide one pass's exit status from what it actually left behind.

    Shared by every command that produces a report, so none of them can disagree about
    what counts as a clean pass - a refused fast-forward reported as success is exactly
    the kind of silence this exists to prevent.

    :param report: What the pass did.
    :return: The process exit code.
    """
    if report.fast_forward_was_refused:
        return MaintenanceExitCode.NOT_FAST_FORWARD
    unpublished = report.branches_left_unpublished
    if any(outcome.outcome is RestackOutcome.REFUSED for outcome in unpublished):
        return MaintenanceExitCode.MOVE_REFUSED
    if unpublished:
        return MaintenanceExitCode.BRANCH_NEEDS_ATTENTION
    return MaintenanceExitCode.SUCCESS


@dataclass(frozen=True)
class MaintenancePass:
    """What one run has resolved so far, built lazily as a command asks for it.

    The board is derived before the credential is resolved, so a caller missing both is
    sent after the board - the thing the previous command produces - rather than after a
    token that would not help them yet.
    """

    configuration: Configuration
    """The resolved configuration naming both repositories and every label."""

    git: GitCommandRunner
    """The runner every git command goes through."""

    def fork(self) -> GitHubRepository:
        """:return: The fork, as this run's credential can read and write it."""
        return GitHubRepository.from_environment(self.configuration.fork_repository)

    def stack(self) -> Stack:
        """:return: The derived stack, read from the exported board."""
        return load_stack()


@dataclass(frozen=True)
class MaintenanceCommand:
    """One command this executor answers.

    A command owns its own name, its own flags and what it does, so adding one is
    writing a subclass - :data:`COMMANDS` finds it, and nothing else has to be told it
    exists.
    """

    invoked_as: ClassVar[str]
    """The name it is invoked by on the command line."""

    description: ClassVar[str]
    """What it does, as ``--help`` puts it."""

    def declare_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Declare this command's own flags.

        :param parser: The subparser to declare them on.
        """

    def run(
        self, maintenance: MaintenancePass, arguments: argparse.Namespace
    ) -> MaintenanceExitCode:
        """Perform the command.

        :param maintenance: What this run has resolved.
        :param arguments: The parsed command line.
        :return: The process exit code.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class BoardCommand(MaintenanceCommand):
    """Fetches the fork's open pull requests and exports them as the board."""

    invoked_as: ClassVar[str] = "board"
    description: ClassVar[str] = "export the fork's open pull requests"

    def declare_arguments(self, parser: argparse.ArgumentParser) -> None:
        """:param parser: The subparser to declare ``--write`` on."""
        parser.add_argument(
            "--write",
            action="store_true",
            help="write board.json rather than printing the export",
        )

    def run(
        self, maintenance: MaintenancePass, arguments: argparse.Namespace
    ) -> MaintenanceExitCode:
        """:param maintenance: What this run has resolved.
        :param arguments: The parsed command line.
        :return: The process exit code."""
        export = BoardExport.from_api_records(maintenance.fork().open_pull_requests())
        print_board_export(export, export.write() if arguments.write else None)
        return MaintenanceExitCode.SUCCESS


@dataclass(frozen=True)
class FastForwardCommand(MaintenanceCommand):
    """Moves the fork's base branch onto the upstream, refusing to force."""

    invoked_as: ClassVar[str] = "fast-forward"
    description: ClassVar[str] = "move the fork's base branch onto the upstream"

    def run(
        self, maintenance: MaintenancePass, arguments: argparse.Namespace
    ) -> MaintenanceExitCode:
        """:param maintenance: What this run has resolved.
        :param arguments: The parsed command line.
        :return: The process exit code."""
        report = fast_forward(maintenance.configuration, maintenance.git)
        print_fast_forward(report)
        return exit_code_for(MaintenanceReport(fast_forward=report))


@dataclass(frozen=True)
class RestackCommand(MaintenanceCommand):
    """Integrates every moved parent and publishes what merged cleanly."""

    invoked_as: ClassVar[str] = "restack"
    description: ClassVar[str] = "integrate every moved parent and publish the result"

    def run(
        self, maintenance: MaintenancePass, arguments: argparse.Namespace
    ) -> MaintenanceExitCode:
        """:param maintenance: What this run has resolved.
        :param arguments: The parsed command line.
        :return: The process exit code."""
        stack = maintenance.stack()
        outcomes = restack(stack, maintenance.git, maintenance.fork())
        print_restack(outcomes)
        return exit_code_for(MaintenanceReport(restacked=tuple(outcomes)))


@dataclass(frozen=True)
class PromoteCommand(MaintenanceCommand):
    """Records the upstream link on every branch ready to be promoted."""

    invoked_as: ClassVar[str] = "promote"
    description: ClassVar[str] = "record the upstream link on every promotable branch"

    def run(
        self, maintenance: MaintenancePass, arguments: argparse.Namespace
    ) -> MaintenanceExitCode:
        """:param maintenance: What this run has resolved.
        :param arguments: The parsed command line.
        :return: The process exit code."""
        stack = maintenance.stack()
        fork = maintenance.fork()
        print_promotions(
            promote(stack, fork), clear_spent_promotion_labels(stack, fork)
        )
        return MaintenanceExitCode.SUCCESS


@dataclass(frozen=True)
class RunReportCommand(MaintenanceCommand):
    """Performs the whole pass and reports it as one document."""

    invoked_as: ClassVar[str] = "run-report"
    description: ClassVar[str] = "perform the whole pass and report it"

    def declare_arguments(self, parser: argparse.ArgumentParser) -> None:
        """:param parser: The subparser to declare ``--json`` on."""
        parser.add_argument(
            "--json",
            action="store_true",
            help="emit the machine-readable document rather than a summary",
        )

    def run(
        self, maintenance: MaintenancePass, arguments: argparse.Namespace
    ) -> MaintenanceExitCode:
        """Perform every step of the pass, then discard the board it derived from.

        The board is a snapshot of one moment's open pull requests, and a stale one read
        by a later run is worse than none at all - so a whole pass ends without one, and
        the next begins by exporting a fresh one.

        :param maintenance: What this run has resolved.
        :param arguments: The parsed command line.
        :return: The process exit code.
        """
        stack = maintenance.stack()
        fork = maintenance.fork()
        fast_forward_report = fast_forward(stack.configuration, maintenance.git)
        report = build_report(
            stack,
            fast_forward_report,
            restack(stack, maintenance.git, fork),
            promote(stack, fork),
            clear_spent_promotion_labels(stack, fork),
        )
        BOARD_PATH.unlink(missing_ok=True)
        if arguments.json:
            print(report.as_json())
        else:
            print_fast_forward(fast_forward_report)
            print_restack(report.restacked)
            print_promotions(report.promoted, report.promotion_labels_cleared)
        return exit_code_for(report)


COMMANDS: tuple[MaintenanceCommand, ...] = tuple(
    subclass() for subclass in MaintenanceCommand.__subclasses__()
)
"""Every command this executor answers, found from the subclasses themselves so a
command cannot exist without being reachable, in the order they are defined."""


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
    """Run the command line and say, in words, what its status means.

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
    """Run the requested command, mapping every refusal to its own status.

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
