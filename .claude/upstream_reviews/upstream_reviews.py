#!/usr/bin/env python3
"""
Report the review threads a fork's pull request has collected upstream.

Thread resolved-state is only exposed by GitHub's GraphQL API, which is unreachable from
a Claude session, so this runs in the fork's own GitHub Actions runner and the session
reads its job log. ``gh`` is what talks to GitHub, rather than a hand-rolled client:
runners ship it and ``GITHUB_TOKEN`` authenticates it, so no access rule is implemented
here a fourth time.

Every repository name comes from configuration or the runner's own environment, so the
script works unchanged in any contributor's fork.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar, ClassVar

# ``stack.py`` is a single-file script rather than an installed package, so its
# directory joins the path the same way the test suites do it. Reusing its
# ``Repository`` keeps one parser for ``owner/name`` references.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stack"))

import tomllib  # noqa: E402
from stack import CONFIGURATION_PATH, Repository  # noqa: E402


QUERY_DIRECTORY = Path(__file__).resolve().parent / "queries"
"""
Where the ``.graphql`` documents live.
"""


# %% the reading contract


class JSONModel(ABC):
    """
    A dataclass mirroring one object GitHub returns, able to read itself from it.

    Declaring the reader here is what lets :meth:`PullRequestJSONKey.read_list` call
    it on any model. Without it the shared name would be a convention every model
    is trusted to have followed, and a model that spelled it differently would fail
    only when something happened to parse that field.

    ..note:: A reader needing more than the object itself cannot state this
        contract, so :class:`PullRequestReviewSnapshot`, which is assembled from
        threads gathered across several responses, does not implement it.
    """

    @classmethod
    @abstractmethod
    def from_json(cls, data: dict[str, Any]) -> JSONModel:
        """
        Read one of these out of the object GitHub returned.

        :param data: The object to read.
        :return: The parsed model.
        """


ParsedItem = TypeVar("ParsedItem", bound=JSONModel)
"""
Whatever model a list-valued field is being parsed into.
"""


# %% wire vocabulary


class PullRequestJSONKey(StrEnum):
    """
    Every field name this script reads out of a GraphQL response.

    Named once here so a key is never spelled twice, and so a rename in the query
    document has exactly one place to follow.
    """

    DATA = "data"
    ERRORS = "errors"
    MESSAGE = "message"
    REPOSITORY = "repository"
    PULL_REQUEST = "pullRequest"
    PULL_REQUESTS = "pullRequests"
    NODES = "nodes"
    PAGE_INFO = "pageInfo"
    HAS_NEXT_PAGE = "hasNextPage"
    END_CURSOR = "endCursor"
    IDENTIFIER = "id"
    IS_RESOLVED = "isResolved"
    IS_OUTDATED = "isOutdated"
    PATH = "path"
    LINE = "line"
    COMMENTS = "comments"
    DATABASE_IDENTIFIER = "databaseId"
    AUTHOR = "author"
    LOGIN = "login"
    BODY = "body"
    CREATED_AT = "createdAt"
    URL = "url"
    REVIEWS = "reviews"
    REVIEW_THREADS = "reviewThreads"
    STATE = "state"
    SUBMITTED_AT = "submittedAt"
    NUMBER = "number"
    TITLE = "title"
    HEAD_REPOSITORY_OWNER = "headRepositoryOwner"
    QUERY = "query"
    VARIABLES = "variables"

    def read_list(
        self, data: dict[str, Any], model: type[ParsedItem]
    ) -> list[ParsedItem]:
        """
        Parse every entry of the list-valued field this key names.

        GitHub returns a list-valued field as an object holding the array under
        ``nodes`` rather than as the array itself, so that step happens here rather
        than at each call site.

        :param data: The object the field belongs to.
        :param model: The model to parse each entry into.
        :return: The parsed entries, in the order GitHub returned them.
        """
        entries = data[self][PullRequestJSONKey.NODES]
        return [model.from_json(entry) for entry in entries]


class QueryVariable(StrEnum):
    """
    The variables the query documents declare.
    """

    OWNER = "owner"
    NAME = "name"
    HEAD_REF_NAME = "headRefName"
    NUMBER = "number"
    THREAD_CURSOR = "threadCursor"


class GraphQLDocument(StrEnum):
    """
    The query documents, stored as ``.graphql`` files beside this script.
    """

    PULL_REQUEST_FOR_BRANCH = "pull_request_for_branch"
    REVIEW_THREADS_PAGE = "review_threads_page"

    def read(self) -> str:
        """:return: The document's text."""
        return (QUERY_DIRECTORY / f"{self}.graphql").read_text()


class EnvironmentVariable(StrEnum):
    """
    Runner-supplied environment this script reads.
    """

    STEP_SUMMARY = "GITHUB_STEP_SUMMARY"
    REPOSITORY_OWNER = "GITHUB_REPOSITORY_OWNER"


class ReviewState(StrEnum):
    """
    The verdict a reviewer submitted with a review.
    """

    APPROVED = "APPROVED"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"
    COMMENTED = "COMMENTED"
    DISMISSED = "DISMISSED"
    PENDING = "PENDING"

    @property
    def spoken(self) -> str:
        """:return: The verdict as it reads in a sentence."""
        return self.replace("_", " ").lower()


class PullRequestState(StrEnum):
    """
    The lifecycle state GitHub reports for a pull request.
    """

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    MERGED = "MERGED"


class ThreadMarker(StrEnum):
    """
    The annotations a thread can carry in the report.
    """

    RESOLVED = "resolved"
    OUTDATED = "outdated"


# %% errors


@dataclass
class UpstreamReviewError(Exception, ABC):
    """
    Base class for every failure this script raises.
    """

    def __post_init__(self) -> None:
        Exception.__init__(self, self.describe())

    @abstractmethod
    def describe(self) -> str:
        """:return: The message a caller should see."""


@dataclass
class GitHubCommandFailed(UpstreamReviewError):
    """
    Raised when the ``gh`` invocation itself exits non-zero.
    """

    executable: str
    """
    The command that was run.
    """

    exit_code: int
    """
    The status it exited with.
    """

    error_output: str
    """
    Whatever it wrote to standard error.
    """

    def describe(self) -> str:
        """:return: The message a caller should see."""
        return f"{self.executable} exited {self.exit_code}: {self.error_output}"


@dataclass
class GraphQLErrorsReturned(UpstreamReviewError):
    """
    Raised when GitHub answers with an ``errors`` array instead of data.
    """

    messages: list[str]
    """
    Every message GitHub reported.
    """

    def describe(self) -> str:
        """:return: The message a caller should see."""
        return "; ".join(self.messages)


@dataclass
class UpstreamPullRequestNotFound(UpstreamReviewError):
    """
    Raised when a branch has no pull request open on the upstream.
    """

    branch: str
    """
    The fork branch that was looked up.
    """

    upstream: Repository
    """
    The repository that was searched.
    """

    fork_owner: str
    """
    The owner whose head branches were being claimed.
    """

    def describe(self) -> str:
        """:return: The message a caller should see."""
        return (
            f"no pull request on {self.upstream} has head "
            f"'{self.fork_owner}:{self.branch}' - the branch has most likely not "
            "been promoted upstream yet"
        )


# %% models mirroring the data


@dataclass(frozen=True)
class Author(JSONModel):
    """
    Whoever wrote a comment or submitted a review.
    """

    login: str
    """
    Their GitHub login, or a placeholder when the account is gone.
    """
    UNKNOWN_LOGIN: ClassVar[str] = "(unknown)"
    """
    Stands in for the author of a comment whose account no longer exists.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> Author:
        """
        Read an author, tolerating the null GitHub returns for deleted users.

        :param data: The ``author`` object, which GitHub may report as null.
        :return: The parsed author.
        """
        if data is None:
            return cls(cls.UNKNOWN_LOGIN)
        return cls(data[PullRequestJSONKey.LOGIN])


@dataclass(frozen=True)
class ThreadComment(JSONModel):
    """
    One comment inside a review thread.
    """

    database_identifier: int
    """
    GitHub's numeric identifier, the one that appears in comment permalinks.
    """

    author: Author
    """
    Whoever wrote the comment.
    """

    body: str
    """
    The comment text, exactly as written.
    """

    created_at: str
    """
    When the comment was posted, as an ISO 8601 timestamp.
    """

    url: str
    """
    The permalink to this comment.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ThreadComment:
        """
        Build a comment from one ``comments`` node.

        :param data: The node to read.
        :return: The parsed comment.
        """
        return cls(
            database_identifier=data[PullRequestJSONKey.DATABASE_IDENTIFIER],
            author=Author.from_json(data[PullRequestJSONKey.AUTHOR]),
            body=data[PullRequestJSONKey.BODY],
            created_at=data[PullRequestJSONKey.CREATED_AT],
            url=data[PullRequestJSONKey.URL],
        )


@dataclass(frozen=True)
class ReviewThread(JSONModel):
    """
    A conversation anchored to one location in the pull request's diff.
    """

    identifier: str
    """
    GitHub's node identifier for the thread.
    """

    is_resolved: bool
    """
    Whether a reviewer has marked the thread resolved.
    """

    is_outdated: bool
    """
    Whether the diff hunk the thread was anchored to has since changed.
    """

    path: str
    """
    The file the thread is attached to.
    """

    line: int | None
    """
    The line the thread is attached to, absent once the hunk is outdated.
    """

    comments: list[ThreadComment]
    """
    Every comment in the thread, oldest first.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ReviewThread:
        """
        Build a thread from one ``reviewThreads`` node.

        :param data: The node to read.
        :return: The parsed thread.
        """
        return cls(
            identifier=data[PullRequestJSONKey.IDENTIFIER],
            is_resolved=data[PullRequestJSONKey.IS_RESOLVED],
            is_outdated=data[PullRequestJSONKey.IS_OUTDATED],
            path=data[PullRequestJSONKey.PATH],
            line=data[PullRequestJSONKey.LINE],
            comments=PullRequestJSONKey.COMMENTS.read_list(data, ThreadComment),
        )

    @property
    def location(self) -> str:
        """:return: The thread's ``path:line``, or just its path when it is outdated."""
        if self.line is None:
            return self.path
        return f"{self.path}:{self.line}"

    @property
    def markers(self) -> list[ThreadMarker]:
        """:return: The annotations this thread carries."""
        carried = []
        if self.is_resolved:
            carried.append(ThreadMarker.RESOLVED)
        if self.is_outdated:
            carried.append(ThreadMarker.OUTDATED)
        return carried


@dataclass(frozen=True)
class Review(JSONModel):
    """
    A submitted review, separate from the threads it may have opened.
    """

    author: Author
    """
    Whoever submitted the review.
    """

    state: ReviewState
    """
    The verdict the reviewer submitted.
    """

    body: str
    """
    The review's summary text, which is often empty.
    """

    submitted_at: str
    """
    When the review was submitted, as an ISO 8601 timestamp.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Review:
        """
        Build a review from one ``reviews`` node.

        :param data: The node to read.
        :return: The parsed review.
        """
        return cls(
            author=Author.from_json(data[PullRequestJSONKey.AUTHOR]),
            state=ReviewState(data[PullRequestJSONKey.STATE]),
            body=data[PullRequestJSONKey.BODY],
            submitted_at=data[PullRequestJSONKey.SUBMITTED_AT],
        )


@dataclass(frozen=True)
class BranchPullRequest(JSONModel):
    """
    One pull request found by searching the upstream for a head branch.
    """

    number: int
    """
    Its number on the upstream repository.
    """

    state: PullRequestState
    """
    Its lifecycle state.
    """

    head_owner: Author
    """
    The owner of the repository the head branch lives in.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> BranchPullRequest:
        """
        Build a summary from one ``pullRequests`` node.

        :param data: The node to read.
        :return: The parsed summary.
        """
        return cls(
            number=data[PullRequestJSONKey.NUMBER],
            state=PullRequestState(data[PullRequestJSONKey.STATE]),
            head_owner=Author.from_json(data[PullRequestJSONKey.HEAD_REPOSITORY_OWNER]),
        )


@dataclass(frozen=True)
class ReviewThreadPage(JSONModel):
    """
    One page of review threads, with the cursor that follows it.
    """

    threads: list[ReviewThread]
    """
    The threads on this page.
    """

    has_next_page: bool
    """
    Whether another page follows.
    """

    end_cursor: str | None
    """
    The cursor to request that next page with.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ReviewThreadPage:
        """
        Build a page from a ``pullRequest`` node.

        :param data: The node to read.
        :return: The parsed page.
        """
        threads_field = data[PullRequestJSONKey.REVIEW_THREADS]
        page_info = threads_field[PullRequestJSONKey.PAGE_INFO]
        return cls(
            threads=PullRequestJSONKey.REVIEW_THREADS.read_list(data, ReviewThread),
            has_next_page=page_info[PullRequestJSONKey.HAS_NEXT_PAGE],
            end_cursor=page_info[PullRequestJSONKey.END_CURSOR],
        )


@dataclass(frozen=True)
class PullRequestReviewSnapshot:
    """
    Everything read from one upstream pull request in a single run.
    """

    number: int
    """
    The pull request's number on the upstream repository.
    """

    title: str
    """
    The pull request's title.
    """

    url: str
    """
    The pull request's web URL.
    """

    reviews: list[Review]
    """
    Every submitted review, oldest first.
    """

    threads: list[ReviewThread]
    """
    Every review thread, in the order GitHub returned them.
    """

    @classmethod
    def from_json(
        cls, data: dict[str, Any], threads: list[ReviewThread]
    ) -> PullRequestReviewSnapshot:
        """
        Build a snapshot from a ``pullRequest`` node and its collected threads.

        :param data: The node to read.
        :param threads: Every thread gathered across the paged reads.
        :return: The parsed snapshot.
        """
        return cls(
            number=data[PullRequestJSONKey.NUMBER],
            title=data[PullRequestJSONKey.TITLE],
            url=data[PullRequestJSONKey.URL],
            reviews=PullRequestJSONKey.REVIEWS.read_list(data, Review),
            threads=threads,
        )

    @property
    def unresolved_threads(self) -> list[ReviewThread]:
        """:return: The threads still awaiting action."""
        return [thread for thread in self.threads if not thread.is_resolved]

    def thread(self, identifier: str) -> ReviewThread:
        """
        Look one thread up by its node identifier.

        :param identifier: The thread's node identifier.
        :return: The matching thread.
        :raises KeyError: If no thread carries that identifier.
        """
        for thread in self.threads:
            if thread.identifier == identifier:
                return thread
        raise KeyError(identifier)


@dataclass(frozen=True)
class RepositoryJSON(JSONModel):
    """
    The ``repository`` object every query in this script selects.

    Owns the one access path from a response's data down to the nodes the models parse,
    so no caller spells that path out again.
    """

    data: dict[str, Any]
    """
    The repository object as returned.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> RepositoryJSON:
        """
        Read the repository out of a response's ``data``.

        :param data: The response's ``data`` object.
        :return: The wrapped repository.
        """
        return cls(data[PullRequestJSONKey.REPOSITORY])

    @property
    def pull_request(self) -> dict[str, Any]:
        """:return: The single ``pullRequest`` this response selected."""
        return self.data[PullRequestJSONKey.PULL_REQUEST]

    @property
    def review_thread_page(self) -> ReviewThreadPage:
        """:return: The pull request's page of review threads."""
        return ReviewThreadPage.from_json(self.pull_request)

    @property
    def pull_request_reviews(self) -> PullRequestReviewSnapshot:
        """:return: The pull request's review state, from this response alone.

        Only complete when the response carried every page of threads; a paged
        read assembles the snapshot itself from the threads it accumulated.
        """
        return PullRequestReviewSnapshot.from_json(
            self.pull_request, self.review_thread_page.threads
        )

    @property
    def branch_pull_requests(self) -> list[BranchPullRequest]:
        """:return: Every pull request the head-branch search matched."""
        return PullRequestJSONKey.PULL_REQUESTS.read_list(self.data, BranchPullRequest)


@dataclass(frozen=True)
class GraphQLResponse:
    """
    A parsed GraphQL response envelope.
    """

    data: dict[str, Any] | None
    """
    The ``data`` data, absent when the query failed outright.
    """

    errors: list[str] = field(default_factory=list)
    """
    Every message from an ``errors`` array, empty on success.
    """

    @classmethod
    def from_json(cls, text: str) -> GraphQLResponse:
        """
        Parse a response body.

        :param text: The raw JSON GitHub returned.
        :return: The parsed envelope.
        """
        body = json.loads(text)
        return cls(
            data=body.get(PullRequestJSONKey.DATA),
            errors=[
                error[PullRequestJSONKey.MESSAGE]
                for error in body.get(PullRequestJSONKey.ERRORS, [])
            ],
        )

    def result(self) -> dict[str, Any]:
        """:return: The query's result.

        :raises GraphQLErrorsReturned: If GitHub reported errors instead.
        """
        if self.errors:
            raise GraphQLErrorsReturned(self.errors)
        return self.data or {}


# %% client


class GraphQLClient(ABC):
    """
    Sends a GraphQL document to GitHub and returns its ``data`` data.
    """

    @abstractmethod
    def execute(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """
        Run one GraphQL query.

        :param query: The GraphQL document.
        :param variables: The document's variables.
        :return: The response's ``data`` data.
        """


@dataclass
class GitHubCommandLineClient(GraphQLClient):
    """
    A client that shells out to ``gh api graphql``.

    The runner already ships ``gh`` and authenticates it from ``GITHUB_TOKEN``, so this
    holds no credential handling of its own.
    """

    executable: str = "gh"
    """
    The command to invoke, overridable for testing.
    """

    def execute(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """
        Run one GraphQL query through ``gh``.

        :param query: The GraphQL document.
        :param variables: The document's variables.
        :return: The response's ``data`` data.
        :raises GitHubCommandFailed: If ``gh`` exits non-zero.
        :raises GraphQLErrorsReturned: If GitHub answers with errors.
        """
        request = json.dumps(
            {PullRequestJSONKey.QUERY: query, PullRequestJSONKey.VARIABLES: variables}
        )
        completed = subprocess.run(
            [self.executable, "api", "graphql", "--input", "-"],
            input=request,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise GitHubCommandFailed(
                self.executable, completed.returncode, completed.stderr.strip()
            )
        return GraphQLResponse.from_json(completed.stdout).result()


# %% reading


@dataclass
class UpstreamReviewReader:
    """
    Reads one upstream pull request's review state through a client.
    """

    client: GraphQLClient
    """
    How GraphQL queries reach GitHub.
    """

    upstream_repository: Repository
    """
    The repository the fork's pull requests are opened against.
    """

    fork_owner: str
    """
    The owner whose branches this reader will claim as its own.
    """

    @property
    def _repository_variables(self) -> dict[str, Any]:
        """:return: The owner and name every query in this script takes."""
        return {
            QueryVariable.OWNER: self.upstream_repository.owner,
            QueryVariable.NAME: self.upstream_repository.name,
        }

    def resolve_pull_request_number(self, branch: str) -> int:
        """
        Find the upstream pull request opened from *branch*.

        Prefers an open pull request, falling back to the most recent closed one so a
        branch under post-merge discussion still resolves.

        :param branch: The fork branch name.
        :return: The upstream pull request number.
        :raises UpstreamPullRequestNotFound: If the fork has no such pull request.
        """
        data = self.client.execute(
            GraphQLDocument.PULL_REQUEST_FOR_BRANCH.read(),
            {**self._repository_variables, QueryVariable.HEAD_REF_NAME: branch},
        )
        candidates = [
            pull_request
            for pull_request in RepositoryJSON.from_json(data).branch_pull_requests
            if pull_request.head_owner.login == self.fork_owner
        ]
        if not candidates:
            raise UpstreamPullRequestNotFound(
                branch, self.upstream_repository, self.fork_owner
            )
        open_candidates = [
            pull_request
            for pull_request in candidates
            if pull_request.state is PullRequestState.OPEN
        ]
        return (open_candidates or candidates)[0].number

    def read_current_state(self, pull_request_number: int) -> PullRequestReviewSnapshot:
        """
        Read every review and review thread on one upstream pull request.

        :param pull_request_number: The upstream pull request's number.
        :return: The assembled snapshot.
        """
        threads: list[ReviewThread] = []
        cursor: str | None = None
        repository = None
        while True:
            data = self.client.execute(
                GraphQLDocument.REVIEW_THREADS_PAGE.read(),
                {
                    **self._repository_variables,
                    QueryVariable.NUMBER: pull_request_number,
                    QueryVariable.THREAD_CURSOR: cursor,
                },
            )
            repository = RepositoryJSON.from_json(data)
            page = repository.review_thread_page
            threads.extend(page.threads)
            if not page.has_next_page:
                break
            cursor = page.end_cursor
        return PullRequestReviewSnapshot.from_json(repository.pull_request, threads)


# %% configuration


def resolve_upstream_repository(
    path: Path = CONFIGURATION_PATH, override: str | None = None
) -> Repository:
    """
    Decide which repository the fork's pull requests are reviewed on.

    Reads the committed defaults only. The per-user layer lives on the personal-notes
    branch, which a runner does not check out, so *override* is the escape hatch for a
    checkout whose upstream differs.

    :param path: The committed stack configuration file.
    :param override: An ``owner/name`` reference outranking the file.
    :return: The upstream repository.
    """
    if override:
        return Repository.parse(override)
    values = tomllib.loads(path.read_text())
    return Repository.parse(values[UPSTREAM_REPOSITORY_SETTING])


UPSTREAM_REPOSITORY_SETTING = "upstream_repository"
"""
The ``stack.toml`` key naming the repository reviews happen on.
"""


# %% report


class ReportText(StrEnum):
    """
    The fixed lines the report states rather than computes.
    """

    REVIEWS_HEADING = "## Reviews"
    NO_REVIEWS = "No reviews submitted."
    NO_UNRESOLVED_HEADING = "## No unresolved review threads"
    NOTHING_TO_ACT_ON = "Nothing to act on."


@dataclass
class UnresolvedThreadReport:
    """
    Renders a pull request's current review state as the markdown a session or a phone
    reads.
    """

    current_pull_request_reviews: PullRequestReviewSnapshot
    """
    The review state to describe.
    """

    include_resolved: bool = False
    """
    Whether threads already marked resolved are shown too.
    """

    def render(self) -> str:
        """:return: The report as markdown."""
        lines = [
            f"# Upstream review: #{self.current_pull_request_reviews.number} {self.current_pull_request_reviews.title}",
            "",
            self.current_pull_request_reviews.url,
            "",
        ]
        lines.extend(self._render_reviews())
        lines.extend(self._render_threads())
        return "\n".join(lines)

    def heading(self, shown_count: int) -> str:
        """
        Describe the set actually being listed, not just the unresolved one.

        :param shown_count: How many threads the section goes on to list.
        :return: The section heading.
        """
        unresolved_count = len(self.current_pull_request_reviews.unresolved_threads)
        if not unresolved_count:
            return ReportText.NO_UNRESOLVED_HEADING
        if self.include_resolved:
            return f"## {shown_count} review threads, {unresolved_count} unresolved"
        return f"## {unresolved_count} unresolved review threads"

    @property
    def shown_threads(self) -> list[ReviewThread]:
        """:return: The threads this report lists."""
        if self.include_resolved:
            return self.current_pull_request_reviews.threads
        return self.current_pull_request_reviews.unresolved_threads

    def _render_reviews(self) -> list[str]:
        """:return: The submitted-reviews section."""
        if not self.current_pull_request_reviews.reviews:
            return [ReportText.REVIEWS_HEADING, "", ReportText.NO_REVIEWS, ""]
        lines = [ReportText.REVIEWS_HEADING, ""]
        for review in self.current_pull_request_reviews.reviews:
            lines.append(
                f"- **{review.author.login}** — {review.state.spoken} "
                f"({review.submitted_at})"
            )
            if review.body.strip():
                lines.append(f"  > {review.body.strip()}")
        lines.append("")
        return lines

    def _render_threads(self) -> list[str]:
        """:return: The review-threads section."""
        shown = self.shown_threads
        lines = [self.heading(len(shown)), ""]
        if not shown:
            lines.extend([ReportText.NOTHING_TO_ACT_ON, ""])
            return lines
        for thread in shown:
            lines.extend(self._render_thread(thread))
        return lines

    def _render_thread(self, thread: ReviewThread) -> list[str]:
        """
        Render one thread with every comment in it.

        :param thread: The thread to render.
        :return: The thread's markdown lines.
        """
        markers = thread.markers
        suffix = f" _({', '.join(markers)})_" if markers else ""
        lines = [f"### `{thread.location}`{suffix}", ""]
        for comment in thread.comments:
            lines.append(f"- **{comment.author.login}**: {comment.body.strip()}")
        if thread.comments:
            lines.extend(["", f"<{thread.comments[0].url}>", ""])
        return lines


# %% command line


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """
    Parse the command line.

    :param argv: The arguments to parse, defaulting to the process's own.
    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--pull-request", type=int, help="the upstream pull request number to read"
    )
    target.add_argument(
        "--branch", help="the fork branch whose upstream pull request to read"
    )
    parser.add_argument(
        "--fork-owner",
        default=os.environ.get(EnvironmentVariable.REPOSITORY_OWNER, ""),
        help="the owner whose branches to claim, defaulting to the runner's own",
    )
    parser.add_argument(
        "--upstream", help="an owner/name upstream outranking the configured one"
    )
    parser.add_argument(
        "--include-resolved",
        action="store_true",
        help="show threads already marked resolved as well",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Read one upstream pull request and print its report.

    A branch that was never promoted upstream is an ordinary answer rather than a crash,
    so this boundary turns the script's own errors into a stated reason. Anything else
    still propagates with its traceback intact.

    :param argv: The arguments to parse, defaulting to the process's own.
    :return: The process exit status.
    """
    arguments = _parse_arguments(argv)
    try:
        report = _build_report(arguments)
    except UpstreamReviewError as failure:
        print(failure, file=sys.stderr)
        return 1
    print(report)
    summary_path = os.environ.get(EnvironmentVariable.STEP_SUMMARY)
    if summary_path:
        Path(summary_path).write_text(report)
    return 0


def _build_report(arguments: argparse.Namespace) -> str:
    """
    Read the requested pull request and render its report.

    :param arguments: The parsed command line.
    :return: The rendered markdown.
    """
    reader = UpstreamReviewReader(
        GitHubCommandLineClient(),
        resolve_upstream_repository(override=arguments.upstream),
        arguments.fork_owner,
    )
    number = arguments.pull_request or reader.resolve_pull_request_number(
        arguments.branch
    )
    return UnresolvedThreadReport(
        reader.read_current_state(number), include_resolved=arguments.include_resolved
    ).render()


if __name__ == "__main__":
    sys.exit(main())
