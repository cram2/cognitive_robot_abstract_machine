"""
Tests for upstream_reviews.py's data parsing, thread pagination, pull request
resolution, report rendering, and the gh-backed client.
"""

import json
import os
import shutil
import stat
from enum import StrEnum
from pathlib import Path

import pytest
from conftest import FixtureName, RecordedCall, ReplayingClient

from upstream_reviews import (
    GitHubCommandFailed,
    GitHubCommandLineClient,
    GraphQLErrorsReturned,
    JSONModel,
    PullRequestJSONKey,
    PullRequestReviewSnapshot,
    QueryVariable,
    ReportText,
    Repository,
    ReviewState,
    ReviewThread,
    ThreadMarker,
    UnresolvedThreadReport,
    UpstreamPullRequestNotFound,
    UpstreamReviewReader,
    main,
    resolve_upstream_repository,
)


class Example(StrEnum):
    """
    The identities the recorded responses were built around.
    """

    UPSTREAM_OWNER = "example-upstream"
    UPSTREAM_NAME = "example-repo"
    FORK_OWNER = "example-fork-owner"
    FOREIGN_OWNER = "another-contributor"
    BRANCH = "some-branch"
    UNPROMOTED_BRANCH = "never-promoted"


class ThreadIdentifier(StrEnum):
    """
    The review threads the recorded responses carry.
    """

    RESOLVED = "THREAD_RESOLVED"
    UNRESOLVED_MIDDLE = "THREAD_UNRESOLVED_MIDDLE"
    UNRESOLVED_OUTDATED = "THREAD_UNRESOLVED_OUTDATED"


class ThreadCursor(StrEnum):
    """
    The cursors the recorded pages hand back.
    """

    PAGE_ONE = "CURSOR_PAGE_ONE"


class StubEnvironmentVariable(StrEnum):
    """
    The knobs the ``gh`` stub reads.
    """

    GRAPHQL_JSON = "STUB_GH_GRAPHQL_JSON"
    EXIT_CODE = "STUB_GH_EXIT_CODE"
    CALL_LOG = "STUB_GH_CALL_LOG"


UPSTREAM = Repository(Example.UPSTREAM_OWNER, Example.UPSTREAM_NAME)
RECORDED_PULL_REQUEST_NUMBER = 513
GRAPHQL_ERROR_MESSAGE = "Could not resolve to a Repository"
UPSTREAM_SETTING_TEMPLATE = 'upstream_repository = "{repository}"\n'


def make_reader(client: ReplayingClient) -> UpstreamReviewReader:
    """
    Build a reader wired to *client* and the recorded upstream.

    :param client: The client to replay responses from.
    :return: The reader under test.
    """
    return UpstreamReviewReader(client, UPSTREAM, Example.FORK_OWNER)


def recorded_thread(fixture: FixtureName, identifier: ThreadIdentifier) -> ReviewThread:
    """
    Read one recorded review thread by its identifier.

    Reaches it through the same model the production code parses into, so a test names
    attributes rather than indexing the raw data.

    :param fixture: The fixture to read.
    :param identifier: The thread to find.
    :return: The recorded thread.
    :raises KeyError: If the fixture carries no such thread.
    """
    for thread in fixture.recorded().review_thread_page.threads:
        if thread.identifier == identifier:
            return thread
    raise KeyError(identifier)


# %% the reading contract


def test_a_model_that_declares_no_reader_cannot_be_built():
    class ModelMissingItsReader(JSONModel):
        """Stands in for a model that forgot the reader every model owes."""

    with pytest.raises(TypeError):
        ModelMissingItsReader()


# %% data parsing


def test_a_thread_maps_each_field_to_its_own_attribute(paginated_client):
    """Pins the mapping itself, with values stated here rather than read back
    through the parser under test - the one assertion that has to restate them."""
    snapshot = make_reader(paginated_client).read_current_state(
        RECORDED_PULL_REQUEST_NUMBER
    )

    parsed = snapshot.thread(ThreadIdentifier.RESOLVED)
    assert parsed.is_resolved is True
    assert parsed.is_outdated is False
    assert parsed.line == 1031
    assert parsed.comments[0].database_identifier == 3728009027


def test_every_recorded_thread_survives_the_read_unchanged(paginated_client):
    snapshot = make_reader(paginated_client).read_current_state(
        RECORDED_PULL_REQUEST_NUMBER
    )

    assert snapshot.thread(ThreadIdentifier.RESOLVED) == recorded_thread(
        FixtureName.PULL_REQUEST_PAGE_ONE, ThreadIdentifier.RESOLVED
    )
    assert snapshot.thread(ThreadIdentifier.UNRESOLVED_OUTDATED) == recorded_thread(
        FixtureName.PULL_REQUEST_PAGE_TWO, ThreadIdentifier.UNRESOLVED_OUTDATED
    )


def test_the_snapshot_identifies_the_pull_request_it_read(paginated_client):
    snapshot = make_reader(paginated_client).read_current_state(
        RECORDED_PULL_REQUEST_NUMBER
    )

    recorded = FixtureName.PULL_REQUEST_PAGE_ONE.recorded().pull_request_reviews
    assert snapshot.number == recorded.number
    assert snapshot.title == recorded.title
    assert snapshot.url == recorded.url


def test_a_thread_keeps_every_comment_in_order(paginated_client):
    snapshot = make_reader(paginated_client).read_current_state(
        RECORDED_PULL_REQUEST_NUMBER
    )

    recorded = recorded_thread(
        FixtureName.PULL_REQUEST_PAGE_ONE, ThreadIdentifier.UNRESOLVED_MIDDLE
    )
    parsed = snapshot.thread(ThreadIdentifier.UNRESOLVED_MIDDLE)
    assert [comment.author for comment in parsed.comments] == [
        comment.author for comment in recorded.comments
    ]


def test_reviews_are_parsed_with_their_state(paginated_client):
    snapshot = make_reader(paginated_client).read_current_state(
        RECORDED_PULL_REQUEST_NUMBER
    )

    assert [review.state for review in snapshot.reviews] == [
        ReviewState.CHANGES_REQUESTED,
        ReviewState.COMMENTED,
    ]


def test_a_thread_on_an_outdated_hunk_has_no_line(paginated_client):
    snapshot = make_reader(paginated_client).read_current_state(
        RECORDED_PULL_REQUEST_NUMBER
    )

    outdated = snapshot.thread(ThreadIdentifier.UNRESOLVED_OUTDATED)
    assert outdated.markers == [ThreadMarker.OUTDATED]
    assert outdated.line is None
    assert outdated.location == outdated.path


def test_a_thread_anchored_to_a_line_is_located_by_it(paginated_client):
    snapshot = make_reader(paginated_client).read_current_state(
        RECORDED_PULL_REQUEST_NUMBER
    )

    anchored = snapshot.thread(ThreadIdentifier.UNRESOLVED_MIDDLE)
    assert anchored.location == f"{anchored.path}:{anchored.line}"


# %% thread pagination


def test_both_pages_are_merged_into_one_snapshot(paginated_client):
    snapshot = make_reader(paginated_client).read_current_state(
        RECORDED_PULL_REQUEST_NUMBER
    )

    assert [thread.identifier for thread in snapshot.threads] == [
        ThreadIdentifier.RESOLVED,
        ThreadIdentifier.UNRESOLVED_MIDDLE,
        ThreadIdentifier.UNRESOLVED_OUTDATED,
    ]


def test_the_second_request_carries_the_first_pages_cursor(paginated_client):
    make_reader(paginated_client).read_current_state(RECORDED_PULL_REQUEST_NUMBER)

    assert len(paginated_client.calls) == 2
    assert paginated_client.calls[0].variables[QueryVariable.THREAD_CURSOR] is None
    assert (
        paginated_client.calls[1].variables[QueryVariable.THREAD_CURSOR]
        == ThreadCursor.PAGE_ONE
    )


def test_paging_stops_once_a_page_reports_no_successor(paginated_client):
    make_reader(paginated_client).read_current_state(RECORDED_PULL_REQUEST_NUMBER)

    assert paginated_client.responses == []


# %% resolved filtering


def test_resolved_threads_are_excluded_by_default(paginated_client):
    snapshot = make_reader(paginated_client).read_current_state(
        RECORDED_PULL_REQUEST_NUMBER
    )

    assert [thread.identifier for thread in snapshot.unresolved_threads] == [
        ThreadIdentifier.UNRESOLVED_MIDDLE,
        ThreadIdentifier.UNRESOLVED_OUTDATED,
    ]


def test_the_report_omits_a_resolved_thread(paginated_client):
    snapshot = make_reader(paginated_client).read_current_state(
        RECORDED_PULL_REQUEST_NUMBER
    )

    report = UnresolvedThreadReport(snapshot)

    assert ThreadIdentifier.RESOLVED not in {
        thread.identifier for thread in report.shown_threads
    }
    assert snapshot.thread(ThreadIdentifier.RESOLVED).comments[0].body not in (
        report.render()
    )


def test_including_resolved_threads_restores_it(paginated_client):
    snapshot = make_reader(paginated_client).read_current_state(
        RECORDED_PULL_REQUEST_NUMBER
    )

    rendered = UnresolvedThreadReport(snapshot, include_resolved=True).render()

    assert snapshot.thread(ThreadIdentifier.RESOLVED).comments[0].body in rendered


def test_including_resolved_threads_counts_what_is_shown(paginated_client):
    snapshot = make_reader(paginated_client).read_current_state(
        RECORDED_PULL_REQUEST_NUMBER
    )

    report = UnresolvedThreadReport(snapshot, include_resolved=True)

    assert report.heading(len(snapshot.threads)) in report.render()
    assert len(report.shown_threads) == len(snapshot.threads)


# %% pull request resolution


def test_a_branch_resolves_to_the_forks_own_pull_request():
    client = ReplayingClient([FixtureName.BRANCH_PULL_REQUESTS.load()])

    number = make_reader(client).resolve_pull_request_number(Example.BRANCH)

    assert number == RECORDED_PULL_REQUEST_NUMBER


def test_the_branch_name_and_upstream_are_sent_as_variables():
    client = ReplayingClient([FixtureName.BRANCH_PULL_REQUESTS.load()])

    make_reader(client).resolve_pull_request_number(Example.BRANCH)

    assert client.calls[0].variables == {
        QueryVariable.OWNER: Example.UPSTREAM_OWNER,
        QueryVariable.NAME: Example.UPSTREAM_NAME,
        QueryVariable.HEAD_REF_NAME: Example.BRANCH,
    }


def test_a_branch_of_another_contributor_is_not_claimed():
    client = ReplayingClient([FixtureName.BRANCH_PULL_REQUESTS_FOREIGN_OWNER.load()])

    with pytest.raises(UpstreamPullRequestNotFound) as raised:
        make_reader(client).resolve_pull_request_number(Example.BRANCH)

    assert raised.value.branch == Example.BRANCH
    assert raised.value.fork_owner == Example.FORK_OWNER


def test_a_branch_never_promoted_upstream_is_reported_clearly():
    client = ReplayingClient([FixtureName.BRANCH_PULL_REQUESTS_NONE.load()])

    with pytest.raises(UpstreamPullRequestNotFound) as raised:
        make_reader(client).resolve_pull_request_number(Example.UNPROMOTED_BRANCH)

    assert raised.value.upstream == UPSTREAM


# %% portability


def test_the_configured_upstream_reaches_the_query():
    client = ReplayingClient([FixtureName.BRANCH_PULL_REQUESTS.load()])
    elsewhere = Repository("another-organization", "another-repository")
    reader = UpstreamReviewReader(client, elsewhere, Example.FOREIGN_OWNER)

    reader.resolve_pull_request_number(Example.BRANCH)

    assert client.calls[0].variables[QueryVariable.OWNER] == elsewhere.owner
    assert client.calls[0].variables[QueryVariable.NAME] == elsewhere.name


def test_the_upstream_is_read_from_the_configuration_file(tmp_path):
    configured = Repository("some-organization", "some-repository")
    configuration = tmp_path / "stack.toml"
    configuration.write_text(UPSTREAM_SETTING_TEMPLATE.format(repository=configured))

    assert resolve_upstream_repository(configuration) == configured


def test_an_explicit_override_outranks_the_configuration_file(tmp_path):
    configured = Repository("some-organization", "some-repository")
    overriding = Repository("override-organization", "override-repository")
    configuration = tmp_path / "stack.toml"
    configuration.write_text(UPSTREAM_SETTING_TEMPLATE.format(repository=configured))

    assert resolve_upstream_repository(configuration, str(overriding)) == overriding


# %% report rendering


@pytest.fixture
def current_state(paginated_client) -> PullRequestReviewSnapshot:
    """:return: The snapshot parsed from both recorded pages."""
    return make_reader(paginated_client).read_current_state(
        RECORDED_PULL_REQUEST_NUMBER
    )


def test_each_unresolved_thread_is_located_by_file_and_line(current_state):
    rendered = UnresolvedThreadReport(current_state).render()

    for thread in current_state.unresolved_threads:
        assert thread.location in rendered


def test_comment_bodies_are_reproduced(current_state):
    rendered = UnresolvedThreadReport(current_state).render()

    for thread in current_state.unresolved_threads:
        for comment in thread.comments:
            assert comment.body in rendered


def test_each_thread_links_back_to_its_first_comment(current_state):
    rendered = UnresolvedThreadReport(current_state).render()

    for thread in current_state.unresolved_threads:
        assert thread.comments[0].url in rendered


def test_an_outdated_thread_is_marked_as_such(current_state):
    rendered = UnresolvedThreadReport(current_state).render()

    assert ThreadMarker.OUTDATED in rendered


def test_the_unresolved_count_is_stated(current_state):
    report = UnresolvedThreadReport(current_state)

    assert report.heading(len(report.shown_threads)) in report.render()


def test_every_reviewer_and_verdict_is_listed(current_state):
    rendered = UnresolvedThreadReport(current_state).render()

    for review in current_state.reviews:
        assert review.author.login in rendered
        assert review.state.spoken in rendered


def test_a_pull_request_with_nothing_outstanding_says_so(current_state):
    settled = PullRequestReviewSnapshot(
        number=current_state.number,
        title=current_state.title,
        url=current_state.url,
        reviews=current_state.reviews,
        threads=[thread for thread in current_state.threads if thread.is_resolved],
    )

    rendered = UnresolvedThreadReport(settled).render()

    assert ReportText.NO_UNRESOLVED_HEADING in rendered
    assert ReportText.NOTHING_TO_ACT_ON in rendered


# %% gh client


@pytest.fixture
def stubbed_gh(tmp_path, monkeypatch) -> Path:
    """
    Put the ``gh`` stub first on ``PATH``.

    :param tmp_path: pytest's per-test temporary directory.
    :param monkeypatch: The fixture used to prepend the stub directory.
    :return: The directory the stub was installed into.
    """
    stub_directory = tmp_path / "bin"
    stub_directory.mkdir()
    installed = stub_directory / "gh"
    shutil.copy(Path(__file__).parent / "stubs" / "gh.sh", installed)
    installed.chmod(installed.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{stub_directory}{os.pathsep}{os.environ['PATH']}")
    return stub_directory


def test_the_data_is_unwrapped(stubbed_gh, monkeypatch):
    data = {PullRequestJSONKey.REPOSITORY: None}
    monkeypatch.setenv(
        StubEnvironmentVariable.GRAPHQL_JSON,
        json.dumps({PullRequestJSONKey.DATA: data}),
    )

    assert GitHubCommandLineClient().execute("query {}", {}) == data


def test_the_query_and_variables_are_sent_as_the_request_body(
    stubbed_gh, monkeypatch, tmp_path
):
    call_log = tmp_path / "calls.txt"
    monkeypatch.setenv(StubEnvironmentVariable.CALL_LOG, str(call_log))
    monkeypatch.setenv(
        StubEnvironmentVariable.GRAPHQL_JSON, json.dumps({PullRequestJSONKey.DATA: {}})
    )
    sent = RecordedCall("query Example {}", {QueryVariable.NUMBER: 513})

    GitHubCommandLineClient().execute(sent.query, sent.variables)

    assert json.loads(call_log.read_text()) == {
        PullRequestJSONKey.QUERY: sent.query,
        PullRequestJSONKey.VARIABLES: sent.variables,
    }


def test_a_failing_gh_invocation_is_raised(stubbed_gh, monkeypatch):
    monkeypatch.setenv(StubEnvironmentVariable.EXIT_CODE, "1")

    with pytest.raises(GitHubCommandFailed) as raised:
        GitHubCommandLineClient().execute("query {}", {})

    assert raised.value.exit_code == 1


def test_graphql_errors_are_raised_rather_than_returned(stubbed_gh, monkeypatch):
    monkeypatch.setenv(
        StubEnvironmentVariable.GRAPHQL_JSON,
        json.dumps(
            {
                PullRequestJSONKey.ERRORS: [
                    {PullRequestJSONKey.MESSAGE: GRAPHQL_ERROR_MESSAGE}
                ]
            }
        ),
    )

    with pytest.raises(GraphQLErrorsReturned) as raised:
        GitHubCommandLineClient().execute("query {}", {})

    assert raised.value.messages == [GRAPHQL_ERROR_MESSAGE]


def test_a_branch_without_an_upstream_pull_request_exits_without_a_traceback(
    stubbed_gh, monkeypatch, capsys
):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setenv(
        StubEnvironmentVariable.GRAPHQL_JSON,
        json.dumps(
            {PullRequestJSONKey.DATA: FixtureName.BRANCH_PULL_REQUESTS_NONE.load()}
        ),
    )

    status = main(
        [
            "--branch",
            str(Example.UNPROMOTED_BRANCH),
            "--fork-owner",
            str(Example.FORK_OWNER),
            "--upstream",
            str(UPSTREAM),
        ]
    )

    assert status == 1
    assert str(Example.UNPROMOTED_BRANCH) in capsys.readouterr().err
