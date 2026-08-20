"""
Tests for the maintenance executor - the half of the pass that moves commits.

``stack.py`` derives and prints; every assertion about it can be made against an
in-memory export. This module executes, so most of its behaviour is only true of a real
repository: whether a push happened, whether a refused push left the destination
untouched, which paths a merge conflicted on. Those run against real git in a scratch
fork built here, with bare repositories standing in for the fork and the upstream, so
nothing touches the network.

The board export and the report are pure, and are tested as such.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclasses_field
from pathlib import Path

import pytest

from scratch_repository import initialize_bare_repository

from stack import (
    BOARD_PATH,
    Configuration,
    IntegrationStrategy,
    PullRequest,
    RefusalReason,
    Repository,
    Stack,
    build_stack,
    load_board,
)

import maintenance_commands
import maintenance_restack_procedure
from class_property import classproperty
from maintenance_board import (
    BoardExport,
    MissingPullRequestFieldError,
    PullRequestField,
    get_session_link_in,
)
from maintenance_commands import (
    COMMANDS,
    BoardCommand,
    MaintenanceCommand,
    MaintenancePass,
    RestackCommand,
    RunReportCommand,
)
from maintenance_constants import CREDENTIAL_VARIABLES, PROMOTION_LINK_LABEL
from maintenance_fast_forward import (
    FastForwardOutcome,
    FastForwardReport,
    fast_forward,
)
from maintenance_git_commands import (
    BranchAncestry,
    GitCommandFailed,
    GitCommandRunner,
    ProposedPush,
)
from maintenance_github import ForkPullRequests
from maintenance_promotion import (
    clear_spent_promotion_labels,
    description_with_promotion_link,
    promote,
)
from maintenance_report import (
    MaintenanceExitCode,
    MaintenanceReport,
    build_report,
    exit_code_for,
)
from maintenance_restack_procedure import restack
from maintenance_restack_steps import BranchOutcome, RestackOutcome, RestackStep

STACK_DIRECTORY = Path(__file__).parent.parent
"""
The directory the executor's modules live in, which is also what a subprocess running
one of them resolves its imports against.
"""

MAINTENANCE_SCRIPT = STACK_DIRECTORY / "maintenance.py"
"""
The executor under test, invoked as a subprocess wherever an exit status is the
assertion.
"""

UPSTREAM_BASE = "main"
"""
The branch every stack in these tests ultimately targets.
"""

A_LABEL_THIS_TOOL_NEVER_WRITES = "a-label-somebody-else-put-here"
"""
Stands for whatever else a pull request happens to carry - the labels a write must
preserve precisely because this tool knows nothing about them.
"""


def make_configuration() -> Configuration:
    """
    :return: The configuration a scratch fork checkout resolves to.
    """
    return Configuration(
        in_review_label="in-review",
        rebase_label="rebase",
        needs_resolution_label="needs-resolution",
        fork_repository=Repository("a-fork-owner", "a-fork"),
        fork_remote="origin",
        upstream_repository=Repository("an-upstream-owner", "a-project"),
        upstream_remote="cram2",
        upstream_base=UPSTREAM_BASE,
        upstream_setup_command=None,
    )


# %% a real fork checkout to execute against


@dataclass
class ForkCheckout:
    """
    A work clone plus the bare repositories standing in for its fork and its upstream.

    Bare repositories live at paths ending ``<owner>/<name>.git`` and are addressed as
    ``file://`` URLs, because a remote is matched by the repository its URL names and a
    plain local path deliberately names none.
    """

    project_root: Path
    """
    The clone the executor runs in.
    """

    fork_path: Path
    """
    The bare repository the fork remote points at.
    """

    upstream_path: Path
    """
    The bare repository the upstream remote points at.
    """

    @classmethod
    def create(cls, parent_directory: Path) -> ForkCheckout:
        """
        Build a checkout with both remotes wired up and ``main`` published to each.

        :param parent_directory: Where to put the clone and the bare repositories.
        :return: The new checkout.
        """
        project_root = parent_directory / "project"
        project_root.mkdir(parents=True)
        checkout = cls(
            project_root,
            cls._bare_repository(parent_directory / "a-fork-owner" / "a-fork.git"),
            cls._bare_repository(
                parent_directory / "an-upstream-owner" / "a-project.git"
            ),
        )
        checkout.run_git("init", "--quiet")
        checkout.run_git("symbolic-ref", "HEAD", f"refs/heads/{UPSTREAM_BASE}")
        checkout.run_git("config", "user.name", "Scratch Fork")
        checkout.run_git("config", "user.email", "scratch-fork@example.com")
        checkout.run_git("remote", "add", "origin", checkout.fork_path.as_uri())
        checkout.run_git("remote", "add", "cram2", checkout.upstream_path.as_uri())
        checkout.commit("a-file", "the first line\n")
        checkout.run_git("push", "--quiet", "origin", UPSTREAM_BASE)
        checkout.run_git("push", "--quiet", "cram2", UPSTREAM_BASE)
        checkout.run_git("fetch", "--quiet", "origin")
        checkout.run_git("fetch", "--quiet", "cram2")
        return checkout

    @staticmethod
    def _bare_repository(path: Path) -> Path:
        """
        :param path: Where to create the bare repository, parents included.
        :return: The same path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        return initialize_bare_repository(path)

    def run_git(self, *arguments: str) -> str:
        """
        Run git in the clone, failing the test if it reports an error.

        :param arguments: The arguments to pass to git.
        :return: The command's stripped stdout.
        """
        result = subprocess.run(
            ["git", *arguments],
            cwd=self.project_root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    def commit(self, name: str, content: str) -> str:
        """
        Write a file and commit it on the checked-out branch.

        :param name: The file to write.
        :param content: What to write into it.
        :return: The new commit's hash.
        """
        (self.project_root / name).write_text(content)
        self.run_git("add", name)
        self.run_git("commit", "--quiet", "-m", f"write {name}")
        return self.run_git("rev-parse", "HEAD")

    def branch_from(self, name: str, start_point: str) -> str:
        """
        Create a branch with a commit of its own, and publish it to the fork.

        The commit is what makes the branch a stack node rather than another name for
        its start point: a branch containing nothing of its own is an ancestor of the
        upstream base, which the derived stack reads - correctly - as already landed.

        :param name: The branch to create.
        :param start_point: What to start it from.
        :return: The branch's published commit hash.
        """
        self.run_git("checkout", "--quiet", "-B", name, start_point)
        commit = self.commit(f"{name}-file", f"the work on {name}\n")
        self.run_git("push", "--quiet", "origin", f"{name}:{name}")
        self.run_git("fetch", "--quiet", "origin")
        return commit

    def commit_on(self, branch: str, name: str, content: str) -> str:
        """
        Add a commit to a branch and publish it to the fork.

        :param branch: The branch to commit on.
        :param name: The file to write.
        :param content: What to write into it.
        :return: The branch's new published commit hash.
        """
        self.run_git("checkout", "--quiet", branch)
        commit = self.commit(name, content)
        self.run_git("push", "--quiet", "origin", f"{branch}:{branch}")
        self.run_git("fetch", "--quiet", "origin")
        return commit

    def published_commit(self, remote: str, branch: str) -> str:
        """
        :param remote: The remote to read from.
        :param branch: The branch to read.
        :return: The commit that branch points at on that remote.
        """
        return self.run_git("rev-parse", f"{remote}/{branch}")

    def commit_on_the_fork(self, branch: str) -> str:
        """
        Read a branch from the fork itself rather than from this clone's view of it.

        :param branch: The branch to read.
        :return: The commit the fork has that branch pointing at.
        """
        return self.run_git("ls-remote", "origin", f"refs/heads/{branch}").split()[0]

    @property
    def git(self) -> GitCommandRunner:
        """
        :return: The runner the executor drives this checkout through.
        """
        return GitCommandRunner(working_directory=self.project_root)


@pytest.fixture
def fork_checkout(tmp_path: Path) -> ForkCheckout:
    """
    A real fork checkout with both remotes wired to local bare repositories.

    :param tmp_path: pytest's per-test temporary directory.
    :return: The checkout.
    """
    return ForkCheckout.create(tmp_path)


def a_stack(checkout: ForkCheckout, pull_requests: list[PullRequest]):
    """
    Build the derived stack the executor consumes, with landedness read from real git.

    :param checkout: The checkout to answer ancestry questions from.
    :param pull_requests: The board entries.
    :return: The derived stack.
    """
    configuration = make_configuration()
    upstream = f"{configuration.upstream_remote}/{configuration.upstream_base}"
    ancestry = BranchAncestry(configuration, checkout.git)

    def is_merged(name: str) -> bool:
        return ancestry.is_ancestor(name, upstream)

    return build_stack(configuration, pull_requests, is_merged)


# %% running git


def test_a_failing_git_command_raises_rather_than_returning_nothing(
    fork_checkout: ForkCheckout,
):
    """
    ``stack.py``'s helper returns an empty string on failure, which here would make a
    push that did nothing indistinguishable from one that worked.
    """
    with pytest.raises(GitCommandFailed) as raised:
        fork_checkout.git.run("rev-parse", "a-ref-that-does-not-exist")

    assert raised.value.arguments == ("rev-parse", "a-ref-that-does-not-exist")
    assert raised.value.status != 0


# %% the board export


def an_api_record(
    number: int = 7,
    head: str = "a-branch",
    base: str = UPSTREAM_BASE,
    draft: bool = False,
    labels: list[str] | None = None,
    body: str = "",
) -> dict:
    """
    :param number: The pull request number.
    :param head: The head branch reference.
    :param base: The base branch reference.
    :param draft: Whether the pull request is a draft.
    :param labels: The label names it carries.
    :param body: The description to read a session link out of.
    :return: One pull request in the shape the REST API returns it.
    """
    return {
        PullRequestField.NUMBER.key: number,
        PullRequestField.HEAD.key: {"ref": head},
        PullRequestField.BASE.key: {"ref": base},
        PullRequestField.DRAFT.key: draft,
        PullRequestField.LABELS.key: [{"name": name} for name in labels or []],
        PullRequestField.BODY.key: body,
    }


def test_every_field_is_named_by_the_key_the_api_answers_under():
    """
    A member's value is the specification's argument tuple; passing a built
    specification instead lands the whole instance in the key, where every read then
    silently finds nothing.
    """
    for field in PullRequestField:
        assert isinstance(field.key, str), field
    assert len({field.key for field in PullRequestField}) == len(list(PullRequestField))


def test_the_export_reads_each_field_out_of_the_shape_the_api_returns_it_in():
    """
    ``head`` and ``base`` arrive nested and ``labels`` as objects, so a field read the
    wrong way produces a board that is wrong rather than empty.
    """
    export = BoardExport.from_api_records(
        [
            an_api_record(
                number=41,
                head="a-child",
                base="a-parent",
                draft=True,
                labels=["rebase"],
                body="see https://claude.ai/code/session_01ABCdef",
            )
        ]
    )

    exported = export.pull_requests[0]
    assert exported.number == 41
    assert exported.head == "a-child"
    assert exported.base == "a-parent"
    assert exported.draft is True
    assert exported.labels == ["rebase"]
    assert exported.session == "https://claude.ai/code/session_01ABCdef"


def test_the_written_board_parses_back_into_the_records_it_was_built_from(
    tmp_path: Path,
):
    """
    The export's only contract is that ``stack.load_board`` reads it, so the expected
    value is derived by reading it rather than by hand-writing the shape twice.
    """
    export = BoardExport.from_api_records(
        [
            an_api_record(number=41, head="a-child", base="a-parent", draft=True),
            an_api_record(
                number=40, head="a-parent", labels=[A_LABEL_THIS_TOOL_NEVER_WRITES]
            ),
        ]
    )
    destination = tmp_path / "board.json"

    export.write(destination)

    assert load_board(destination) == list(export.pull_requests)


def test_a_pull_request_missing_a_required_field_is_rejected_rather_than_defaulted():
    """
    A dropped field is what made #119's bad data indistinguishable from good data, so
    the parser refuses it at the point it enters rather than substituting a default.
    """
    record = an_api_record(number=41)
    del record["draft"]

    with pytest.raises(MissingPullRequestFieldError) as raised:
        BoardExport.from_api_records([record])

    assert raised.value.field_name == PullRequestField.DRAFT
    assert raised.value.pull_request_number == 41


def test_the_board_snapshot_is_never_committable():
    """
    ``board --write`` writes into the working tree, so nothing but ``.gitignore`` stands
    between a pass and a committed snapshot of a stack that has since moved.
    """
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", str(BOARD_PATH)],
        cwd=BOARD_PATH.parent,
        capture_output=True,
    )

    assert ignored.returncode == 0


def test_a_session_link_is_read_out_of_the_description():
    body = "Some prose.\n\nSession: https://claude.ai/code/session_01ABCdef\n"

    assert get_session_link_in(body) == "https://claude.ai/code/session_01ABCdef"


def test_a_description_naming_no_session_yields_none():
    assert get_session_link_in("Some prose with no link.") is None


# %% fast-forward


def test_the_fork_base_is_fast_forwarded_to_the_upstream(fork_checkout: ForkCheckout):
    fork_checkout.run_git("checkout", "--quiet", UPSTREAM_BASE)
    advanced = fork_checkout.commit("another-file", "upstream moved\n")
    fork_checkout.run_git("push", "--quiet", "cram2", UPSTREAM_BASE)
    fork_checkout.run_git(
        "push", "--quiet", "--force", "origin", f"HEAD~1:{UPSTREAM_BASE}"
    )
    fork_checkout.run_git("fetch", "--quiet", "origin")

    report = fast_forward(make_configuration(), fork_checkout.git)

    assert report.outcome == FastForwardOutcome.PUSHED
    assert fork_checkout.published_commit("origin", UPSTREAM_BASE) == advanced


def test_a_fork_base_already_level_with_the_upstream_is_left_alone(
    fork_checkout: ForkCheckout,
):
    report = fast_forward(make_configuration(), fork_checkout.git)

    assert report.outcome == FastForwardOutcome.ALREADY_CURRENT


def test_a_non_fast_forward_is_refused_and_the_fork_base_is_untouched(
    fork_checkout: ForkCheckout,
):
    """
    The doctrine says stop rather than force; this makes it unable to force, and the
    assertion is on the destination ref rather than on the command having failed.
    """
    fork_checkout.run_git("checkout", "--quiet", UPSTREAM_BASE)
    fork_checkout.commit("a-fork-only-file", "only on the fork\n")
    fork_checkout.run_git("push", "--quiet", "origin", UPSTREAM_BASE)
    fork_checkout.run_git("checkout", "--quiet", "-B", "a-divergent-line", "HEAD~1")
    fork_checkout.commit("an-upstream-only-file", "only upstream\n")
    fork_checkout.run_git(
        "push", "--quiet", "--force", "cram2", f"HEAD:{UPSTREAM_BASE}"
    )
    fork_checkout.run_git("fetch", "--quiet", "origin")
    fork_checkout.run_git("fetch", "--quiet", "cram2")
    before = fork_checkout.published_commit("origin", UPSTREAM_BASE)

    report = fast_forward(make_configuration(), fork_checkout.git)

    assert report.outcome == FastForwardOutcome.REFUSED_NOT_FAST_FORWARD
    assert fork_checkout.published_commit("origin", UPSTREAM_BASE) == before


# %% restack


def a_parent_and_child(fork_checkout: ForkCheckout) -> None:
    """
    Publish a two-branch stack: ``a-parent`` on the base, ``a-child`` on the parent.

    :param fork_checkout: The checkout to build the branches in.
    """
    fork_checkout.branch_from("a-parent", UPSTREAM_BASE)
    fork_checkout.branch_from("a-child", "a-parent")


def an_unrelated_history_on(fork_checkout: ForkCheckout, branch: str) -> None:
    """
    Republish a branch on a root commit of its own, sharing no history with the stack.

    A merge refuses this outright, so it is a way to stop an integration before it
    begins that depends on nothing but what the fork carries.

    :param fork_checkout: The checkout to republish the branch from.
    :param branch: The branch to republish.
    """
    fork_checkout.run_git("checkout", "--quiet", "--orphan", "an-unrelated-root")
    fork_checkout.run_git("rm", "--quiet", "-rf", ".")
    fork_checkout.commit("an-unrelated-file", "a history of its own\n")
    fork_checkout.run_git("branch", "--force", branch, "HEAD")
    fork_checkout.run_git("push", "--quiet", "--force", "origin", f"{branch}:{branch}")
    fork_checkout.run_git("fetch", "--quiet", "origin")


def the_board(labels: list[str] | None = None) -> list[PullRequest]:
    """
    :param labels: The labels the child's pull request carries.
    :return: The two-branch board matching :func:`a_parent_and_child`.
    """
    return [
        PullRequest(number=40, head="a-parent", base=UPSTREAM_BASE, draft=False),
        PullRequest(
            number=41, head="a-child", base="a-parent", draft=False, labels=labels or []
        ),
    ]


def test_a_branch_whose_parent_has_not_moved_is_reported_up_to_date(
    fork_checkout: ForkCheckout,
):
    a_parent_and_child(fork_checkout)

    outcomes = restack(
        a_stack(fork_checkout, the_board()), fork_checkout.git, RecordingPullRequests()
    )

    assert [outcome.outcome for outcome in outcomes] == [
        RestackOutcome.UP_TO_DATE,
        RestackOutcome.UP_TO_DATE,
    ]


def test_a_branch_whose_parent_moved_is_integrated_and_pushed(
    fork_checkout: ForkCheckout,
):
    a_parent_and_child(fork_checkout)
    fork_checkout.commit_on("a-parent", "a-parent-file", "the parent moved\n")
    before = fork_checkout.published_commit("origin", "a-child")

    outcomes = restack(
        a_stack(fork_checkout, the_board()), fork_checkout.git, RecordingPullRequests()
    )

    child = next(outcome for outcome in outcomes if outcome.branch == "a-child")
    assert child.outcome == RestackOutcome.PUSHED
    after = fork_checkout.published_commit("origin", "a-child")
    assert after != before
    assert child.pushed_commit == after


def test_a_conflicting_integration_pushes_nothing_and_names_the_files(
    fork_checkout: ForkCheckout,
):
    a_parent_and_child(fork_checkout)
    fork_checkout.commit_on("a-parent", "a-contested-file", "the parent's version\n")
    fork_checkout.commit_on("a-child", "a-contested-file", "the child's version\n")
    before = fork_checkout.published_commit("origin", "a-child")

    outcomes = restack(
        a_stack(fork_checkout, the_board()), fork_checkout.git, RecordingPullRequests()
    )

    child = next(outcome for outcome in outcomes if outcome.branch == "a-child")
    assert child.outcome == RestackOutcome.CONFLICT
    assert child.conflicting_paths == ("a-contested-file",)
    assert fork_checkout.published_commit("origin", "a-child") == before


def test_an_integration_stopped_before_it_began_is_not_reported_as_a_conflict(
    fork_checkout: ForkCheckout,
):
    """
    A merge exits non-zero for reasons that leave nothing conflicted at all - here a
    parent whose history shares no root with the branch. Reading any failure as a
    conflict labels a branch that merges perfectly well and sends its owner a report
    naming no files.
    """
    a_parent_and_child(fork_checkout)
    an_unrelated_history_on(fork_checkout, "a-parent")
    before = fork_checkout.published_commit("origin", "a-child")
    fork = RecordingPullRequests()

    outcomes = restack(a_stack(fork_checkout, the_board()), fork_checkout.git, fork)

    child = next(outcome for outcome in outcomes if outcome.branch == "a-child")
    assert child.outcome == RestackOutcome.INTEGRATION_FAILED
    assert child.conflicting_paths == ()
    assert "refusing to merge unrelated histories" in child.explanation
    assert fork.label_writes == []
    assert fork.comments == []
    assert child.reported_at is None
    assert fork_checkout.published_commit("origin", "a-child") == before


def test_a_rebase_labelled_branch_is_rebased_rather_than_merged(
    fork_checkout: ForkCheckout,
):
    """
    The strategy is the only thing that authorises a force-push, so it has to come from
    the label rather than from the executor's own judgement.
    """
    a_parent_and_child(fork_checkout)
    fork_checkout.commit_on("a-parent", "a-parent-file", "the parent moved\n")

    outcomes = restack(
        a_stack(fork_checkout, the_board(labels=["rebase"])),
        fork_checkout.git,
        RecordingPullRequests(),
    )

    child = next(outcome for outcome in outcomes if outcome.branch == "a-child")
    assert child.strategy == IntegrationStrategy.REBASE
    assert child.outcome == RestackOutcome.PUSHED
    merges = fork_checkout.run_git(
        "rev-list", "--merges", "--count", f"origin/{UPSTREAM_BASE}..origin/a-child"
    )
    assert merges == "0"


def test_only_the_rebase_strategy_authorises_rewriting_published_history():
    """
    Forcing is decided in exactly one place, so that is where it is pinned - a test that
    a push happened cannot tell a fast-forward from an overwrite.
    """
    configuration = make_configuration()

    merging = ProposedPush.publishing(
        configuration, "a-branch", IntegrationStrategy.MERGE
    )
    rebasing = ProposedPush.publishing(
        configuration, "a-branch", IntegrationStrategy.REBASE
    )

    assert not merging.with_lease
    assert rebasing.with_lease


def test_a_branch_that_moved_under_the_pass_is_incorporated_rather_than_overwritten(
    fork_checkout: ForkCheckout,
):
    """
    The integration starts from the branch's published tip, not from whatever this
    checkout last saw, so work pushed by somebody else survives the restack.
    """
    a_parent_and_child(fork_checkout)
    fork_checkout.commit_on("a-parent", "a-parent-file", "the parent moved\n")
    fork_checkout.run_git("checkout", "--quiet", "-B", "a-side-line", "origin/a-child")
    somebody_else_s = fork_checkout.commit("a-file-somebody-else-pushed", "not ours\n")
    fork_checkout.run_git("push", "--quiet", "origin", "a-side-line:a-child")
    fork_checkout.run_git("fetch", "--quiet", "origin")

    outcomes = restack(
        a_stack(fork_checkout, the_board()), fork_checkout.git, RecordingPullRequests()
    )

    child = next(outcome for outcome in outcomes if outcome.branch == "a-child")
    assert child.outcome == RestackOutcome.PUSHED
    assert fork_checkout.git.contains(somebody_else_s, "origin/a-child")


def test_a_rebase_whose_lease_has_expired_is_rejected_rather_than_forced_through(
    fork_checkout: ForkCheckout,
):
    """
    The lease is what stops a rebase overwriting a push this pass never saw.

    Staleness is arranged by winding the remote-tracking ref back, which is the state a
    concurrent push leaves this checkout in.
    """
    a_parent_and_child(fork_checkout)
    fork_checkout.commit_on("a-parent", "a-parent-file", "the parent moved\n")
    stale = fork_checkout.published_commit("origin", "a-child")
    somebody_else_s = fork_checkout.commit_on(
        "a-child", "a-file-somebody-else-pushed", "not ours\n"
    )
    fork_checkout.run_git("update-ref", "refs/remotes/origin/a-child", stale)

    outcomes = restack(
        a_stack(fork_checkout, the_board(labels=["rebase"])),
        fork_checkout.git,
        RecordingPullRequests(),
    )

    child = next(outcome for outcome in outcomes if outcome.branch == "a-child")
    assert child.outcome == RestackOutcome.PUSH_REJECTED
    assert fork_checkout.commit_on_the_fork("a-child") == somebody_else_s


def test_a_push_the_move_checks_refuse_is_not_made(fork_checkout: ForkCheckout):
    """
    A parent that has swallowed its own child would, once pushed, make the child an
    ancestor of its own base - which GitHub reads as the child having merged and closes
    its pull request. The refusal is asserted as its reason rather than as the sentence
    explaining it.
    """
    a_parent_and_child(fork_checkout)
    fork_checkout.run_git("checkout", "--quiet", "a-parent")
    fork_checkout.run_git("merge", "--quiet", "--no-edit", "a-child")
    fork_checkout.run_git("push", "--quiet", "origin", "a-parent:a-parent")
    fork_checkout.commit_on(UPSTREAM_BASE, "a-base-file", "the base moved\n")
    before = fork_checkout.published_commit("origin", "a-parent")

    outcomes = restack(
        a_stack(fork_checkout, the_board()), fork_checkout.git, RecordingPullRequests()
    )

    parent = next(outcome for outcome in outcomes if outcome.branch == "a-parent")
    assert parent.outcome == RestackOutcome.REFUSED
    assert RefusalReason.FALSE_MERGE in parent.refusals
    assert fork_checkout.published_commit("origin", "a-parent") == before


# %% the checkout the pass was invoked in

TOOLING_PATH = ".claude/stack/maintenance.py"
"""
Where the pass's own tooling sits: tracked content, so a branch cut before it landed
does not carry it, and checking that branch out deletes it from the working tree.
"""

TOOLING_CONTENT = "the pass's own tooling\n"
"""
What the stand-in tooling file holds, so its survival can be asserted by value.
"""


def a_stack_cut_before_the_tooling_landed(fork_checkout: ForkCheckout) -> Path:
    """
    Publish the two-branch stack, add the tooling to the base alone, and move the parent
    so the child has something to restack.

    Branching first is what most of a real stack looks like - the branches were cut
    before the tooling landed, so their trees do not carry it - and it leaves the
    checkout on the base, holding a file neither branch has.

    :param fork_checkout: The checkout to build the stack in.
    :return: The tooling file in the working tree.
    """
    a_parent_and_child(fork_checkout)
    fork_checkout.run_git("checkout", "--quiet", UPSTREAM_BASE)
    tooling = fork_checkout.project_root / TOOLING_PATH
    tooling.parent.mkdir(parents=True)
    fork_checkout.commit(TOOLING_PATH, TOOLING_CONTENT)
    fork_checkout.commit_on("a-parent", "a-parent-file", "the parent moved\n")
    fork_checkout.run_git("checkout", "--quiet", UPSTREAM_BASE)
    return tooling


def test_a_restack_leaves_the_invoking_checkout_on_its_own_branch(
    fork_checkout: ForkCheckout,
):
    a_stack_cut_before_the_tooling_landed(fork_checkout)

    restack(
        a_stack(fork_checkout, the_board()), fork_checkout.git, RecordingPullRequests()
    )

    assert fork_checkout.run_git("branch", "--show-current") == UPSTREAM_BASE


def test_a_restack_keeps_what_the_branches_it_switches_to_do_not_have(
    fork_checkout: ForkCheckout,
):
    """
    The tooling every step of the pass shells out to is tracked content in the checkout
    the pass runs in, so a restack that switched branches there would delete it.
    """
    tooling = a_stack_cut_before_the_tooling_landed(fork_checkout)

    restack(
        a_stack(fork_checkout, the_board()), fork_checkout.git, RecordingPullRequests()
    )

    assert tooling.read_text() == TOOLING_CONTENT


def test_a_restack_publishes_a_branch_the_caller_is_sitting_on(
    fork_checkout: ForkCheckout,
):
    """
    git refuses to check one branch out in two worktrees at once, and the caller of a
    pass is normally sitting on a branch of the stack - so the restack has to be lent it
    rather than blocked by it.
    """
    a_parent_and_child(fork_checkout)
    fork_checkout.commit_on("a-parent", "a-parent-file", "the parent moved\n")
    fork_checkout.run_git("checkout", "--quiet", "a-child")
    before = fork_checkout.published_commit("origin", "a-child")

    outcomes = restack(
        a_stack(fork_checkout, the_board()), fork_checkout.git, RecordingPullRequests()
    )

    child = next(outcome for outcome in outcomes if outcome.branch == "a-child")
    assert child.outcome == RestackOutcome.PUSHED
    assert fork_checkout.published_commit("origin", "a-child") != before


def test_the_caller_holds_no_branch_while_a_restack_runs(fork_checkout: ForkCheckout):
    """
    Lending the branch is what makes the restack possible at all, so the caller has to
    be off it for as long as the restack has it - not merely put back afterwards.
    """
    a_parent_and_child(fork_checkout)
    fork_checkout.commit_on("a-parent", "a-contested-file", "the parent's version\n")
    fork_checkout.commit_on("a-child", "a-contested-file", "the child's version\n")
    fork = PullRequestsWatchingTheCaller(caller=fork_checkout.git)

    restack(a_stack(fork_checkout, the_board()), fork_checkout.git, fork)

    assert fork.branches_held_while_restacking == [""]


def test_a_restack_gives_back_the_branch_the_caller_lent_it(
    fork_checkout: ForkCheckout,
):
    a_parent_and_child(fork_checkout)
    fork_checkout.commit_on("a-parent", "a-parent-file", "the parent moved\n")
    fork_checkout.run_git("checkout", "--quiet", "a-child")

    restack(
        a_stack(fork_checkout, the_board()), fork_checkout.git, RecordingPullRequests()
    )

    assert fork_checkout.run_git("branch", "--show-current") == "a-child"


def test_a_restack_leaves_no_worktree_of_its_own_behind(fork_checkout: ForkCheckout):
    a_stack_cut_before_the_tooling_landed(fork_checkout)

    restack(
        a_stack(fork_checkout, the_board()), fork_checkout.git, RecordingPullRequests()
    )

    listed = fork_checkout.run_git("worktree", "list", "--porcelain")
    assert [
        line.removeprefix("worktree ")
        for line in listed.splitlines()
        if line.startswith("worktree ")
    ] == [str(fork_checkout.project_root.resolve())]


def test_a_restack_that_raises_still_takes_its_worktree_with_it(
    fork_checkout: ForkCheckout,
):
    """
    A pass abandoned part-way must not leave a worktree behind holding a checked-out
    branch, so the cleanup cannot sit on the success path.
    """
    a_parent_and_child(fork_checkout)
    fork_checkout.commit_on("a-parent", "a-contested-file", "the parent's version\n")
    fork_checkout.commit_on("a-child", "a-contested-file", "the child's version\n")

    with pytest.raises(ReportingRefused):
        restack(
            a_stack(fork_checkout, the_board()),
            fork_checkout.git,
            PullRequestsRefusingToReport(),
        )

    listed = fork_checkout.run_git("worktree", "list", "--porcelain")
    assert [
        line.removeprefix("worktree ")
        for line in listed.splitlines()
        if line.startswith("worktree ")
    ] == [str(fork_checkout.project_root.resolve())]


# %% reporting a branch back to its owner


@dataclass(frozen=True)
class RecordedLabelWrite:
    """
    One label set written to a pull request.
    """

    pull_request_number: int
    """
    The pull request written to.
    """

    labels: tuple[str, ...]
    """
    The complete set it was left carrying.
    """


@dataclass(frozen=True)
class RecordedComment:
    """
    One comment posted on a pull request.
    """

    pull_request_number: int
    """
    The pull request commented on.
    """

    body: str
    """
    What the comment said.
    """


@dataclass(frozen=True)
class RecordedDescription:
    """
    One description written to a pull request.
    """

    pull_request_number: int
    """
    The pull request written to.
    """

    body: str
    """
    The description it was left carrying.
    """


@dataclass(frozen=True)
class RecordingPullRequests(ForkPullRequests):
    """
    Stands in for the fork, recording every write instead of making it.

    The three writes it records were each probed against the live API before this existed
    - a label replace, an issue comment and a body-only description write all succeed on
    the credential a session carries - so what is faked here is the network, not the
    permission.
    """

    states: dict[int, str] = dataclasses_field(default_factory=dict)
    """
    What ``mergeable_state`` to report per pull request number.
    """

    descriptions: dict[int, str] = dataclasses_field(default_factory=dict)
    """
    What description to report per pull request number.
    """

    titles: dict[int, str] = dataclasses_field(default_factory=dict)
    """
    What title to report per pull request number.
    """

    labels: dict[int, list[str]] = dataclasses_field(default_factory=dict)
    """
    What labels to report per pull request number, which is what the branch carries
    *now* rather than what the board snapshot opened the pass with.
    """

    label_writes: list[RecordedLabelWrite] = dataclasses_field(default_factory=list)
    """
    Every label set written, in order.
    """

    comments: list[RecordedComment] = dataclasses_field(default_factory=list)
    """
    Every comment posted, in order.
    """

    description_writes: list[RecordedDescription] = dataclasses_field(
        default_factory=list
    )
    """
    Every description written, in order.
    """

    def open_pull_requests(self) -> list[dict]:
        """
        :return: Every pull request this stand-in has been given state for, in number
            order - the same records reading one of them by number answers with.
        """
        known = {*self.states, *self.descriptions, *self.titles, *self.labels}
        return [self.pull_request(number) for number in sorted(known)]

    def pull_request(self, number: int) -> dict:
        """
        :param number: The pull request to read.
        :return: The fields the executor reads off one pull request, keyed the way the
            executor itself names them.
        """
        return {
            PullRequestField.NUMBER.key: number,
            PullRequestField.MERGEABLE_STATE.key: self.states.get(number, "clean"),
            PullRequestField.BODY.key: self.descriptions.get(number, ""),
            PullRequestField.TITLE.key: self.titles.get(
                number, f"Pull request {number}"
            ),
            PullRequestField.LABELS.key: list(self.labels.get(number, [])),
        }

    def replace_labels(self, number: int, labels: Sequence[str]) -> None:
        """
        :param number: The pull request to write.
        :param labels: The complete label set to write.
        """
        self.label_writes.append(RecordedLabelWrite(number, tuple(labels)))

    def add_comment(self, number: int, body: str) -> str:
        """
        :param number: The pull request to comment on.
        :param body: The comment.
        :return: A stand-in for the comment's URL.
        """
        self.comments.append(RecordedComment(number, body))
        return f"https://example.invalid/comment/{len(self.comments)}"

    def set_description(self, number: int, body: str) -> None:
        """
        :param number: The pull request to write.
        :param body: The new description.
        """
        self.description_writes.append(RecordedDescription(number, body))
        self.descriptions[number] = body


@dataclass(frozen=True)
class PullRequestsWatchingTheCaller(RecordingPullRequests):
    """
    Reads the invoking checkout at the one moment a pass is provably mid-restack.

    A write to the fork only happens from inside a branch's restack, so it is the hook
    for asserting what the invoking checkout looks like while the restack has its
    branch - which is otherwise over before a test can look.
    """

    caller: GitCommandRunner = dataclasses_field(kw_only=True)
    """
    The invoking checkout to read.
    """

    branches_held_while_restacking: list[str] = dataclasses_field(default_factory=list)
    """
    What it had checked out at each write, in order, empty if no write ever came.
    """

    def replace_labels(self, number: int, labels: Sequence[str]) -> None:
        """
        :param number: The pull request to write.
        :param labels: The complete label set to write.
        """
        self.branches_held_while_restacking.append(self.caller.checked_out_branch())
        super().replace_labels(number, labels)


@dataclass
class ReportingRefused(RuntimeError):
    """
    Raised instead of writing to the fork, to abandon a pass where it stands.
    """

    def __str__(self) -> str:
        """
        :return: Why the write did not happen.
        """
        return "the fork refused the write"


@dataclass(frozen=True)
class PullRequestsRefusingToReport(RecordingPullRequests):
    """
    A fork whose label writes raise, so a pass dies part-way through a branch.
    """

    def replace_labels(self, number: int, labels: Sequence[str]) -> None:
        """
        :param number: The pull request that would have been written.
        :param labels: The label set that would have been written.
        :raises ReportingRefused: Always.
        """
        raise ReportingRefused


def test_a_conflict_labels_the_branch_and_tells_its_owner(
    fork_checkout: ForkCheckout,
):
    """
    A conflict this pass cannot resolve is somebody else's to fix, and a report that
    only reaches a run summary reaches nobody.
    """
    a_parent_and_child(fork_checkout)
    fork_checkout.commit_on("a-parent", "a-contested-file", "the parent's version\n")
    fork_checkout.commit_on("a-child", "a-contested-file", "the child's version\n")
    board = the_board()
    board[1].session = "https://claude.ai/code/session_01ABCdef"
    fork = RecordingPullRequests()

    outcomes = restack(a_stack(fork_checkout, board), fork_checkout.git, fork)

    child = next(outcome for outcome in outcomes if outcome.branch == "a-child")
    assert child.outcome == RestackOutcome.CONFLICT
    assert fork.label_writes == [
        RecordedLabelWrite(41, (make_configuration().needs_resolution_label,))
    ]
    comment = fork.comments[0]
    assert comment.pull_request_number == 41
    assert "a-contested-file" in comment.body
    assert "https://claude.ai/code/session_01ABCdef" in comment.body
    assert child.reported_at == "https://example.invalid/comment/1"


def test_a_label_write_keeps_every_label_the_branch_already_carried(
    fork_checkout: ForkCheckout,
):
    """
    The write replaces the whole set, so a label this pass knows nothing about has to be
    sent back with it.
    """
    a_parent_and_child(fork_checkout)
    fork_checkout.commit_on("a-parent", "a-contested-file", "the parent's version\n")
    fork_checkout.commit_on("a-child", "a-contested-file", "the child's version\n")
    fork = RecordingPullRequests()

    restack(
        a_stack(fork_checkout, the_board(labels=[A_LABEL_THIS_TOOL_NEVER_WRITES])),
        fork_checkout.git,
        fork,
    )

    assert fork.label_writes == [
        RecordedLabelWrite(
            41,
            (
                A_LABEL_THIS_TOOL_NEVER_WRITES,
                make_configuration().needs_resolution_label,
            ),
        )
    ]


def test_a_branch_still_conflicting_is_withheld_without_being_relabelled(
    fork_checkout: ForkCheckout,
):
    """
    Re-reporting the same conflict every run is what the label exists to prevent, so a
    branch already carrying it is left entirely alone while it is still dirty.
    """
    a_parent_and_child(fork_checkout)
    fork_checkout.commit_on("a-parent", "a-parent-file", "the parent moved\n")
    fork = RecordingPullRequests(states={41: "dirty"})

    outcomes = restack(
        a_stack(
            fork_checkout,
            the_board(labels=[make_configuration().needs_resolution_label]),
        ),
        fork_checkout.git,
        fork,
    )

    child = next(outcome for outcome in outcomes if outcome.branch == "a-child")
    assert child.outcome == RestackOutcome.WITHHELD
    assert fork.label_writes == []
    assert fork.comments == []


def test_a_branch_that_no_longer_conflicts_has_its_label_cleared_and_is_restacked(
    fork_checkout: ForkCheckout,
):
    """
    GitHub reports ``dirty`` only while there are conflicts, so anything else means the
    owner has resolved it and the branch rejoins the pass.
    """
    a_parent_and_child(fork_checkout)
    fork_checkout.commit_on("a-parent", "a-parent-file", "the parent moved\n")
    fork = RecordingPullRequests(states={41: "clean"})

    outcomes = restack(
        a_stack(
            fork_checkout,
            the_board(labels=[make_configuration().needs_resolution_label]),
        ),
        fork_checkout.git,
        fork,
    )

    child = next(outcome for outcome in outcomes if outcome.branch == "a-child")
    assert child.outcome == RestackOutcome.PUSHED
    assert fork.label_writes == [RecordedLabelWrite(41, ())]


# %% promotion


def test_the_promotion_link_goes_into_the_description_and_the_branch_is_labelled(
    fork_checkout: ForkCheckout,
):
    a_parent_and_child(fork_checkout)
    fork = RecordingPullRequests(
        descriptions={40: "What this branch does.\n\nMore detail.\n"},
        titles={40: "A parent branch"},
    )

    promoted = promote(a_stack(fork_checkout, the_board()), fork)

    assert [entry.branch for entry in promoted] == ["a-parent"]
    written = fork.description_writes[0]
    assert written.pull_request_number == 40
    assert "## Promote" in written.body
    assert promoted[0].url in written.body
    assert "What this branch does." in written.body
    assert fork.label_writes == [RecordedLabelWrite(40, (PROMOTION_LINK_LABEL,))]


def test_a_branch_already_carrying_the_link_label_is_not_promoted_again(
    fork_checkout: ForkCheckout,
):
    a_parent_and_child(fork_checkout)
    fork = RecordingPullRequests(labels={40: [PROMOTION_LINK_LABEL]})

    promoted = promote(a_stack(fork_checkout, the_board()), fork)

    assert promoted == []
    assert fork.description_writes == []


def test_a_branch_labelled_needs_resolution_during_this_pass_is_not_promoted(
    fork_checkout: ForkCheckout,
):
    """
    The board is a snapshot taken before the restack runs, so a branch the restack has
    just withheld still looks promotable in it.

    Promotion has to ask the branch, not the snapshot, or a pass promotes what the same
    pass just conflicted on.
    """
    a_parent_and_child(fork_checkout)
    fork = RecordingPullRequests(
        labels={40: [make_configuration().needs_resolution_label]}
    )

    promoted = promote(a_stack(fork_checkout, the_board()), fork)

    assert promoted == []
    assert fork.description_writes == []
    assert fork.label_writes == []


def test_the_promotion_label_write_keeps_a_label_added_since_the_board_was_taken(
    fork_checkout: ForkCheckout,
):
    """
    A label write replaces the whole set, so computing it from the snapshot silently
    strips anything applied since - including the label the restack applies mid-pass.
    """
    a_parent_and_child(fork_checkout)
    fork = RecordingPullRequests(labels={40: [A_LABEL_THIS_TOOL_NEVER_WRITES]})

    promote(a_stack(fork_checkout, the_board()), fork)

    assert fork.label_writes == [
        RecordedLabelWrite(40, (A_LABEL_THIS_TOOL_NEVER_WRITES, PROMOTION_LINK_LABEL))
    ]


def test_a_second_promotion_replaces_the_link_rather_than_appending_another():
    """
    The description is rewritten on every run that rebuilds a link, so the section has to
    be replaced in place - two Promote headings would leave a reader guessing which link
    is current.
    """
    first = description_with_promotion_link("Prose.\n", "https://example.invalid/first")

    second = description_with_promotion_link(first, "https://example.invalid/second")

    assert second.count("## Promote") == 1
    assert "https://example.invalid/second" in second
    assert "https://example.invalid/first" not in second
    assert second.startswith("Prose.")


def test_a_promoted_branch_that_reached_review_has_its_link_label_removed(
    fork_checkout: ForkCheckout,
):
    """
    The label exists to stop a link being rebuilt; once the branch is in review the link
    has been acted on, so leaving it would misreport the branch forever.
    """
    a_parent_and_child(fork_checkout)
    board = the_board()
    board[0].labels = [PROMOTION_LINK_LABEL, make_configuration().in_review_label]
    fork = RecordingPullRequests()

    cleared = clear_spent_promotion_labels(a_stack(fork_checkout, board), fork)

    assert cleared == ("a-parent",)
    assert fork.label_writes == [
        RecordedLabelWrite(40, (make_configuration().in_review_label,))
    ]


def test_a_branch_no_step_concludes_is_an_error_rather_than_a_silent_pass(
    fork_checkout: ForkCheckout, monkeypatch: pytest.MonkeyPatch
):
    """
    The last step always concludes a branch, so a branch reaching the end of the steps
    means the procedure lost it - which must not read as a branch nothing happened to.
    """
    a_parent_and_child(fork_checkout)
    monkeypatch.setattr(maintenance_restack_procedure, "RESTACK_STEPS", ())

    with pytest.raises(
        maintenance_restack_procedure.RestackConcludedNothingError
    ) as raised:
        restack(
            a_stack(fork_checkout, the_board()),
            fork_checkout.git,
            RecordingPullRequests(),
        )

    assert raised.value.branch in {"a-parent", "a-child"}


# %% the report


def test_the_report_serialises_every_command_s_outcome(fork_checkout: ForkCheckout):
    a_parent_and_child(fork_checkout)
    fork_checkout.commit_on("a-parent", "a-parent-file", "the parent moved\n")
    stack = a_stack(fork_checkout, the_board())

    report = build_report(
        stack,
        fast_forward(make_configuration(), fork_checkout.git),
        restack(stack, fork_checkout.git, RecordingPullRequests()),
    )
    document = json.loads(report.as_json())

    assert document["fast_forward"]["outcome"] == FastForwardOutcome.ALREADY_CURRENT
    assert {entry["branch"] for entry in document["restacked"]} == {
        "a-parent",
        "a-child",
    }
    assert document["promotable"] == ["a-parent"]
    assert document["landed"] == []
    assert document["reparents"] == []


def test_a_whole_pass_leaves_no_board_behind(
    fork_checkout: ForkCheckout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    A board is a snapshot of one moment's open pull requests, and a later run reading a
    stale one is worse than one finding none - so a pass that has finished with it
    removes it, and the next pass starts by exporting a fresh one.
    """
    a_parent_and_child(fork_checkout)
    board_path = tmp_path / "board.json"
    board_path.write_text("{}")
    monkeypatch.setattr(maintenance_commands, "BOARD_PATH", board_path)

    RunReportCommand().run(
        AlreadyResolvedPass.over(fork_checkout, the_board()),
        argparse.Namespace(json=True),
    )

    assert not board_path.exists()


# %% the exit status every command derives from what it left behind


def a_report(
    fast_forward_outcome: FastForwardOutcome = FastForwardOutcome.ALREADY_CURRENT,
    restack_outcome: RestackOutcome = RestackOutcome.PUSHED,
) -> MaintenanceReport:
    """
    :param fast_forward_outcome: What became of the fork's base branch.
    :param restack_outcome: What became of the one branch in the pass.
    :return: A report carrying exactly those two outcomes.
    """
    return MaintenanceReport(
        fast_forward=FastForwardReport(
            fast_forward_outcome, "cram2/main", "origin/main", "a-commit"
        ),
        restacked=(
            BranchOutcome(
                "a-branch", "main", IntegrationStrategy.MERGE, restack_outcome
            ),
        ),
        promoted=(),
        promotion_labels_cleared=(),
        reparents=(),
        landed=(),
        promotable=(),
    )


def test_a_pass_that_published_everything_is_a_success():
    assert exit_code_for(a_report()) == MaintenanceExitCode.SUCCESS
    assert (
        exit_code_for(a_report(restack_outcome=RestackOutcome.UP_TO_DATE))
        == MaintenanceExitCode.SUCCESS
    )


def test_a_refused_fast_forward_is_never_reported_as_a_clean_pass():
    """
    Observed exiting zero against the live fork: the fork's base was left behind the
    upstream and the run said nothing was wrong, which is the one thing a status is for.
    """
    refused = a_report(fast_forward_outcome=FastForwardOutcome.REFUSED_NOT_FAST_FORWARD)

    assert exit_code_for(refused) == MaintenanceExitCode.NOT_FAST_FORWARD


@pytest.mark.parametrize(
    "left_behind",
    [
        RestackOutcome.CONFLICT,
        RestackOutcome.WITHHELD,
        RestackOutcome.PUSH_REJECTED,
        RestackOutcome.INTEGRATION_FAILED,
    ],
)
def test_a_branch_left_unpublished_is_never_reported_as_a_clean_pass(
    left_behind: RestackOutcome,
):
    """
    Also observed against the live fork: a conflict exited zero, so a caller acting on
    the status alone would have read the pass as having nothing outstanding.
    """
    assert (
        exit_code_for(a_report(restack_outcome=left_behind))
        == MaintenanceExitCode.BRANCH_NEEDS_ATTENTION
    )


def test_every_status_names_itself_distinctly():
    """
    The name is derived from the member rather than written beside it, so a status
    cannot end up with a name that belongs to a different one.
    """
    named = {code: code.name_for_a_caller for code in MaintenanceExitCode}

    assert named[MaintenanceExitCode.BRANCH_NEEDS_ATTENTION] == "branch-needs-attention"
    assert named[MaintenanceExitCode.SUCCESS] == "success"
    assert len(set(named.values())) == len(MaintenanceExitCode)


def test_the_report_carries_what_its_status_means(fork_checkout: ForkCheckout):
    """
    A scheduled run reads this document rather than the process status, so the meaning
    has to be in it - mapping an integer back to a meaning is a step it should not have.
    """
    document = json.loads(a_report(restack_outcome=RestackOutcome.CONFLICT).as_json())

    assert document["status"] == "branch-needs-attention"
    assert document["exit_code"] == MaintenanceExitCode.BRANCH_NEEDS_ATTENTION


def test_a_non_zero_status_says_what_it_means_on_the_way_out(
    fork_checkout: ForkCheckout,
):
    """
    A caller reading a bare number has to look it up; the executor already knows.
    """
    fork_checkout.run_git("remote", "remove", "cram2")

    result = run_maintenance(fork_checkout, RestackCommand())

    assert result.returncode == MaintenanceExitCode.BOARD_UNAVAILABLE
    assert "board-unavailable" in result.stderr


def test_a_clean_run_says_nothing_about_its_status(fork_checkout: ForkCheckout):
    """
    Success is the absence of news; announcing it would make every run noisy.
    """
    result = run_maintenance(fork_checkout, BoardCommand(), "--help")

    assert result.returncode == MaintenanceExitCode.SUCCESS
    assert "success" not in result.stderr


def test_a_refused_move_keeps_its_own_status():
    """
    Distinct from a branch needing attention: the branch is fine and the move was wrong.
    """
    assert (
        exit_code_for(a_report(restack_outcome=RestackOutcome.REFUSED))
        == MaintenanceExitCode.MOVE_REFUSED
    )


# %% the command line a caller acts on the exit status of


@dataclass(frozen=True)
class AlreadyResolvedPass(MaintenancePass):
    """
    A pass whose stack and fork are handed to it, so a whole run can be performed
    without a network or a board file to read.
    """

    resolved_stack: Stack = None  # type: ignore[assignment]
    """
    The stack the commands run against.
    """

    recorded_fork: RecordingPullRequests = dataclasses_field(
        default_factory=RecordingPullRequests
    )
    """
    The stand-in that records writes instead of making them.
    """

    @classmethod
    def over(
        cls, checkout: ForkCheckout, pull_requests: list[PullRequest]
    ) -> AlreadyResolvedPass:
        """
        :param checkout: The checkout to execute in.
        :param pull_requests: The board entries the stack is derived from.
        :return: The pass, with its stack already derived.
        """
        return cls(
            configuration=make_configuration(),
            git=checkout.git,
            resolved_stack=a_stack(checkout, pull_requests),
        )

    def stack(self) -> Stack:
        """
        :return: The stack derived from the scratch checkout.
        """
        return self.resolved_stack

    def fork(self) -> RecordingPullRequests:
        """
        :return: The stand-in that records writes instead of making them.
        """
        return self.recorded_fork


def run_maintenance(
    checkout: ForkCheckout, command: MaintenanceCommand, *flags: str
) -> subprocess.CompletedProcess[str]:
    """
    Invoke the executor as a caller does, so its exit status is exercised.

    The credential is stripped from the environment for every one of these, so no
    assertion about an exit status can come out differently on a machine that happens to
    have a token exported. A run that needed one is exactly the case worth asserting
    about, and it cannot be asserted at all if the ambient environment answers it.

    :param checkout: The checkout to run in.
    :param command: The command to invoke.
    :param flags: That command's own flags.
    :return: The finished subprocess.
    """
    return subprocess.run(
        [sys.executable, str(MAINTENANCE_SCRIPT), command.invoked_as, *flags],
        capture_output=True,
        text=True,
        cwd=checkout.project_root,
        env={
            name: value
            for name, value in os.environ.items()
            if name not in CREDENTIAL_VARIABLES
        },
    )


def test_an_unknown_command_is_a_usage_error(fork_checkout: ForkCheckout):
    result = subprocess.run(
        [sys.executable, str(MAINTENANCE_SCRIPT), "not-a-command"],
        capture_output=True,
        text=True,
        cwd=fork_checkout.project_root,
    )

    assert result.returncode == MaintenanceExitCode.USAGE


def test_every_command_class_is_one_reachable_command():
    """
    Commands are found from their own subclasses, so a class that exists is a command -
    and two answering to the same name would make one of them unreachable.
    """
    assert {type(command) for command in COMMANDS} == set(
        MaintenanceCommand.__subclasses__()
    )
    assert len({command.invoked_as for command in COMMANDS}) == len(COMMANDS)


def test_a_command_names_itself_without_being_instantiated():
    """
    The parser reads both to build the command line before any command is constructed,
    so they have to answer on the class.
    """
    assert BoardCommand.invoked_as == "board"
    assert isinstance(BoardCommand.description, str)


def test_a_command_that_omits_its_name_cannot_be_built():
    """
    A command stays abstract until it says what it is called, and COMMANDS builds every
    subclass - so one that never says cannot reach the parser at all.
    """

    @dataclass(frozen=True)
    class CommandWithoutAName(MaintenanceCommand):
        @classproperty
        def description(cls) -> str:
            return "a command that forgot what it is called"

        def run(self, maintenance, arguments):
            return MaintenanceExitCode.SUCCESS

    with pytest.raises(TypeError, match="invoked_as"):
        CommandWithoutAName()


def test_a_command_answers_with_its_own_name_rather_than_the_one_on_its_base():
    """
    An abstract class property answers with itself rather than calling its accessor,
    which is what leaves a subclass supplying nothing abstract instead of silently
    answering ``None``.
    """
    assert isinstance(MaintenanceCommand.invoked_as, classproperty)
    assert MaintenanceCommand.invoked_as.__isabstractmethod__
    assert BoardCommand.invoked_as == "board"


def test_a_fork_client_that_cannot_make_one_of_the_writes_cannot_be_built():
    """
    The reading and the writing halves are inherited rather than matched by shape, so a
    client that could not post the comment a conflict is reported in is refused when it
    is built, rather than when the first conflict needs reporting.
    """

    @dataclass(frozen=True)
    class ForkThatCannotComment(ForkPullRequests):
        def open_pull_requests(self) -> list[dict]:
            return []

        def pull_request(self, number: int) -> dict:
            return {}

        def replace_labels(self, number: int, labels: Sequence[str]) -> None:
            """
            Recorded nowhere, since this fork is never expected to be built.
            """

        def set_description(self, number: int, body: str) -> None:
            """
            Recorded nowhere, since this fork is never expected to be built.
            """

    with pytest.raises(TypeError, match="add_comment"):
        ForkThatCannotComment()


def test_every_module_of_the_executor_imports_on_its_own():
    """
    The executor is split across modules, and a cycle between two of them shows up only
    when whichever one a caller imports first is the one that has to be complete.
    """
    modules = sorted(path.stem for path in STACK_DIRECTORY.glob("*.py"))

    for module in modules:
        result = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=STACK_DIRECTORY,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{module}: {result.stderr}"


def test_a_restack_step_that_does_nothing_cannot_be_built():
    """
    A step is only ever reached through RESTACK_STEPS, where one that concludes nothing
    would silently pass every branch along to the next step.
    """

    @dataclass(frozen=True)
    class StepWithoutAnAttempt(RestackStep):
        pass

    with pytest.raises(TypeError, match="attempt"):
        StepWithoutAnAttempt()


def test_every_command_is_reachable_from_the_command_line(fork_checkout: ForkCheckout):
    """
    A command in the enum that the parser never registers is unreachable, and nothing
    else would notice.
    """
    for command in COMMANDS:
        result = run_maintenance(fork_checkout, command, "--help")
        assert result.returncode == MaintenanceExitCode.SUCCESS, result.stderr


def test_a_run_needing_a_credential_it_has_not_got_is_its_own_exit_status(
    fork_checkout: ForkCheckout,
):
    """
    Distinguishable from a missing board, since the fix is a token rather than a fetch.
    """
    fork_checkout.run_git("remote", "remove", "cram2")

    assert (
        run_maintenance(fork_checkout, BoardCommand()).returncode
        == MaintenanceExitCode.CREDENTIAL_UNAVAILABLE
    )


def test_a_missing_board_is_reported_ahead_of_a_missing_credential(
    fork_checkout: ForkCheckout,
):
    """
    ``restack`` needs both, and the board is the one its caller fixes with the previous
    command - so reporting the credential first would send them after the wrong thing.

    The upstream remote is dropped first because a subprocess reads the committed
    ``stack.toml``, whose upstream is this repository's own - against which both of the
    fixture's remotes look like candidate forks, and inference rightly refuses to guess.
    """
    fork_checkout.run_git("remote", "remove", "cram2")

    assert (
        run_maintenance(fork_checkout, RestackCommand()).returncode
        == MaintenanceExitCode.BOARD_UNAVAILABLE
    )
