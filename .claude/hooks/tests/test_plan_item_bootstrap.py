"""
Tests for plan_item_bootstrap.py's two operations, recording an item and opening its
work.

Run against the local scratch repository fixture rather than a real remote, and against
a recording pull request opener rather than GitHub, so nothing here needs network access
or credentials.

Every manifest line asserted on is rendered by the :class:`ManifestKey` that owns it,
and every path by the :class:`PlanDocument` that lives at it, so a test cannot pin a
second, independently-drifting copy of the manifest's own vocabulary.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

import plan_item_bootstrap
from plan_item_bootstrap import (
    BLOCK_STYLED_KEYS,
    PLANS_DIRECTORY,
    CreatedPullRequest,
    ExitCode,
    HookScript,
    ItemRecordRequest,
    ItemStatus,
    KeySpecification,
    ManifestKey,
    PlanDocument,
    PullRequestRequest,
    UnknownItemError,
    UnknownPlanError,
    ValueStyle,
    WorkOpenRequest,
    open_work,
    record_item,
)
from scratch_repository import WORK_BRANCH, ScratchRepository

FIXTURES_DIRECTORY = Path(__file__).parent / "fixtures"

PLAN_IDENTIFIER = "test-plan"

PLAN_MANIFEST = (FIXTURES_DIRECTORY / "bootstrap-plan.yaml").read_text()
"""
The manifest every test starts from.
"""

PLAN_ROADMAP = (FIXTURES_DIRECTORY / "bootstrap-roadmap.md").read_text()
"""
The roadmap every test starts from.
"""

EXISTING_ITEM = "an-existing-item"
"""
The fixture item the plan already tracks, with no branch of its own yet.
"""

NEW_ITEM = "a-brand-new-item"
"""
An item the fixture plan does not track, for the entry-creating path.
"""

NEW_BRANCH = "claude/a-new-branch"
"""
The branch opening the work publishes.
"""

SESSION_URL = "https://example.invalid/session_first"
"""
The session recorded on an item whose work was opened.
"""


def manifest_line(manifest_key: ManifestKey, value: str) -> str:
    """
    One manifest line as the key that owns it writes it.

    :param manifest_key: The key the line sets.
    :param value: The value it carries.
    :return: The rendered line.
    """
    return manifest_key.render(value)


# %% fixtures


@dataclass
class RecordingPullRequestOpener:
    """
    Stands in for the GitHub pull request endpoint, recording what it was asked to
    create instead of calling it.
    """

    number: int = 99
    """
    The pull request number handed back to the caller.
    """

    requests: list[PullRequestRequest] = field(default_factory=list)
    """
    Every request this opener was given, in call order.
    """

    def open_pull_request(self, request: PullRequestRequest) -> CreatedPullRequest:
        """
        Record *request* and hand back a pull request as GitHub would.

        :param request: The pull request to create.
        :return: The created pull request.
        """
        self.requests.append(request)
        return CreatedPullRequest(
            number=self.number,
            html_url=f"https://example.invalid/pull/{self.number}",
        )


@dataclass
class RefusingPullRequestOpener:
    """
    Stands in for a GitHub endpoint that refuses the creation.
    """

    def open_pull_request(self, request: PullRequestRequest) -> CreatedPullRequest:
        """
        Refuse the creation the way the real opener does on a non-success response.

        :param request: The pull request that will not be created.
        :raises PullRequestRefusedError: Always.
        """
        raise plan_item_bootstrap.PullRequestRefusedError(detail="422 refused")


@pytest.fixture
def bootstrap_repository(scratch_repository: ScratchRepository) -> ScratchRepository:
    """
    A scratch repository carrying the hook scripts this module drives, with a plan
    already published on its notes branch.

    :param scratch_repository: The initialized scratch repository and notes remote.
    :return: The same repository, ready to bootstrap an item in.
    """
    scratch_repository.install_hook_scripts(
        HookScript.CONFIGURATION.value,
        HookScript.SAVE_PLAN.value,
        "plan_manifest_tools.py",
        HookScript.PLAN_ITEM_BOOTSTRAP.value,
    )
    scratch_repository.write("README.md", "scratch repo\n")
    scratch_repository.commit_everything("initial commit")
    scratch_repository.publish_notes_branch(
        {
            PlanDocument.MANIFEST.path_within_notes_branch(PLAN_IDENTIFIER): (
                PLAN_MANIFEST
            ),
            PlanDocument.ROADMAP.path_within_notes_branch(PLAN_IDENTIFIER): (
                PLAN_ROADMAP
            ),
        }
    )
    scratch_repository.resolve_notes_remote_to()
    scratch_repository.add_work_remote()
    return scratch_repository


def published_plan(repository: ScratchRepository) -> dict[PlanDocument, str]:
    """
    Read the plan's documents as they actually are on the notes branch, rather than what
    a run reported.

    Asks each document where it lives, so this never states a plan's layout
    independently of the code that owns it.

    :param repository: The scratch repository whose notes remote to read.
    :return: Each document's content.
    """
    checkout = repository.project_root.parent / "published-plan-checkout"
    shutil.rmtree(checkout, ignore_errors=True)
    repository.clone_notes_branch(checkout)
    return {
        document: (
            checkout / document.path_within_notes_branch(PLAN_IDENTIFIER)
        ).read_text()
        for document in PlanDocument
    }


def roadmap_section(repository: ScratchRepository, content: str) -> Path:
    """
    Write a roadmap section to a scratch file, the way a caller hands one over.

    The file is named after its content so that a test overriding the default section
    cannot have its file overwritten by the default one being built alongside it.

    :param repository: The scratch repository to write within.
    :param content: The section's markdown.
    :return: The path written to.
    """
    digest = hashlib.sha256(content.encode()).hexdigest()[:12]
    return repository.write(f"sections/{digest}.md", content)


def record_request(repository: ScratchRepository, **overrides: object):
    """
    Build a record request, overriding only what a test cares about.

    :param repository: The scratch repository the roadmap section is written in.
    :param overrides: Fields to replace on the default request.
    :return: The request.
    """
    defaults = dict(
        plan_identifier=PLAN_IDENTIFIER,
        item_identifier=EXISTING_ITEM,
        status=ItemStatus.IN_PROGRESS,
        roadmap_section_path=roadmap_section(repository, "## A new section\n"),
    )
    defaults.update(overrides)
    return ItemRecordRequest(**defaults)


def open_request(**overrides: object) -> WorkOpenRequest:
    """
    Build a work-open request, overriding only what a test cares about.

    :param overrides: Fields to replace on the default request.
    :return: The request.
    """
    defaults = dict(
        plan_identifier=PLAN_IDENTIFIER,
        item_identifier=EXISTING_ITEM,
        branch=NEW_BRANCH,
        base_branch=WORK_BRANCH,
        session_url=SESSION_URL,
        pull_request_title="An item that has not been started",
        pull_request_body="What it does.",
    )
    defaults.update(overrides)
    return WorkOpenRequest(**defaults)


# %% recording an item


def test_recording_an_existing_item_sets_its_status(
    bootstrap_repository: ScratchRepository,
):
    result = record_item(
        record_request(bootstrap_repository),
        project_root=bootstrap_repository.project_root,
    )

    assert result.exit_code is ExitCode.SUCCESS
    published = published_plan(bootstrap_repository)
    assert (
        manifest_line(ManifestKey.STATUS, ItemStatus.IN_PROGRESS.value)
        in published[PlanDocument.MANIFEST]
    )


def test_recording_leaves_every_other_manifest_line_byte_identical(
    bootstrap_repository: ScratchRepository,
):
    record_item(
        record_request(bootstrap_repository),
        project_root=bootstrap_repository.project_root,
    )

    expected = PLAN_MANIFEST.replace(
        manifest_line(ManifestKey.STATUS, ItemStatus.NOT_STARTED.value),
        manifest_line(ManifestKey.STATUS, ItemStatus.IN_PROGRESS.value),
        1,
    )
    assert published_plan(bootstrap_repository)[PlanDocument.MANIFEST] == expected


def test_recording_appends_the_roadmap_section_without_rewriting_the_roadmap(
    bootstrap_repository: ScratchRepository,
):
    section = "## An appended section\n\nIts body.\n"
    record_item(
        record_request(
            bootstrap_repository,
            roadmap_section_path=roadmap_section(bootstrap_repository, section),
        ),
        project_root=bootstrap_repository.project_root,
    )

    roadmap = published_plan(bootstrap_repository)[PlanDocument.ROADMAP]
    assert roadmap.startswith(PLAN_ROADMAP)
    assert roadmap.endswith(section)


def test_recording_a_new_item_appends_it_to_the_manifest(
    bootstrap_repository: ScratchRepository,
):
    record_item(
        record_request(
            bootstrap_repository,
            item_identifier=NEW_ITEM,
            title="A brand new item",
            track="a-track",
            status=ItemStatus.NOT_STARTED,
        ),
        project_root=bootstrap_repository.project_root,
    )

    manifest = published_plan(bootstrap_repository)[PlanDocument.MANIFEST]
    assert manifest.startswith(PLAN_MANIFEST)
    assert manifest.endswith(
        ManifestKey.IDENTIFIER.render(NEW_ITEM, opening_the_item=True)
        + manifest_line(ManifestKey.TITLE, "A brand new item")
        + manifest_line(ManifestKey.BRANCH, "null")
        + manifest_line(ManifestKey.TRACK, "a-track")
        + manifest_line(ManifestKey.DEPENDS_ON, "[]")
        + manifest_line(ManifestKey.STATUS, ItemStatus.NOT_STARTED.value)
    )


def test_recording_a_new_item_without_a_title_names_the_key_it_needs(
    bootstrap_repository: ScratchRepository,
):
    with pytest.raises(plan_item_bootstrap.IncompleteNewItemError) as refusal:
        record_item(
            record_request(
                bootstrap_repository,
                item_identifier=NEW_ITEM,
                track="a-track",
                status=ItemStatus.NOT_STARTED,
            ),
            project_root=bootstrap_repository.project_root,
        )

    assert refusal.value.missing_keys == (ManifestKey.TITLE,)


def test_recording_against_an_unknown_plan_is_refused(
    bootstrap_repository: ScratchRepository,
):
    with pytest.raises(UnknownPlanError) as refusal:
        record_item(
            record_request(bootstrap_repository, plan_identifier="no-such-plan"),
            project_root=bootstrap_repository.project_root,
        )

    assert refusal.value.plan_identifier == "no-such-plan"


# %% opening the work


def test_opening_writes_the_branch_pull_request_and_session_onto_the_item(
    bootstrap_repository: ScratchRepository,
):
    opener = RecordingPullRequestOpener(number=143)

    result = open_work(
        open_request(),
        project_root=bootstrap_repository.project_root,
        pull_request_opener=opener,
    )

    assert result.exit_code is ExitCode.SUCCESS
    assert result.pull_request_number == 143
    manifest = published_plan(bootstrap_repository)[PlanDocument.MANIFEST]
    for written_key, value in (
        (ManifestKey.BRANCH, NEW_BRANCH),
        (ManifestKey.PULL_REQUEST_NUMBER, "143"),
        (ManifestKey.SESSION, SESSION_URL),
        (ManifestKey.STATUS, ItemStatus.IN_PROGRESS.value),
    ):
        assert manifest_line(written_key, value) in manifest


def test_opening_asks_for_a_draft_pull_request_against_the_plans_repository(
    bootstrap_repository: ScratchRepository,
):
    opener = RecordingPullRequestOpener()

    open_work(
        open_request(),
        project_root=bootstrap_repository.project_root,
        pull_request_opener=opener,
    )

    assert len(opener.requests) == 1
    request = opener.requests[0]
    assert request.draft is True
    assert request.repository == "an-owner/a-repository"
    assert request.head == NEW_BRANCH
    assert request.base == WORK_BRANCH


def test_opening_publishes_the_branch_to_the_repositorys_own_remote(
    bootstrap_repository: ScratchRepository,
):
    open_work(
        open_request(),
        project_root=bootstrap_repository.project_root,
        pull_request_opener=RecordingPullRequestOpener(),
    )

    published = bootstrap_repository.run_git(
        "ls-remote",
        "--heads",
        str(bootstrap_repository.work_remote_path),
        NEW_BRANCH,
    )
    assert NEW_BRANCH in published.stdout


def test_opening_an_already_published_branch_is_refused(
    bootstrap_repository: ScratchRepository,
):
    opener = RecordingPullRequestOpener()
    open_work(
        open_request(),
        project_root=bootstrap_repository.project_root,
        pull_request_opener=opener,
    )

    with pytest.raises(plan_item_bootstrap.BranchAlreadyPublishedError) as refusal:
        open_work(
            open_request(),
            project_root=bootstrap_repository.project_root,
            pull_request_opener=opener,
        )
    assert refusal.value.branch == NEW_BRANCH
    assert len(opener.requests) == 1


def test_opening_an_unknown_item_is_refused_before_anything_is_created(
    bootstrap_repository: ScratchRepository,
):
    opener = RecordingPullRequestOpener()

    with pytest.raises(UnknownItemError):
        open_work(
            open_request(item_identifier="no-such-item"),
            project_root=bootstrap_repository.project_root,
            pull_request_opener=opener,
        )

    assert opener.requests == []
    published = bootstrap_repository.run_git(
        "ls-remote", "--heads", str(bootstrap_repository.work_remote_path)
    )
    assert NEW_BRANCH not in published.stdout


def test_a_refused_pull_request_leaves_the_manifest_untouched(
    bootstrap_repository: ScratchRepository,
):
    with pytest.raises(plan_item_bootstrap.PullRequestRefusedError):
        open_work(
            open_request(),
            project_root=bootstrap_repository.project_root,
            pull_request_opener=RefusingPullRequestOpener(),
        )

    assert published_plan(bootstrap_repository)[PlanDocument.MANIFEST] == PLAN_MANIFEST


def test_a_refused_pull_request_leaves_the_branch_it_already_published(
    bootstrap_repository: ScratchRepository,
):
    with pytest.raises(plan_item_bootstrap.PullRequestRefusedError):
        open_work(
            open_request(),
            project_root=bootstrap_repository.project_root,
            pull_request_opener=RefusingPullRequestOpener(),
        )

    published = bootstrap_repository.run_git(
        "ls-remote",
        "--heads",
        str(bootstrap_repository.work_remote_path),
        NEW_BRANCH,
    )
    assert NEW_BRANCH in published.stdout


def test_a_supplied_pull_request_number_is_recorded_without_creating_one(
    bootstrap_repository: ScratchRepository,
):
    opener = RecordingPullRequestOpener()

    result = open_work(
        open_request(pull_request_number=57),
        project_root=bootstrap_repository.project_root,
        pull_request_opener=opener,
    )

    assert opener.requests == []
    assert result.pull_request_number == 57
    assert (
        manifest_line(ManifestKey.PULL_REQUEST_NUMBER, "57")
        in published_plan(bootstrap_repository)[PlanDocument.MANIFEST]
    )


def test_creating_a_pull_request_without_a_title_or_body_is_refused_before_publishing(
    bootstrap_repository: ScratchRepository,
):
    with pytest.raises(plan_item_bootstrap.PullRequestDetailsMissingError):
        open_work(
            open_request(pull_request_title=None, pull_request_body=None),
            project_root=bootstrap_repository.project_root,
            pull_request_opener=RecordingPullRequestOpener(),
        )

    published = bootstrap_repository.run_git(
        "ls-remote", "--heads", str(bootstrap_repository.work_remote_path)
    )
    assert NEW_BRANCH not in published.stdout


def test_a_supplied_pull_request_number_adopts_the_branch_its_caller_published(
    bootstrap_repository: ScratchRepository,
):
    open_work(
        open_request(),
        project_root=bootstrap_repository.project_root,
        pull_request_opener=RecordingPullRequestOpener(number=99),
    )

    result = open_work(
        open_request(pull_request_number=57),
        project_root=bootstrap_repository.project_root,
        pull_request_opener=RecordingPullRequestOpener(),
    )

    assert result.pull_request_number == 57


# %% the vocabulary the manifest is written in


def test_a_rendered_field_line_matches_how_a_real_manifest_writes_it():
    """
    The renderer's indentation and spacing have to match a manifest written by hand,
    since every other test compares the two.
    """
    assert (
        manifest_line(ManifestKey.STATUS, ItemStatus.NOT_STARTED.value) in PLAN_MANIFEST
    )
    assert manifest_line(ManifestKey.TRACK, "a-track") in PLAN_MANIFEST


def test_a_key_quotes_its_own_value_when_its_style_says_to():
    """
    Quoting is the key's to decide, so no caller has to know that a title is prose and a
    track is a bare identifier.
    """
    assert ManifestKey.TITLE.render("A brand new item").endswith(
        ': "A brand new item"\n'
    )
    assert ManifestKey.TRACK.render("a-track").endswith(": a-track\n")
    assert ManifestKey.TITLE.style is ValueStyle.DOUBLE_QUOTED
    assert ManifestKey.TRACK.style is ValueStyle.PLAIN


def test_every_key_is_a_specification_in_its_own_right():
    """
    Mixing the specification into the enum is what lets a key carry its style without a
    lookup beside it, so the relationship is asserted rather than assumed.
    """
    assert issubclass(ManifestKey, KeySpecification)
    assert all(
        isinstance(manifest_key, KeySpecification) for manifest_key in ManifestKey
    )


def test_every_key_was_declared_as_specification_arguments():
    """
    A member declared as a built ``KeySpecification`` rather than as its argument tuple
    is accepted silently by the enum machinery and lands the whole instance in ``key``.

    This is what catches that.
    """
    assert all(isinstance(manifest_key.key, str) for manifest_key in ManifestKey)
    assert all(
        isinstance(manifest_key.style, ValueStyle) for manifest_key in ManifestKey
    )


def test_a_key_indexes_a_parsed_manifest_by_the_string_it_names():
    """
    A key reads parsed YAML through its own ``key``, which is the manifest's own
    spelling of it.
    """
    item = yaml.safe_load(PLAN_MANIFEST)[ManifestKey.ITEMS.key][0]
    assert item[ManifestKey.IDENTIFIER.key] == EXISTING_ITEM
    assert item[ManifestKey.STATUS.key] == ItemStatus.NOT_STARTED


def test_the_plans_directory_matches_the_shell_configuration_that_owns_it(
    bootstrap_repository: ScratchRepository,
):
    """
    ``PLANS_DIRECTORY`` mirrors ``PLANS_DIR`` in the shell configuration; this is what
    stops the mirror drifting, since the two are edited in different files.
    """
    resolved = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{HookScript.CONFIGURATION.path}" && '
            'printf "%s\\n%s\\n" "${PLANS_DIR}" "$(plan_manifest_path "$1")"',
            "test",
            PLAN_IDENTIFIER,
        ],
        cwd=bootstrap_repository.project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    plans_directory, manifest_path = resolved.stdout.strip().split("\n")
    assert plans_directory == PLANS_DIRECTORY
    assert manifest_path == PlanDocument.MANIFEST.path_within_notes_branch(
        PLAN_IDENTIFIER
    )


def test_only_the_keys_whose_values_run_over_lines_are_block_styled():
    assert BLOCK_STYLED_KEYS == {ManifestKey.NOTES, ManifestKey.BLOCKERS}


# %% exit statuses


def test_every_exit_code_names_itself_from_its_own_member():
    for exit_code in ExitCode:
        assert exit_code.name_for_a_caller == exit_code.name.lower()


def test_each_refusal_carries_its_own_exit_code():
    codes = {
        UnknownPlanError: ExitCode.UNKNOWN_PLAN,
        UnknownItemError: ExitCode.UNKNOWN_ITEM,
        plan_item_bootstrap.IncompleteNewItemError: ExitCode.INCOMPLETE_NEW_ITEM,
        plan_item_bootstrap.BranchAlreadyPublishedError: (
            ExitCode.BRANCH_ALREADY_PUBLISHED
        ),
        plan_item_bootstrap.PullRequestDetailsMissingError: (
            ExitCode.PULL_REQUEST_DETAILS_MISSING
        ),
        plan_item_bootstrap.PullRequestRefusedError: ExitCode.PULL_REQUEST_REFUSED,
    }
    assert {error: error.exit_code for error in codes} == codes


def test_a_refusal_composes_its_message_from_its_own_fields():
    refusal = UnknownItemError(
        plan_identifier=PLAN_IDENTIFIER, item_identifier="no-such-item"
    )
    assert refusal.error_message() in str(refusal)
    assert refusal.suggest_correction() in str(refusal)


# %% the command line


def run_bootstrap(
    repository: ScratchRepository, *arguments: str
) -> subprocess.CompletedProcess[str]:
    """
    Run the scratch layout's plan_item_bootstrap.py with *arguments*.

    :param repository: A fixture-built scratch repository.
    :param arguments: CLI arguments to pass.
    :return: The finished subprocess.
    """
    return subprocess.run(
        [
            "python3",
            str(repository.project_root / HookScript.PLAN_ITEM_BOOTSTRAP.path),
            *arguments,
        ],
        cwd=repository.project_root,
        capture_output=True,
        text=True,
    )


def record_arguments(section: Path, plan: str = PLAN_IDENTIFIER) -> list[str]:
    """
    The command line for recording the existing item.

    :param section: The roadmap section to append.
    :param plan: The plan to record against.
    :return: The arguments.
    """
    return [
        "record",
        "--plan",
        plan,
        "--item",
        EXISTING_ITEM,
        "--status",
        ItemStatus.IN_PROGRESS.value,
        "--roadmap-section",
        str(section),
    ]


def test_the_record_subcommand_reports_status_and_exit_code_first(
    bootstrap_repository: ScratchRepository,
):
    section = roadmap_section(bootstrap_repository, "## From the command line\n")

    result = run_bootstrap(bootstrap_repository, *record_arguments(section))

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert list(report)[:2] == ["status", "exit_code"]
    assert report["status"] == ExitCode.SUCCESS.name_for_a_caller
    assert report["exit_code"] == 0


def test_the_command_line_names_the_status_it_failed_with(
    bootstrap_repository: ScratchRepository,
):
    section = roadmap_section(bootstrap_repository, "## Section\n")

    result = run_bootstrap(
        bootstrap_repository, *record_arguments(section, plan="no-such-plan")
    )

    assert result.returncode == ExitCode.UNKNOWN_PLAN
    assert ExitCode.UNKNOWN_PLAN.name_for_a_caller in result.stderr


def test_the_dashboard_republish_is_handed_back_rather_than_attempted(
    bootstrap_repository: ScratchRepository,
):
    section = roadmap_section(bootstrap_repository, "## Section\n")

    result = run_bootstrap(bootstrap_repository, *record_arguments(section))

    report = json.loads(result.stdout)
    assert report["dashboard_command"] == f"/plan-dashboard {PLAN_IDENTIFIER}"
