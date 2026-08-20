"""
Tests for build_dashboard.py's validation, live-state classification, drift detection,
and rendering.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from build_dashboard import (
    AVAILABLE_MODELS,
    DashboardRenderer,
    DependencyCycle,
    DuplicateItemId,
    InvalidBlockers,
    InvalidDependsOn,
    InvalidManifestRoot,
    InvalidSchemaVersion,
    Item,
    ItemStatus,
    LiveState,
    MalformedPullRequestDataError,
    MAXIMUM_DEPENDENCY_STACK_LEVEL,
    MissingMergeTimestampError,
    Plan,
    PlanValidationError,
    PullRequestLabel,
    PullRequestRecord,
    PullRequestsByRepository,
    PullRequestState,
    StackedItem,
    Track,
    UnknownDependency,
    UnknownStatus,
    UnknownTrack,
    UnknownWave,
    Wave,
    classify_live_state,
    load_pull_requests_by_repository,
    main,
    validate_plan,
)

EXAMPLE_DIRECTORY = Path(__file__).parent.parent / "example"
"""The example-walkthrough.md doc's committed sample plan.yaml/roadmap.md/
pr_data.json - see the tests at the bottom of this file for why."""


def minimal_plan(**overrides: Any) -> dict[str, Any]:
    """
    Build one raw, plan.yaml-shaped ``dict`` with a single not-started item - the
    smallest input :func:`validate_plan`/:meth:`Plan.from_mapping` accept, for tests
    that only care about one specific field.

    :param overrides: Top-level keys to replace in the returned mapping.
    """
    plan = {
        "schema_version": 1,
        "id": "test-plan",
        "title": "Test Plan",
        "description": "A plan.",
        "default_repository": "owner/repo",
        "waves": [{"id": "wave-1", "name": "Wave 1"}],
        "tracks": [{"id": "track-1", "name": "Track 1", "wave": "wave-1"}],
        "items": [
            {
                "id": "a",
                "title": "Item A",
                "branch": "a",
                "track": "track-1",
                "status": "not_started",
            }
        ],
    }
    plan.update(overrides)
    return plan


# %% validate_plan


def test_validate_plan_accepts_a_well_formed_manifest():
    validate_plan(minimal_plan())  # must not raise


def test_validate_plan_rejects_wrong_schema_version():
    with pytest.raises(PlanValidationError) as error:
        validate_plan(minimal_plan(schema_version=2))
    assert isinstance(error.value.problems[0], InvalidSchemaVersion)
    assert error.value.problems[0].actual_value == 2


def test_validate_plan_rejects_duplicate_item_ids():
    items = [
        {
            "id": "a",
            "title": "A",
            "branch": "a",
            "track": "track-1",
            "status": "not_started",
        },
        {
            "id": "a",
            "title": "A again",
            "branch": "a2",
            "track": "track-1",
            "status": "not_started",
        },
    ]
    with pytest.raises(PlanValidationError) as error:
        validate_plan(minimal_plan(items=items))
    duplicate_problems = [
        p for p in error.value.problems if isinstance(p, DuplicateItemId)
    ]
    assert duplicate_problems == [DuplicateItemId(["a"])]


def test_validate_plan_rejects_unknown_track():
    items = [
        {
            "id": "a",
            "title": "A",
            "branch": "a",
            "track": "no-such-track",
            "status": "not_started",
        }
    ]
    with pytest.raises(PlanValidationError) as error:
        validate_plan(minimal_plan(items=items))
    assert any(isinstance(problem, UnknownTrack) for problem in error.value.problems)


def test_validate_plan_rejects_unknown_wave():
    tracks = [{"id": "track-1", "name": "Track 1", "wave": "no-such-wave"}]
    with pytest.raises(PlanValidationError) as error:
        validate_plan(minimal_plan(tracks=tracks))
    assert any(isinstance(problem, UnknownWave) for problem in error.value.problems)


def test_validate_plan_rejects_unknown_depends_on():
    items = [
        {
            "id": "a",
            "title": "A",
            "branch": "a",
            "track": "track-1",
            "status": "not_started",
            "depends_on": ["ghost"],
        }
    ]
    with pytest.raises(PlanValidationError) as error:
        validate_plan(minimal_plan(items=items))
    assert any(
        isinstance(problem, UnknownDependency) for problem in error.value.problems
    )


def test_validate_plan_rejects_depends_on_that_is_not_a_list():
    # A plain string is iterable char-by-char - must be rejected outright,
    # not silently misinterpreted as a list of one-character dependencies.
    items = [
        {
            "id": "a",
            "title": "A",
            "branch": "a",
            "track": "track-1",
            "status": "not_started",
            "depends_on": "b",
        }
    ]
    with pytest.raises(PlanValidationError) as error:
        validate_plan(minimal_plan(items=items))
    assert any(
        isinstance(problem, InvalidDependsOn) for problem in error.value.problems
    )


def test_validate_plan_rejects_blockers_that_is_not_a_list():
    # A plain string is iterable char-by-char - must be rejected outright,
    # not silently misinterpreted as one blocker per character.
    items = [
        {
            "id": "a",
            "title": "A",
            "branch": "a",
            "track": "track-1",
            "status": "not_started",
            "blockers": "some prose describing the blocker",
        }
    ]
    with pytest.raises(PlanValidationError) as error:
        validate_plan(minimal_plan(items=items))
    assert any(isinstance(problem, InvalidBlockers) for problem in error.value.problems)


def test_validate_plan_rejects_unknown_status():
    items = [
        {
            "id": "a",
            "title": "A",
            "branch": "a",
            "track": "track-1",
            "status": "in-review",
        }
    ]
    with pytest.raises(PlanValidationError) as error:
        validate_plan(minimal_plan(items=items))
    assert any(isinstance(problem, UnknownStatus) for problem in error.value.problems)


def test_validate_plan_collects_every_problem_not_just_the_first():
    with pytest.raises(PlanValidationError) as error:
        validate_plan(minimal_plan(schema_version=2, tracks=[]))
    problem_types = {type(problem) for problem in error.value.problems}
    assert InvalidSchemaVersion in problem_types
    assert UnknownTrack in problem_types


def test_plan_validation_error_message_joins_every_problem_description():
    with pytest.raises(PlanValidationError) as error:
        validate_plan(minimal_plan(schema_version=2))
    assert str(error.value) == "schema_version must be 1, got 2"


def test_validate_plan_rejects_an_empty_manifest_instead_of_crashing():
    # yaml.safe_load returns None for an empty file - validate_plan must
    # report this as a PlanValidationError, not crash with an AttributeError
    # from calling .get() on None.
    with pytest.raises(PlanValidationError) as error:
        validate_plan(None)
    assert isinstance(error.value.problems[0], InvalidManifestRoot)
    assert error.value.problems[0].actual_value is None


def test_validate_plan_rejects_a_non_mapping_manifest_instead_of_crashing():
    with pytest.raises(PlanValidationError) as error:
        validate_plan(["not", "a", "mapping"])
    assert isinstance(error.value.problems[0], InvalidManifestRoot)


def test_validate_plan_rejects_a_same_track_dependency_cycle():
    # A same-track cycle (a depends on b, b depends on a) must not pass
    # validation - without this check the cycle silently drops both items
    # (and anything depending on them) out of the rendered track later, with
    # no indication data was lost.
    items = [
        {
            "id": "a",
            "title": "A",
            "branch": "a",
            "track": "track-1",
            "status": "not_started",
            "depends_on": ["b"],
        },
        {
            "id": "b",
            "title": "B",
            "branch": "b",
            "track": "track-1",
            "status": "not_started",
            "depends_on": ["a"],
        },
    ]
    with pytest.raises(PlanValidationError) as error:
        validate_plan(minimal_plan(items=items))
    assert error.value.problems == [DependencyCycle(["a", "b", "a"])]


# %% ItemStatus / LiveState labels


def test_item_status_display_labels():
    assert ItemStatus.NOT_STARTED.display_label == "Not started"
    assert ItemStatus.DONE.display_label == "Done"


def test_live_state_display_labels_including_no_pull_request():
    assert LiveState.NO_PULL_REQUEST.display_label == "No pull request yet"
    assert LiveState.MERGED.display_label == "Merged"


# %% classify_live_state


def test_classify_live_state_is_closed_unmerged_for_a_closed_unlabeled_pull_request():
    pull_requests_by_repository = {
        "owner/repo": {"1": PullRequestRecord(state=PullRequestState.CLOSED)}
    }
    assert (
        classify_live_state(1, "owner/repo", pull_requests_by_repository)
        is LiveState.CLOSED_UNMERGED
    )


# %% PullRequestRecord.was_merged


def test_was_merged_true_when_github_recorded_a_merge():
    record = PullRequestRecord(
        state=PullRequestState.CLOSED, merged_at=datetime(2026, 1, 1)
    )
    assert record.was_merged


def test_was_merged_true_for_an_out_of_band_merge_marked_by_label():
    # merged_at is never set for a pull request merged by pushing its branch directly
    # and closing by hand - this repo's convention is a "merged" label instead.
    record = PullRequestRecord(
        state=PullRequestState.CLOSED, labels=["in-review", "merged"]
    )
    assert record.was_merged


def test_was_merged_false_for_a_plain_closed_pull_request():
    record = PullRequestRecord(state=PullRequestState.CLOSED, labels=["in-review"])
    assert not record.was_merged


def test_was_merged_false_for_an_open_pull_request():
    record = PullRequestRecord(state=PullRequestState.OPEN)
    assert not record.was_merged


# %% PullRequestRecord.from_mapping - merge signal


def test_from_mapping_rejects_a_closed_entry_without_a_merge_timestamp():
    # An omitted key is a gatherer that never asked GitHub for merged_at, not a
    # statement that the pull request went unmerged - accepting it silently
    # reports every merged pull request as closed-unmerged.
    with pytest.raises(MissingMergeTimestampError):
        PullRequestRecord.from_mapping(
            {"state": "closed", "draft": False, "labels": ["bug"]}
        )


def test_from_mapping_accepts_a_closed_entry_whose_merge_timestamp_is_null():
    record = PullRequestRecord.from_mapping(
        {"state": "closed", "draft": False, "merged_at": None, "labels": ["bug"]}
    )
    assert record == PullRequestRecord(state=PullRequestState.CLOSED, labels=["bug"])


def test_from_mapping_accepts_an_open_entry_without_a_merge_timestamp():
    record = PullRequestRecord.from_mapping({"state": "open", "draft": True})
    assert record == PullRequestRecord(state=PullRequestState.OPEN, draft=True)


# %% PullRequestRecord.identified_labels


def test_identified_labels_recognizes_known_labels():
    record = PullRequestRecord(
        state=PullRequestState.OPEN, labels=["in-review", "merged"]
    )
    assert record.identified_labels == {
        PullRequestLabel.IN_REVIEW,
        PullRequestLabel.MERGED,
    }


def test_identified_labels_silently_excludes_an_unrecognized_label():
    # cram2-link-sent is real label traffic on this repo's own PRs (added by
    # other automation) that this codebase has no reason to recognize.
    record = PullRequestRecord(state=PullRequestState.OPEN, labels=["cram2-link-sent"])
    assert record.identified_labels == frozenset()


def test_identified_labels_empty_when_no_labels():
    record = PullRequestRecord(state=PullRequestState.OPEN)
    assert record.identified_labels == frozenset()


# %% load_pull_requests_by_repository


def test_load_pull_requests_by_repository_parses_every_repository_and_number():
    pull_requests_by_repository = load_pull_requests_by_repository(
        {"owner/repo": {"1": {"state": "open", "draft": True}}}
    )
    assert pull_requests_by_repository["owner/repo"]["1"] == PullRequestRecord(
        state=PullRequestState.OPEN, draft=True
    )


def test_load_pull_requests_by_repository_reports_repository_and_number_for_a_missing_state():
    with pytest.raises(MalformedPullRequestDataError, match="owner/repo#1"):
        load_pull_requests_by_repository({"owner/repo": {"1": {"draft": True}}})


def test_load_pull_requests_by_repository_reports_repository_and_number_for_an_invalid_state():
    with pytest.raises(MalformedPullRequestDataError, match="owner/repo#7"):
        load_pull_requests_by_repository({"owner/repo": {"7": {"state": "sideways"}}})


def test_load_pull_requests_by_repository_reports_repository_and_number_for_a_closed_entry_without_a_merge_timestamp():
    with pytest.raises(MalformedPullRequestDataError, match="owner/repo#103"):
        load_pull_requests_by_repository(
            {"owner/repo": {"103": {"state": "closed", "draft": False}}}
        )


# %% Item / StackedItem - precomputed template values


def test_status_and_drift_css_class_without_drift():
    plain_item = Item(
        title="A", branch="a", track="track-1", status=ItemStatus.IN_PROGRESS
    )
    assert plain_item.status_and_drift_css_class == "status-in_progress"


def test_status_and_drift_css_class_with_drift():
    drifted_item = Item(title="A", branch="a", track="track-1", status=ItemStatus.DONE)
    drifted_item.drift_description = "marked done, but pull request #1 is still open"
    assert drifted_item.status_and_drift_css_class == "status-done has-drift"


def test_is_ready_to_unblock_dependents_true_when_done():
    done_item = Item(title="A", branch="a", track="track-1", status=ItemStatus.DONE)
    assert done_item.is_ready_to_unblock_dependents()


def test_is_ready_to_unblock_dependents_true_when_open_and_ready_for_review():
    open_item = Item(
        title="A", branch="a", track="track-1", status=ItemStatus.IN_PROGRESS
    )
    open_item.live_state = LiveState.OPEN_READY
    assert open_item.is_ready_to_unblock_dependents()


def test_is_ready_to_unblock_dependents_false_while_still_a_draft():
    draft_item = Item(
        title="A", branch="a", track="track-1", status=ItemStatus.IN_PROGRESS
    )
    draft_item.live_state = LiveState.OPEN_DRAFT
    assert not draft_item.is_ready_to_unblock_dependents()


def test_is_ready_to_unblock_dependents_false_when_not_started():
    fresh_item = Item(
        title="A", branch="a", track="track-1", status=ItemStatus.NOT_STARTED
    )
    assert not fresh_item.is_ready_to_unblock_dependents()


def test_is_ready_for_dependent_review_true_while_still_a_draft():
    draft_item = Item(
        title="A", branch="a", track="track-1", status=ItemStatus.IN_PROGRESS
    )
    draft_item.live_state = LiveState.OPEN_DRAFT
    assert draft_item.is_ready_for_dependent_review()


def test_is_ready_for_dependent_review_true_when_merged():
    merged_item = Item(
        title="A", branch="a", track="track-1", status=ItemStatus.IN_PROGRESS
    )
    merged_item.live_state = LiveState.MERGED
    assert merged_item.is_ready_for_dependent_review()


def test_is_ready_for_dependent_review_true_when_done_without_a_pull_request():
    done_item = Item(title="A", branch="a", track="track-1", status=ItemStatus.DONE)
    assert done_item.is_ready_for_dependent_review()


def test_is_ready_for_dependent_review_false_when_there_is_no_pull_request():
    fresh_item = Item(
        title="A", branch="a", track="track-1", status=ItemStatus.NOT_STARTED
    )
    assert not fresh_item.is_ready_for_dependent_review()


def test_stacked_item_indent_style_exposes_both_indent_levels_as_css_variables():
    stacked = StackedItem(
        item=Item(title="A", branch="a", track="track-1", status=ItemStatus.DONE),
        indent_level=2,
        wrap_parent=None,
        indent_level_with_done_hidden=0,
        wrap_parent_with_done_hidden=None,
    )
    assert stacked.indent_style == "--indent-level: 2; --indent-level-hidden-done: 0;"


def test_plan_repository_url():
    plan = Plan(
        id="p",
        title="P",
        description="d",
        default_repository="owner/repo",
        waves=[],
        tracks=[],
        items=[],
    )
    assert plan.repository_url == "https://github.com/owner/repo"


def test_plan_from_mapping_reads_optional_wave_description():
    plan = Plan.from_mapping(
        minimal_plan(waves=[{"id": "wave-1", "name": "Wave 1", "description": "why"}])
    )
    assert plan.waves[0].description == "why"


def test_plan_from_mapping_defaults_missing_wave_description_to_none():
    plan = Plan.from_mapping(minimal_plan())
    assert plan.waves[0].description is None


def test_plan_from_mapping_ignores_an_unexpected_wave_key_instead_of_crashing():
    plan = Plan.from_mapping(
        minimal_plan(
            waves=[{"id": "wave-1", "name": "Wave 1", "future_field": "unused"}]
        )
    )
    assert plan.waves[0].id == "wave-1"


def test_plan_from_mapping_ignores_an_unexpected_track_key_instead_of_crashing():
    plan = Plan.from_mapping(
        minimal_plan(
            tracks=[
                {
                    "id": "track-1",
                    "name": "Track 1",
                    "wave": "wave-1",
                    "future_field": "unused",
                }
            ]
        )
    )
    assert plan.tracks[0].id == "track-1"


def test_item_from_mapping_keeps_an_http_session_url():
    plan = Plan.from_mapping(
        minimal_plan(
            items=[
                {
                    "id": "a",
                    "title": "Item A",
                    "branch": "a",
                    "track": "track-1",
                    "status": "not_started",
                    "session": "https://claude.ai/code/session/abc",
                }
            ]
        )
    )
    assert plan.items[0].session == "https://claude.ai/code/session/abc"


def test_item_from_mapping_rejects_a_non_http_session_url():
    plan = Plan.from_mapping(
        minimal_plan(
            items=[
                {
                    "id": "a",
                    "title": "Item A",
                    "branch": "a",
                    "track": "track-1",
                    "status": "not_started",
                    "session": "javascript:alert(1)",
                }
            ]
        )
    )
    assert plan.items[0].session is None


def test_item_from_mapping_reads_required_and_optional_fields():
    item_instance = Item.from_mapping(
        {
            "id": "a",
            "title": "Item A",
            "branch": "a",
            "track": "track-1",
            "status": "in_progress",
            "pull_request_number": 7,
            "repository": "owner/other-repo",
            "notes": "  some notes  ",
            "depends_on": ["b"],
            "blockers": ["waiting on design review"],
        }
    )
    assert item_instance.id == "a"
    assert item_instance.title == "Item A"
    assert item_instance.branch == "a"
    assert item_instance.track == "track-1"
    assert item_instance.status is ItemStatus.IN_PROGRESS
    assert item_instance.pull_request_number == 7
    assert item_instance.repository == "owner/other-repo"
    assert item_instance.notes == "some notes"
    assert item_instance.depends_on == ["b"]
    assert item_instance.blockers == ["waiting on design review"]


def test_item_from_mapping_defaults_every_optional_field_when_omitted():
    item_instance = Item.from_mapping(
        {
            "title": "Item A",
            "branch": "a",
            "track": "track-1",
            "status": "not_started",
        }
    )
    assert item_instance.id is None
    assert item_instance.pull_request_number is None
    assert item_instance.repository is None
    assert item_instance.session is None
    assert item_instance.notes is None
    assert item_instance.depends_on == []
    assert item_instance.blockers == []


# %% DashboardRenderer - live state + drift

# "Drift" is a manifest item's manually-maintained :class:`ItemStatus`
# disagreeing with its live GitHub :class:`LiveState` - see
# :meth:`DashboardRenderer._drift_description_of` for the exact rules. It
# flags a manifest that's gone stale (e.g. marked ``done`` while its pull
# request is still open - was it marked done too early, or has something
# regressed?) so a human can look, not something this code silently corrects
# (except the one unambiguous direction ``sync_manifest_status.py`` handles).


def make_renderer(
    items: list[Item],
    pull_requests_by_repository: PullRequestsByRepository | None = None,
) -> DashboardRenderer:
    """
    Build one :class:`DashboardRenderer` over a fixed, otherwise-empty test
    plan holding *items* - the shared entry point every live-state/drift test
    in this file renders through.

    :param items: The plan's items.
    :param pull_requests_by_repository: Live pull request state, or ``{}`` if omitted.
    """
    plan = Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository="owner/repo",
        waves=[],
        tracks=[],
        items=items,
    )
    return DashboardRenderer(
        plan=plan,
        roadmap_text="",
        pull_requests_by_repository=pull_requests_by_repository or {},
        tracking_url=None,
    )


def item(
    identifier: str,
    status: ItemStatus,
    pull_request_number: int | None = None,
    depends_on: list[str] | None = None,
) -> Item:
    """
    Build one :class:`Item` for a test, filling in the boilerplate.

    (``title``/``branch``/``id`` all equal to *identifier*, a fixed ``track``)
    that every one of this file's ~90 items would otherwise repeat - a plain
    :class:`Item` constructor call remains available and used directly
    wherever a test needs a field this shortcut doesn't expose.
    """
    return Item(
        title=identifier,
        branch=identifier,
        track="track-1",
        status=status,
        id=identifier,
        pull_request_number=pull_request_number,
        depends_on=depends_on or [],
    )


def test_item_with_no_pull_request_has_no_pull_request_live_state_and_no_drift():
    renderer = make_renderer([item("a", ItemStatus.NOT_STARTED)])
    output, summary = renderer.render()
    assert renderer.plan.items[0].live_state is LiveState.NO_PULL_REQUEST
    assert summary.drift_items == []


def test_merged_pull_request_marks_not_started_item_as_drifted():
    pull_requests_by_repository = {
        "owner/repo": {
            "1": PullRequestRecord(
                state=PullRequestState.CLOSED, merged_at=datetime(2026, 1, 1)
            )
        }
    }
    renderer = make_renderer(
        [item("a", ItemStatus.NOT_STARTED, pull_request_number=1)],
        pull_requests_by_repository=pull_requests_by_repository,
    )
    _, summary = renderer.render()
    assert summary.drift_items == ["a"]
    assert renderer.plan.items[0].live_state is LiveState.MERGED


def test_closed_pull_request_with_merged_label_is_merged_not_closed_unmerged():
    # merged_at is unset here on purpose - this is the out-of-band-merge case
    # PullRequestRecord.was_merged exists for.
    pull_requests_by_repository = {
        "owner/repo": {
            "1": PullRequestRecord(state=PullRequestState.CLOSED, labels=["merged"])
        }
    }
    renderer = make_renderer(
        [item("a", ItemStatus.IN_PROGRESS, pull_request_number=1)],
        pull_requests_by_repository=pull_requests_by_repository,
    )
    _, summary = renderer.render()
    assert renderer.plan.items[0].live_state is LiveState.MERGED
    assert summary.drift_items == ["a"]


def test_open_pull_request_marks_done_item_as_drifted():
    pull_requests_by_repository = {
        "owner/repo": {"1": PullRequestRecord(state=PullRequestState.OPEN, draft=False)}
    }
    renderer = make_renderer(
        [item("a", ItemStatus.DONE, pull_request_number=1)],
        pull_requests_by_repository=pull_requests_by_repository,
    )
    _, summary = renderer.render()
    assert summary.drift_items == ["a"]


def test_pull_request_missing_from_live_data_is_not_found_and_drifted():
    renderer = make_renderer(
        [item("a", ItemStatus.NOT_STARTED, pull_request_number=999)]
    )
    _, summary = renderer.render()
    assert renderer.plan.items[0].live_state is LiveState.NOT_FOUND
    assert summary.drift_items == ["a"]


def test_closed_unmerged_pull_request_marks_done_item_as_drifted():
    pull_requests_by_repository = {
        "owner/repo": {"1": PullRequestRecord(state=PullRequestState.CLOSED)}
    }
    renderer = make_renderer(
        [item("a", ItemStatus.DONE, pull_request_number=1)],
        pull_requests_by_repository=pull_requests_by_repository,
    )
    _, summary = renderer.render()
    assert renderer.plan.items[0].live_state is LiveState.CLOSED_UNMERGED
    assert summary.drift_items == ["a"]


@pytest.mark.parametrize("status", [ItemStatus.IN_PROGRESS, ItemStatus.BLOCKED])
def test_closed_unmerged_pull_request_marks_in_progress_or_blocked_item_as_drifted(
    status,
):
    pull_requests_by_repository = {
        "owner/repo": {"1": PullRequestRecord(state=PullRequestState.CLOSED)}
    }
    renderer = make_renderer(
        [item("a", status, pull_request_number=1)],
        pull_requests_by_repository=pull_requests_by_repository,
    )
    _, summary = renderer.render()
    assert summary.drift_items == ["a"]


def test_matching_status_and_live_state_is_not_drifted():
    pull_requests_by_repository = {
        "owner/repo": {"1": PullRequestRecord(state=PullRequestState.OPEN, draft=True)}
    }
    renderer = make_renderer(
        [item("a", ItemStatus.IN_PROGRESS, pull_request_number=1)],
        pull_requests_by_repository=pull_requests_by_repository,
    )
    _, summary = renderer.render()
    assert summary.drift_items == []


# %% DashboardRenderer - ready-to-start / blocker-maybe-cleared


def test_item_becomes_ready_to_start_once_all_dependencies_are_done():
    items = [
        item("a", ItemStatus.DONE),
        item("b", ItemStatus.NOT_STARTED, depends_on=["a"]),
    ]
    renderer = make_renderer(items)
    _, summary = renderer.render()
    assert summary.ready_to_start == ["b"]


def test_blocked_item_with_partial_dependencies_done_is_recheck_candidate():
    items = [
        item("a", ItemStatus.DONE),
        item("b", ItemStatus.BLOCKED),
        item("c", ItemStatus.BLOCKED, depends_on=["a", "b"]),
    ]
    renderer = make_renderer(items)
    _, summary = renderer.render()
    assert summary.blocker_maybe_cleared == ["c"]
    assert summary.ready_to_start == []


def test_blocked_item_with_every_dependency_done_is_recheck_not_ready_to_start():
    # A blocked item is still blocked even once its dependencies clear -
    # it belongs in "blocker may be cleared" (actionable: resolve it), never
    # in "ready to start" (that implies starting fresh, which is wrong for
    # an item that already has real state).
    items = [
        item("a", ItemStatus.DONE),
        item("b", ItemStatus.BLOCKED, depends_on=["a"]),
    ]
    renderer = make_renderer(items)
    _, summary = renderer.render()
    assert summary.blocker_maybe_cleared == ["b"]
    assert summary.ready_to_start == []


def test_item_becomes_ready_to_start_once_dependency_is_open_and_ready_for_review():
    # Stacking a branch on a same-track dependency that's already open and
    # out of draft is this repo's normal workflow - waiting for a full merge
    # first would be stricter than how the stack is actually built.
    pull_requests_by_repository = {
        "owner/repo": {"1": PullRequestRecord(state=PullRequestState.OPEN, draft=False)}
    }
    items = [
        item("a", ItemStatus.IN_PROGRESS, pull_request_number=1),
        item("b", ItemStatus.NOT_STARTED, depends_on=["a"]),
    ]
    renderer = make_renderer(
        items, pull_requests_by_repository=pull_requests_by_repository
    )
    _, summary = renderer.render()
    assert summary.ready_to_start == ["b"]


def test_item_not_ready_to_start_while_dependency_is_still_a_draft():
    pull_requests_by_repository = {
        "owner/repo": {"1": PullRequestRecord(state=PullRequestState.OPEN, draft=True)}
    }
    items = [
        item("a", ItemStatus.IN_PROGRESS, pull_request_number=1),
        item("b", ItemStatus.NOT_STARTED, depends_on=["a"]),
    ]
    renderer = make_renderer(
        items, pull_requests_by_repository=pull_requests_by_repository
    )
    _, summary = renderer.render()
    assert summary.ready_to_start == []


def test_not_started_item_with_partial_dependencies_is_neither_list():
    items = [
        item("a", ItemStatus.DONE),
        item("b", ItemStatus.BLOCKED),
        item("c", ItemStatus.NOT_STARTED, depends_on=["a", "b"]),
    ]
    renderer = make_renderer(items)
    _, summary = renderer.render()
    assert summary.ready_to_start == []
    assert summary.blocker_maybe_cleared == []


def test_dependency_free_not_started_item_is_ready_to_start():
    items = [item("a", ItemStatus.NOT_STARTED)]
    renderer = make_renderer(items)
    _, summary = renderer.render()
    assert summary.ready_to_start == ["a"]


def test_dependency_free_blocked_item_is_neither_list():
    items = [item("a", ItemStatus.BLOCKED)]
    renderer = make_renderer(items)
    _, summary = renderer.render()
    assert summary.ready_to_start == []
    assert summary.blocker_maybe_cleared == []


# %% DashboardRenderer - ready-to-review


def test_needs_review_true_for_an_open_draft_pull_request():
    pull_requests_by_repository = {
        "owner/repo": {"1": PullRequestRecord(state=PullRequestState.OPEN, draft=True)}
    }
    renderer = make_renderer(
        [item("a", ItemStatus.IN_PROGRESS, pull_request_number=1)],
        pull_requests_by_repository=pull_requests_by_repository,
    )
    renderer.render()
    assert renderer.plan.items[0].needs_review


def test_needs_review_false_once_marked_ready_for_review():
    pull_requests_by_repository = {
        "owner/repo": {"1": PullRequestRecord(state=PullRequestState.OPEN, draft=False)}
    }
    renderer = make_renderer(
        [item("a", ItemStatus.IN_PROGRESS, pull_request_number=1)],
        pull_requests_by_repository=pull_requests_by_repository,
    )
    renderer.render()
    assert not renderer.plan.items[0].needs_review


def test_needs_review_false_with_no_pull_request():
    renderer = make_renderer([item("a", ItemStatus.NOT_STARTED)])
    renderer.render()
    assert not renderer.plan.items[0].needs_review


def test_needs_review_false_for_a_deferred_item_with_an_open_draft_pull_request():
    pull_requests_by_repository = {
        "owner/repo": {"1": PullRequestRecord(state=PullRequestState.OPEN, draft=True)}
    }
    renderer = make_renderer(
        [item("a", ItemStatus.DEFERRED, pull_request_number=1)],
        pull_requests_by_repository=pull_requests_by_repository,
    )
    renderer.render()
    assert not renderer.plan.items[0].needs_review


def test_has_open_pull_request_true_for_draft_and_ready():
    draft_item = item("a", ItemStatus.IN_PROGRESS, pull_request_number=1)
    draft_item.live_state = LiveState.OPEN_DRAFT
    ready_item = item("b", ItemStatus.IN_PROGRESS, pull_request_number=2)
    ready_item.live_state = LiveState.OPEN_READY
    assert draft_item.has_open_pull_request
    assert ready_item.has_open_pull_request


def test_has_open_pull_request_false_when_merged_or_absent():
    merged_item = item("a", ItemStatus.DONE, pull_request_number=1)
    merged_item.live_state = LiveState.MERGED
    no_pull_request_item = item("b", ItemStatus.NOT_STARTED)
    assert not merged_item.has_open_pull_request
    assert not no_pull_request_item.has_open_pull_request


def test_item_with_no_dependency_and_draft_pull_request_is_ready_to_review():
    pull_requests_by_repository = {
        "owner/repo": {"1": PullRequestRecord(state=PullRequestState.OPEN, draft=True)}
    }
    renderer = make_renderer(
        [item("a", ItemStatus.IN_PROGRESS, pull_request_number=1)],
        pull_requests_by_repository=pull_requests_by_repository,
    )
    _, summary = renderer.render()
    assert summary.ready_to_review == ["a"]


def test_blocked_item_with_draft_pull_request_is_not_ready_to_review():
    pull_requests_by_repository = {
        "owner/repo": {"1": PullRequestRecord(state=PullRequestState.OPEN, draft=True)}
    }
    renderer = make_renderer(
        [item("a", ItemStatus.BLOCKED, pull_request_number=1)],
        pull_requests_by_repository=pull_requests_by_repository,
    )
    _, summary = renderer.render()
    assert summary.ready_to_review == []


def test_deferred_item_with_draft_pull_request_is_not_ready_to_review():
    pull_requests_by_repository = {
        "owner/repo": {"1": PullRequestRecord(state=PullRequestState.OPEN, draft=True)}
    }
    renderer = make_renderer(
        [item("a", ItemStatus.DEFERRED, pull_request_number=1)],
        pull_requests_by_repository=pull_requests_by_repository,
    )
    _, summary = renderer.render()
    assert summary.ready_to_review == []


def test_item_not_ready_to_review_while_dependency_has_no_pull_request():
    pull_requests_by_repository = {
        "owner/repo": {"2": PullRequestRecord(state=PullRequestState.OPEN, draft=True)}
    }
    items = [
        item("a", ItemStatus.NOT_STARTED),
        item("b", ItemStatus.IN_PROGRESS, pull_request_number=2, depends_on=["a"]),
    ]
    renderer = make_renderer(
        items, pull_requests_by_repository=pull_requests_by_repository
    )
    _, summary = renderer.render()
    assert summary.ready_to_review == []


def test_item_ready_to_review_once_dependency_has_an_open_pull_request():
    # The dependency need not itself be past review - it just needs a pull
    # request open, so a whole reviewable stack can surface before its base merges.
    pull_requests_by_repository = {
        "owner/repo": {
            "1": PullRequestRecord(state=PullRequestState.OPEN, draft=True),
            "2": PullRequestRecord(state=PullRequestState.OPEN, draft=True),
        }
    }
    items = [
        item("a", ItemStatus.IN_PROGRESS, pull_request_number=1),
        item("b", ItemStatus.IN_PROGRESS, pull_request_number=2, depends_on=["a"]),
    ]
    renderer = make_renderer(
        items, pull_requests_by_repository=pull_requests_by_repository
    )
    _, summary = renderer.render()
    assert summary.ready_to_review == ["a", "b"]


def test_item_ready_to_review_once_its_dependency_has_merged():
    pull_requests_by_repository = {
        "owner/repo": {
            "1": PullRequestRecord(
                state=PullRequestState.CLOSED, merged_at=datetime(2026, 1, 1)
            ),
            "2": PullRequestRecord(state=PullRequestState.OPEN, draft=True),
        }
    }
    items = [
        item("a", ItemStatus.DONE, pull_request_number=1),
        item("b", ItemStatus.IN_PROGRESS, pull_request_number=2, depends_on=["a"]),
    ]
    renderer = make_renderer(
        items, pull_requests_by_repository=pull_requests_by_repository
    )
    _, summary = renderer.render()
    assert summary.ready_to_review == ["b"]


def test_item_ready_to_review_when_its_dependency_is_done_without_a_pull_request():
    pull_requests_by_repository = {
        "owner/repo": {"2": PullRequestRecord(state=PullRequestState.OPEN, draft=True)}
    }
    items = [
        item("a", ItemStatus.DONE),
        item("b", ItemStatus.IN_PROGRESS, pull_request_number=2, depends_on=["a"]),
    ]
    renderer = make_renderer(
        items, pull_requests_by_repository=pull_requests_by_repository
    )
    _, summary = renderer.render()
    assert summary.ready_to_review == ["b"]


def test_item_not_ready_to_review_while_its_dependency_is_closed_unmerged():
    pull_requests_by_repository = {
        "owner/repo": {
            "1": PullRequestRecord(state=PullRequestState.CLOSED),
            "2": PullRequestRecord(state=PullRequestState.OPEN, draft=True),
        }
    }
    items = [
        item("a", ItemStatus.IN_PROGRESS, pull_request_number=1),
        item("b", ItemStatus.IN_PROGRESS, pull_request_number=2, depends_on=["a"]),
    ]
    renderer = make_renderer(
        items, pull_requests_by_repository=pull_requests_by_repository
    )
    _, summary = renderer.render()
    assert summary.ready_to_review == []


# %% DashboardRenderer - bug-fix marking


def test_item_is_a_bug_fix_when_its_pull_request_carries_the_bug_label():
    pull_requests_by_repository = {
        "owner/repo": {
            "1": PullRequestRecord(
                state=PullRequestState.OPEN,
                draft=True,
                labels=[PullRequestLabel.BUG.value],
            )
        }
    }
    renderer = make_renderer(
        [item("a", ItemStatus.IN_PROGRESS, pull_request_number=1)],
        pull_requests_by_repository=pull_requests_by_repository,
    )
    renderer.render()
    assert renderer.plan.items[0].is_bug_fix is True


def test_item_is_not_a_bug_fix_when_its_pull_request_carries_other_labels():
    pull_requests_by_repository = {
        "owner/repo": {
            "1": PullRequestRecord(
                state=PullRequestState.OPEN,
                draft=True,
                labels=[PullRequestLabel.IN_REVIEW.value, "enhancement"],
            )
        }
    }
    renderer = make_renderer(
        [item("a", ItemStatus.IN_PROGRESS, pull_request_number=1)],
        pull_requests_by_repository=pull_requests_by_repository,
    )
    renderer.render()
    assert renderer.plan.items[0].is_bug_fix is False


def test_item_without_a_pull_request_is_not_a_bug_fix():
    renderer = make_renderer([item("a", ItemStatus.NOT_STARTED)])
    renderer.render()
    assert renderer.plan.items[0].is_bug_fix is False


def test_bug_fix_stays_in_its_ordinary_action_group():
    # A bug fix is an attribute of an item, not an action of its own: it must
    # not be lifted out of the action group it already belongs to.
    pull_requests_by_repository = {
        "owner/repo": {
            "1": PullRequestRecord(
                state=PullRequestState.OPEN,
                draft=True,
                labels=[PullRequestLabel.BUG.value],
            )
        }
    }
    renderer = make_renderer(
        [item("a", ItemStatus.IN_PROGRESS, pull_request_number=1)],
        pull_requests_by_repository=pull_requests_by_repository,
    )
    _, summary = renderer.render()
    assert summary.ready_to_review == ["a"]


def test_render_marks_a_bug_fix_sidebar_entry_with_a_bug_chip():
    pull_requests_by_repository = {
        "owner/repo": {
            "1": PullRequestRecord(
                state=PullRequestState.OPEN,
                draft=True,
                labels=[PullRequestLabel.BUG.value],
            )
        }
    }
    renderer = make_renderer(
        [item("a", ItemStatus.IN_PROGRESS, pull_request_number=1)],
        pull_requests_by_repository=pull_requests_by_repository,
    )
    output, _ = renderer.render()
    assert '<span class="next-bug-chip">bug</span>' in output


def test_render_leaves_a_non_bug_sidebar_entry_without_a_bug_chip():
    pull_requests_by_repository = {
        "owner/repo": {"1": PullRequestRecord(state=PullRequestState.OPEN, draft=True)}
    }
    renderer = make_renderer(
        [item("a", ItemStatus.IN_PROGRESS, pull_request_number=1)],
        pull_requests_by_repository=pull_requests_by_repository,
    )
    output, _ = renderer.render()
    assert '<span class="next-bug-chip">bug</span>' not in output


# %% DashboardRenderer - bug-fixes-only sidebar filter


def _renderer_with_one_bug_fix_and_one_other_entry() -> DashboardRenderer:
    """Build a renderer whose sidebar holds two entries in two different
    groups, exactly one of which is a bug fix - the smallest input that
    distinguishes the filter's per-entry and per-group behaviour."""
    pull_requests_by_repository = {
        "owner/repo": {
            "1": PullRequestRecord(
                state=PullRequestState.OPEN,
                draft=True,
                labels=[PullRequestLabel.BUG.value],
            ),
            "2": PullRequestRecord(state=PullRequestState.OPEN, draft=False),
        }
    }
    return make_renderer(
        [
            item("a", ItemStatus.IN_PROGRESS, pull_request_number=1),
            item("b", ItemStatus.DONE, pull_request_number=2),
        ],
        pull_requests_by_repository=pull_requests_by_repository,
    )


def test_bug_fixes_only_toggle_is_offered_when_a_sidebar_entry_is_a_bug_fix():
    renderer = _renderer_with_one_bug_fix_and_one_other_entry()
    output, _ = renderer.render()
    assert 'id="bug-fixes-only-toggle"' in output


def test_bug_fixes_only_toggle_is_omitted_when_no_sidebar_entry_is_a_bug_fix():
    pull_requests_by_repository = {
        "owner/repo": {"1": PullRequestRecord(state=PullRequestState.OPEN, draft=True)}
    }
    renderer = make_renderer(
        [item("a", ItemStatus.IN_PROGRESS, pull_request_number=1)],
        pull_requests_by_repository=pull_requests_by_repository,
    )
    output, _ = renderer.render()
    assert 'id="bug-fixes-only-toggle"' not in output


def test_bug_fix_entry_is_marked_so_the_filter_can_keep_it():
    renderer = _renderer_with_one_bug_fix_and_one_other_entry()
    output, _ = renderer.render()
    assert '<li class="next-entry next-entry-bug">' in output
    assert '<li class="next-entry">' in output


def test_group_holding_a_bug_fix_is_marked_so_the_filter_can_keep_it():
    renderer = _renderer_with_one_bug_fix_and_one_other_entry()
    output, _ = renderer.render()
    assert '<div class="next-group next-review next-group-has-bugs">' in output
    assert '<div class="next-group next-drift">' in output


def test_group_heading_carries_both_the_total_and_the_bug_fix_count():
    renderer = _renderer_with_one_bug_fix_and_one_other_entry()
    output, _ = renderer.render()
    assert (
        'Ready to review <span class="next-count-all">(1)</span>'
        '<span class="next-count-bug">(1)</span>' in output
    )
    assert (
        'Fix the manifest <span class="next-count-all">(1)</span>'
        '<span class="next-count-bug">(0)</span>' in output
    )


# %% DashboardRenderer - item action button


def test_action_is_start_now_for_a_not_started_item():
    renderer = make_renderer([item("a", ItemStatus.NOT_STARTED)])
    renderer.render()
    action = renderer.plan.items[0].action
    assert action.label == "Start now"
    assert action.command == "/plan-item-kickoff test-plan a"


def test_action_none_for_a_not_started_item_while_a_dependency_is_not_ready():
    items = [
        item("a", ItemStatus.NOT_STARTED),
        item("b", ItemStatus.NOT_STARTED, depends_on=["a"]),
    ]
    renderer = make_renderer(items)
    renderer.render()
    assert renderer.items_by_identifier["b"].action is None


def test_action_set_once_every_dependency_is_ready():
    items = [
        item("a", ItemStatus.DONE),
        item("b", ItemStatus.NOT_STARTED, depends_on=["a"]),
    ]
    renderer = make_renderer(items)
    renderer.render()
    assert (
        renderer.items_by_identifier["b"].action.command
        == "/plan-item-kickoff test-plan b"
    )


def test_action_set_when_dependency_is_open_and_ready_for_review():
    pull_requests_by_repository = {
        "owner/repo": {"1": PullRequestRecord(state=PullRequestState.OPEN, draft=False)}
    }
    items = [
        item("a", ItemStatus.IN_PROGRESS, pull_request_number=1),
        item("b", ItemStatus.NOT_STARTED, depends_on=["a"]),
    ]
    renderer = make_renderer(
        items, pull_requests_by_repository=pull_requests_by_repository
    )
    renderer.render()
    assert renderer.items_by_identifier["b"].action is not None


def test_action_none_for_a_not_started_item_when_dependency_is_still_a_draft():
    pull_requests_by_repository = {
        "owner/repo": {"1": PullRequestRecord(state=PullRequestState.OPEN, draft=True)}
    }
    items = [
        item("a", ItemStatus.IN_PROGRESS, pull_request_number=1),
        item("b", ItemStatus.NOT_STARTED, depends_on=["a"]),
    ]
    renderer = make_renderer(
        items, pull_requests_by_repository=pull_requests_by_repository
    )
    renderer.render()
    assert renderer.items_by_identifier["b"].action is None


def test_action_ready_check_is_order_independent():
    # "b" depends on "a", but "a" appears later in plan.items - the
    # dependency's live_state must still be classified before "b"'s
    # action is computed.
    items = [
        item("b", ItemStatus.NOT_STARTED, depends_on=["a"]),
        item("a", ItemStatus.DONE),
    ]
    renderer = make_renderer(items)
    renderer.render()
    assert (
        renderer.items_by_identifier["b"].action.command
        == "/plan-item-kickoff test-plan b"
    )


def test_action_none_for_a_done_item():
    renderer = make_renderer([item("a", ItemStatus.DONE)])
    renderer.render()
    assert renderer.plan.items[0].action is None


def test_action_is_resolve_for_a_blocked_item():
    renderer = make_renderer([item("a", ItemStatus.BLOCKED)])
    renderer.render()
    action = renderer.plan.items[0].action
    assert action.label == "Resolve"
    assert action.command == "/plan-item-resolve test-plan a"


def test_action_is_resume_for_an_in_progress_item():
    renderer = make_renderer([item("a", ItemStatus.IN_PROGRESS)])
    renderer.render()
    action = renderer.plan.items[0].action
    assert action.label == "Resume"
    assert action.command == "/plan-item-resolve test-plan a"


def test_action_is_reconsider_for_a_deferred_item():
    renderer = make_renderer([item("a", ItemStatus.DEFERRED)])
    renderer.render()
    action = renderer.plan.items[0].action
    assert action.label == "Reconsider"
    assert action.command == "/plan-item-resolve test-plan a"


# %% DashboardRenderer - dependency stacking / wrap-around


def test_track_stack_wraps_past_the_maximum_level():
    # A chain one longer than the cap: item N depends on item N-1.
    chain_length = MAXIMUM_DEPENDENCY_STACK_LEVEL + 2
    items = [item("item-0", ItemStatus.NOT_STARTED)]
    for index in range(1, chain_length):
        items.append(
            item(
                f"item-{index}",
                ItemStatus.NOT_STARTED,
                depends_on=[f"item-{index - 1}"],
            )
        )
    renderer = make_renderer(items)
    stacked_items = renderer._build_track_stack(items)
    assert [stacked.indent_level for stacked in stacked_items] == [0, 1, 2, 3, 4, 0]
    assert stacked_items[-1].wrap_parent.identifier == "item-4"


def test_track_stack_does_not_wrap_within_the_maximum_level():
    items = [item("item-0", ItemStatus.NOT_STARTED)]
    for index in range(1, MAXIMUM_DEPENDENCY_STACK_LEVEL):
        items.append(
            item(
                f"item-{index}",
                ItemStatus.NOT_STARTED,
                depends_on=[f"item-{index - 1}"],
            )
        )
    renderer = make_renderer(items)
    stacked_items = renderer._build_track_stack(items)
    assert all(stacked.wrap_parent is None for stacked in stacked_items)


# %% DashboardRenderer - dependency stacking / hidden-done dedent


def test_hidden_done_indent_dedents_a_dependent_of_a_done_item_to_zero():
    items = [
        item("a", ItemStatus.DONE),
        item("b", ItemStatus.NOT_STARTED, depends_on=["a"]),
    ]
    renderer = make_renderer(items)
    stacked_items = renderer._build_track_stack(items)
    stacked_b = next(s for s in stacked_items if s.item.identifier == "b")
    assert stacked_b.indent_level == 1
    assert stacked_b.indent_level_with_done_hidden == 0
    assert stacked_b.wrap_parent_with_done_hidden is None


def test_hidden_done_indent_unaffected_when_dependency_is_not_done():
    items = [
        item("a", ItemStatus.IN_PROGRESS),
        item("b", ItemStatus.NOT_STARTED, depends_on=["a"]),
    ]
    renderer = make_renderer(items)
    stacked_items = renderer._build_track_stack(items)
    stacked_b = next(s for s in stacked_items if s.item.identifier == "b")
    assert stacked_b.indent_level == 1
    assert stacked_b.indent_level_with_done_hidden == 1


def test_hidden_done_indent_only_dedents_the_immediate_done_dependency():
    # c depends on b (in progress), b depends on a (done). Hiding a only
    # removes b's own dependency on it - c still indents under the still-
    # visible b, one level, not zero.
    items = [
        item("a", ItemStatus.DONE),
        item("b", ItemStatus.IN_PROGRESS, depends_on=["a"]),
        item("c", ItemStatus.NOT_STARTED, depends_on=["b"]),
    ]
    renderer = make_renderer(items)
    stacked_items = renderer._build_track_stack(items)
    stacked_b = next(s for s in stacked_items if s.item.identifier == "b")
    stacked_c = next(s for s in stacked_items if s.item.identifier == "c")
    assert stacked_b.indent_level_with_done_hidden == 0
    assert stacked_c.indent_level_with_done_hidden == 1


def test_hidden_done_indent_skips_a_chain_of_done_dependencies():
    items = [
        item("a", ItemStatus.DONE),
        item("b", ItemStatus.DONE, depends_on=["a"]),
        item("c", ItemStatus.NOT_STARTED, depends_on=["b"]),
    ]
    renderer = make_renderer(items)
    stacked_items = renderer._build_track_stack(items)
    stacked_c = next(s for s in stacked_items if s.item.identifier == "c")
    assert stacked_c.indent_level == 2
    assert stacked_c.indent_level_with_done_hidden == 0


def test_hidden_done_wrap_parent_is_never_a_done_item():
    # A chain of dependencies just long enough to wrap once the two done
    # items at its base are hidden: after hiding, c is the effective root
    # (level 0), d=1, e=2, f=3, g=4, h wraps back to 0 continuing from g -
    # never from a done item, even though the full (unhidden) chain would
    # wrap earlier and reference a different, done, parent.
    items = [
        item("a", ItemStatus.DONE),
        item("b", ItemStatus.DONE, depends_on=["a"]),
        item("c", ItemStatus.NOT_STARTED, depends_on=["b"]),
        item("d", ItemStatus.NOT_STARTED, depends_on=["c"]),
        item("e", ItemStatus.NOT_STARTED, depends_on=["d"]),
        item("f", ItemStatus.NOT_STARTED, depends_on=["e"]),
        item("g", ItemStatus.NOT_STARTED, depends_on=["f"]),
        item("h", ItemStatus.NOT_STARTED, depends_on=["g"]),
    ]
    renderer = make_renderer(items)
    stacked_items = renderer._build_track_stack(items)
    stacked_h = next(s for s in stacked_items if s.item.identifier == "h")
    assert stacked_h.indent_level_with_done_hidden == 0
    assert stacked_h.wrap_parent_with_done_hidden.identifier == "g"


# %% end-to-end wave/track/item wiring


def test_render_wires_an_item_into_its_wave_and_track_sections():
    plan = Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository="owner/repo",
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[Track(id="track-1", name="Track One", wave="wave-1")],
        items=[item("a", ItemStatus.NOT_STARTED)],
    )
    renderer = DashboardRenderer(
        plan=plan, roadmap_text="", pull_requests_by_repository={}, tracking_url=None
    )
    output, _ = renderer.render()
    assert "Wave One" in output
    assert "Track One" in output
    assert 'id="wave-wave-1"' in output


def test_render_shows_placeholder_for_a_track_with_no_items():
    plan = Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository="owner/repo",
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[
            Track(
                id="track-1",
                name="Empty Track",
                wave="wave-1",
                description="Nothing here yet.",
            )
        ],
        items=[],
    )
    renderer = DashboardRenderer(
        plan=plan, roadmap_text="", pull_requests_by_repository={}, tracking_url=None
    )
    output, _ = renderer.render()
    assert "Nothing here yet." in output


def test_render_shows_pull_request_link_when_item_has_one():
    pull_requests_by_repository = {
        "owner/repo": {"5": PullRequestRecord(state=PullRequestState.OPEN, draft=False)}
    }
    plan = Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository="owner/repo",
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[Track(id="track-1", name="Track One", wave="wave-1")],
        items=[item("a", ItemStatus.IN_PROGRESS, pull_request_number=5)],
    )
    renderer = DashboardRenderer(
        plan=plan,
        roadmap_text="",
        pull_requests_by_repository=pull_requests_by_repository,
        tracking_url=None,
    )
    output, _ = renderer.render()
    assert 'href="https://github.com/owner/repo/pull/5"' in output
    assert "#5" in output


def test_render_shows_start_now_button_for_a_not_started_item():
    plan = Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository="owner/repo",
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[Track(id="track-1", name="Track One", wave="wave-1")],
        items=[item("a", ItemStatus.NOT_STARTED)],
    )
    renderer = DashboardRenderer(
        plan=plan, roadmap_text="", pull_requests_by_repository={}, tracking_url=None
    )
    output, _ = renderer.render()
    assert 'data-action-command="/plan-item-kickoff test-plan a"' in output
    assert "Start now" in output


def test_render_shows_resolve_resume_reconsider_buttons_for_underway_items():
    plan = Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository="owner/repo",
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[Track(id="track-1", name="Track One", wave="wave-1")],
        items=[
            item("a", ItemStatus.IN_PROGRESS),
            item("b", ItemStatus.BLOCKED),
            item("c", ItemStatus.DEFERRED),
        ],
    )
    renderer = DashboardRenderer(
        plan=plan, roadmap_text="", pull_requests_by_repository={}, tracking_url=None
    )
    output, _ = renderer.render()
    assert 'data-action-command="/plan-item-resolve test-plan a"' in output
    assert 'data-action-command="/plan-item-resolve test-plan b"' in output
    assert 'data-action-command="/plan-item-resolve test-plan c"' in output
    assert "Resume" in output
    assert "Resolve" in output
    assert "Reconsider" in output


def test_render_shows_review_button_for_an_item_with_a_draft_pull_request():
    pull_requests_by_repository = {
        "owner/repo": {"5": PullRequestRecord(state=PullRequestState.OPEN, draft=True)}
    }
    plan = Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository="owner/repo",
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[Track(id="track-1", name="Track One", wave="wave-1")],
        items=[item("a", ItemStatus.IN_PROGRESS, pull_request_number=5)],
    )
    renderer = DashboardRenderer(
        plan=plan,
        roadmap_text="",
        pull_requests_by_repository=pull_requests_by_repository,
        tracking_url=None,
    )
    output, _ = renderer.render()
    assert 'class="review-button" href="https://github.com/owner/repo/pull/5"' in output
    assert "Review" in output


def test_render_omits_review_button_once_pull_request_is_ready_for_review():
    pull_requests_by_repository = {
        "owner/repo": {"5": PullRequestRecord(state=PullRequestState.OPEN, draft=False)}
    }
    plan = Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository="owner/repo",
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[Track(id="track-1", name="Track One", wave="wave-1")],
        items=[item("a", ItemStatus.IN_PROGRESS, pull_request_number=5)],
    )
    renderer = DashboardRenderer(
        plan=plan,
        roadmap_text="",
        pull_requests_by_repository=pull_requests_by_repository,
        tracking_url=None,
    )
    output, _ = renderer.render()
    assert 'class="review-button"' not in output


def test_render_omits_review_button_for_a_deferred_item_with_a_draft_pull_request():
    pull_requests_by_repository = {
        "owner/repo": {"5": PullRequestRecord(state=PullRequestState.OPEN, draft=True)}
    }
    plan = Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository="owner/repo",
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[Track(id="track-1", name="Track One", wave="wave-1")],
        items=[item("a", ItemStatus.DEFERRED, pull_request_number=5)],
    )
    renderer = DashboardRenderer(
        plan=plan,
        roadmap_text="",
        pull_requests_by_repository=pull_requests_by_repository,
        tracking_url=None,
    )
    output, _ = renderer.render()
    assert 'class="review-button"' not in output


def test_render_shows_ready_to_review_sidebar_section():
    pull_requests_by_repository = {
        "owner/repo": {"5": PullRequestRecord(state=PullRequestState.OPEN, draft=True)}
    }
    plan = Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository="owner/repo",
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[Track(id="track-1", name="Track One", wave="wave-1")],
        items=[item("a", ItemStatus.IN_PROGRESS, pull_request_number=5)],
    )
    renderer = DashboardRenderer(
        plan=plan,
        roadmap_text="",
        pull_requests_by_repository=pull_requests_by_repository,
        tracking_url=None,
    )
    output, _ = renderer.render()
    assert '<div class="next-group next-review">' in output
    assert (
        'class="next-review-link" href="https://github.com/owner/repo/pull/5"' in output
    )


def test_render_shows_ready_to_review_section_last_in_the_sidebar():
    pull_requests_by_repository = {
        "owner/repo": {"5": PullRequestRecord(state=PullRequestState.OPEN, draft=True)}
    }
    plan = Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository="owner/repo",
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[Track(id="track-1", name="Track One", wave="wave-1")],
        items=[
            item("a", ItemStatus.IN_PROGRESS, pull_request_number=5),
            item("b", ItemStatus.DONE),
            item("c", ItemStatus.NOT_STARTED, depends_on=["b"]),
        ],
    )
    renderer = DashboardRenderer(
        plan=plan,
        roadmap_text="",
        pull_requests_by_repository=pull_requests_by_repository,
        tracking_url=None,
    )
    output, _ = renderer.render()
    assert output.index("Ready to review") > output.index("Ready to start")


def test_render_omits_action_button_for_a_done_item():
    plan = Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository="owner/repo",
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[Track(id="track-1", name="Track One", wave="wave-1")],
        items=[item("a", ItemStatus.DONE)],
    )
    renderer = DashboardRenderer(
        plan=plan, roadmap_text="", pull_requests_by_repository={}, tracking_url=None
    )
    output, _ = renderer.render()
    assert 'data-action-command="' not in output


def test_render_hides_done_items_by_default_with_a_sidebar_toggle():
    plan = Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository="owner/repo",
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[Track(id="track-1", name="Track One", wave="wave-1")],
        items=[item("a", ItemStatus.DONE)],
    )
    renderer = DashboardRenderer(
        plan=plan, roadmap_text="", pull_requests_by_repository={}, tracking_url=None
    )
    output, _ = renderer.render()
    assert 'id="plan-dashboard-page"' in output
    assert 'class="page hide-done"' in output
    assert 'id="show-done-toggle"' in output


def test_render_offers_every_model_option_in_each_action_buttons_dropdown():
    plan = Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository="owner/repo",
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[Track(id="track-1", name="Track One", wave="wave-1")],
        items=[item("a", ItemStatus.NOT_STARTED)],
    )
    renderer = DashboardRenderer(
        plan=plan, roadmap_text="", pull_requests_by_repository={}, tracking_url=None
    )
    output, _ = renderer.render()
    for model in AVAILABLE_MODELS:
        assert f'data-value="{model.value}"' in output
        assert f">{model.label}</li>" in output
    assert 'class="model-picker-toggle"' in output
    assert 'class="model-select"' not in output


def test_render_exposes_both_indent_levels_as_css_variables_on_the_item():
    plan = Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository="owner/repo",
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[Track(id="track-1", name="Track One", wave="wave-1")],
        items=[
            item("a", ItemStatus.DONE),
            item("b", ItemStatus.NOT_STARTED, depends_on=["a"]),
        ],
    )
    renderer = DashboardRenderer(
        plan=plan, roadmap_text="", pull_requests_by_repository={}, tracking_url=None
    )
    output, _ = renderer.render()
    assert "--indent-level: 1; --indent-level-hidden-done: 0;" in output


def test_render_shows_dependency_chip_with_dependency_title_as_tooltip():
    plan = Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository="owner/repo",
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[Track(id="track-1", name="Track One", wave="wave-1")],
        items=[
            Item(
                title="Item A",
                branch="a",
                track="track-1",
                status=ItemStatus.DONE,
                id="a",
            ),
            Item(
                title="Item B",
                branch="b",
                track="track-1",
                status=ItemStatus.NOT_STARTED,
                id="b",
                depends_on=["a"],
            ),
        ],
    )
    renderer = DashboardRenderer(
        plan=plan, roadmap_text="", pull_requests_by_repository={}, tracking_url=None
    )
    output, _ = renderer.render()
    assert 'title="Item A"' in output


def test_dependency_chip_falls_back_to_the_raw_identifier_when_unresolved():
    renderer = make_renderer([item("a", ItemStatus.NOT_STARTED, depends_on=["ghost"])])
    renderer.render()
    chip = renderer.plan.items[0].dependency_chips[0]
    assert chip.identifier == "ghost"
    assert chip.tooltip == "ghost"


def test_dependency_chip_is_not_ready_when_the_dependency_has_not_started():
    renderer = make_renderer(
        [
            item("a", ItemStatus.NOT_STARTED),
            item("b", ItemStatus.NOT_STARTED, depends_on=["a"]),
        ]
    )
    renderer.render()
    chip = renderer.items_by_identifier["b"].dependency_chips[0]
    assert chip.is_ready is False


def test_dependency_chip_is_ready_when_the_dependency_is_done():
    renderer = make_renderer(
        [
            item("a", ItemStatus.DONE),
            item("b", ItemStatus.NOT_STARTED, depends_on=["a"]),
        ]
    )
    renderer.render()
    chip = renderer.items_by_identifier["b"].dependency_chips[0]
    assert chip.is_ready is True


def test_dependency_chip_is_not_ready_when_unresolved():
    renderer = make_renderer([item("a", ItemStatus.NOT_STARTED, depends_on=["ghost"])])
    renderer.render()
    chip = renderer.plan.items[0].dependency_chips[0]
    assert chip.is_ready is False


def test_render_marks_an_unmet_dependency_chip_with_the_chip_unmet_class():
    plan = Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository="owner/repo",
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[Track(id="track-1", name="Track One", wave="wave-1")],
        items=[
            item("a", ItemStatus.NOT_STARTED),
            item("b", ItemStatus.NOT_STARTED, depends_on=["a"]),
        ],
    )
    renderer = DashboardRenderer(
        plan=plan, roadmap_text="", pull_requests_by_repository={}, tracking_url=None
    )
    output, _ = renderer.render()
    assert 'class="chip chip-unmet"' in output


def test_render_does_not_mark_a_ready_dependency_chip_as_unmet():
    plan = Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository="owner/repo",
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[Track(id="track-1", name="Track One", wave="wave-1")],
        items=[
            item("a", ItemStatus.DONE),
            item("b", ItemStatus.NOT_STARTED, depends_on=["a"]),
        ],
    )
    renderer = DashboardRenderer(
        plan=plan, roadmap_text="", pull_requests_by_repository={}, tracking_url=None
    )
    output, _ = renderer.render()
    assert 'class="chip"' in output
    assert 'class="chip chip-unmet"' not in output


# %% sidebar next-step links


def test_render_gives_each_item_card_a_stable_id_anchor():
    plan = Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository="owner/repo",
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[Track(id="track-1", name="Track One", wave="wave-1")],
        items=[item("a", ItemStatus.NOT_STARTED)],
    )
    renderer = DashboardRenderer(
        plan=plan, roadmap_text="", pull_requests_by_repository={}, tracking_url=None
    )
    output, _ = renderer.render()
    assert 'id="item-a"' in output


def test_render_links_a_ready_to_start_sidebar_entry_to_its_item_card():
    plan = Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository="owner/repo",
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[Track(id="track-1", name="Track One", wave="wave-1")],
        items=[
            item("a", ItemStatus.DONE),
            item("b", ItemStatus.NOT_STARTED, depends_on=["a"]),
        ],
    )
    renderer = DashboardRenderer(
        plan=plan, roadmap_text="", pull_requests_by_repository={}, tracking_url=None
    )
    output, _ = renderer.render()
    assert 'href="#item-b"' in output
    assert 'data-item-identifier="b"' in output
    assert 'onclick="planDashboardHighlightItem(event, this)"' in output


def test_render_links_a_drift_sidebar_entry_to_its_item_card():
    pull_requests_by_repository = {
        "owner/repo": {"1": PullRequestRecord(state=PullRequestState.OPEN, draft=False)}
    }
    plan = Plan(
        id="test-plan",
        title="Test Plan",
        description="desc",
        default_repository="owner/repo",
        waves=[Wave(id="wave-1", name="Wave One")],
        tracks=[Track(id="track-1", name="Track One", wave="wave-1")],
        items=[item("a", ItemStatus.DONE, pull_request_number=1)],
    )
    renderer = DashboardRenderer(
        plan=plan,
        roadmap_text="",
        pull_requests_by_repository=pull_requests_by_repository,
        tracking_url=None,
    )
    output, _ = renderer.render()
    assert 'href="#item-a"' in output
    assert 'data-item-identifier="a"' in output


# %% status counts


def test_status_counts_cover_every_status_even_when_zero():
    renderer = make_renderer([item("a", ItemStatus.DONE)])
    _, summary = renderer.render()
    assert summary.status_counts[ItemStatus.DONE] == 1
    assert summary.status_counts[ItemStatus.BLOCKED] == 0


def test_summary_to_json_dict_uses_plain_string_status_keys():
    renderer = make_renderer([item("a", ItemStatus.DONE)])
    _, summary = renderer.render()
    json_dict = summary.to_json_dict()
    assert json_dict["counts"]["done"] == 1
    assert json_dict["drift_count"] == 0


# %% main


def _write_minimal_plan_files(directory: Path) -> tuple[Path, Path, Path, Path]:
    """
    Write a well-formed plan.yaml/roadmap.md/pr_data.json trio to *directory*, for a
    ``main()`` end-to-end test.

    :param directory: Where to write the files.
    :return:``(plan_path, roadmap_path, pull_request_data_path, output_path)``.
    """
    plan_path = directory / "plan.yaml"
    plan_path.write_text(yaml.dump(minimal_plan()))
    roadmap_path = directory / "roadmap.md"
    roadmap_path.write_text("# Roadmap\n")
    pull_request_data_path = directory / "pr_data.json"
    pull_request_data_path.write_text("{}")
    output_path = directory / "dashboard.html"
    return plan_path, roadmap_path, pull_request_data_path, output_path


def test_main_renders_the_dashboard_and_prints_the_summary(
    tmp_path, monkeypatch, capsys
):
    plan_path, roadmap_path, pull_request_data_path, output_path = (
        _write_minimal_plan_files(tmp_path)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_dashboard.py",
            "--plan",
            str(plan_path),
            "--roadmap",
            str(roadmap_path),
            "--pr-data",
            str(pull_request_data_path),
            "--output",
            str(output_path),
        ],
    )
    exit_code = main()
    assert exit_code == 0
    assert "<title>Test Plan</title>" in output_path.read_text()


def test_main_prints_the_status_summary_as_json(tmp_path, monkeypatch, capsys):
    plan_path, roadmap_path, pull_request_data_path, output_path = (
        _write_minimal_plan_files(tmp_path)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_dashboard.py",
            "--plan",
            str(plan_path),
            "--roadmap",
            str(roadmap_path),
            "--pr-data",
            str(pull_request_data_path),
            "--output",
            str(output_path),
        ],
    )
    main()
    summary = json.loads(capsys.readouterr().out)
    assert summary["counts"]["not_started"] == 1


def test_main_rejects_an_invalid_manifest_instead_of_crashing(
    tmp_path, monkeypatch, capsys
):
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("")  # yaml.safe_load("") -> None, not a mapping
    roadmap_path = tmp_path / "roadmap.md"
    roadmap_path.write_text("")
    pull_request_data_path = tmp_path / "pr_data.json"
    pull_request_data_path.write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_dashboard.py",
            "--plan",
            str(plan_path),
            "--roadmap",
            str(roadmap_path),
            "--pr-data",
            str(pull_request_data_path),
            "--output",
            str(tmp_path / "dashboard.html"),
        ],
    )
    exit_code = main()
    assert exit_code == 1
    with pytest.raises(PlanValidationError) as expected_error:
        validate_plan(None)
    assert capsys.readouterr().err == (
        f"plan.yaml failed validation: {expected_error.value}\n"
    )


# %% example-walkthrough.md's committed sample plan


def _render_example_plan():
    plan_mapping = yaml.safe_load((EXAMPLE_DIRECTORY / "plan.yaml").read_text())
    validate_plan(plan_mapping)
    plan = Plan.from_mapping(plan_mapping)
    pull_requests_by_repository = load_pull_requests_by_repository(
        json.loads((EXAMPLE_DIRECTORY / "pr_data.json").read_text())
    )
    renderer = DashboardRenderer(
        plan=plan,
        roadmap_text=(EXAMPLE_DIRECTORY / "roadmap.md").read_text(),
        pull_requests_by_repository=pull_requests_by_repository,
        tracking_url=None,
    )
    return renderer.render()


def test_example_plan_passes_the_same_validation_plan_create_must_satisfy():
    plan_mapping = yaml.safe_load((EXAMPLE_DIRECTORY / "plan.yaml").read_text())
    validate_plan(plan_mapping)  # raises PlanValidationError on any problem


def test_example_plan_renders_the_counts_and_sections_the_walkthrough_describes():
    """Locks example-walkthrough.md's screenshots and prose to the actual
    schema/logic - a future change to either would fail this test instead of
    silently leaving the doc showing stale numbers."""
    _, summary = _render_example_plan()
    assert summary.status_counts[ItemStatus.NOT_STARTED] == 1
    assert summary.status_counts[ItemStatus.IN_PROGRESS] == 2
    assert summary.status_counts[ItemStatus.BLOCKED] == 1
    assert summary.status_counts[ItemStatus.DONE] == 2
    assert summary.drift_items == ["Dead-letter queue for exhausted retries"]
    assert summary.ready_to_start == ["Retry metrics dashboard"]
    assert summary.blocker_maybe_cleared == [
        "Load-test the retry path under failure injection"
    ]
    assert summary.ready_to_review == [
        "Circuit breaker around the retry loop",
        "Feature flag for the new retry behavior",
    ]


def test_example_plan_demonstrates_the_bug_chip_and_its_filter():
    """The walkthrough's screenshots show both, so the sample data has to keep
    producing them - and the chip must not disturb the grouping the test above
    pins."""
    output, _ = _render_example_plan()
    assert output.count('<span class="next-bug-chip">bug</span>') == 1
    assert 'id="bug-fixes-only-toggle"' in output
    assert '<div class="next-group next-drift next-group-has-bugs">' in output
