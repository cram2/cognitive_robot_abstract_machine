"""
Tests for sync_manifest_status.py: auto-correcting a plan.yaml's item statuses to "done"
wherever GitHub confirms the item's pull request is merged.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from build_dashboard import (
    ItemStatus,
    PlanValidationError,
    PullRequestRecord,
    PullRequestState,
    validate_plan,
)
from sync_manifest_status import (
    MissingStatusLineError,
    apply_status_corrections,
    find_items_to_correct,
    main,
)


def plan(**overrides: Any) -> dict[str, Any]:
    """
    Build one raw, plan.yaml-shaped ``dict`` for a test - the same shape
    ``yaml.safe_load`` would produce, since ``find_items_to_correct``
    operates directly on that raw structure, never on a parsed :class:`Plan`.
    """
    data = {
        "schema_version": 1,
        "id": "test-plan",
        "title": "Test Plan",
        "description": "A plan.",
        "default_repository": "owner/repo",
        "waves": [{"id": "wave-1", "name": "Wave 1"}],
        "tracks": [{"id": "track-1", "name": "Track 1", "wave": "wave-1"}],
        "items": [],
    }
    data.update(overrides)
    return data


def item(
    identifier: str,
    status: str,
    pull_request_number: int | None = None,
    repository: str | None = None,
) -> dict[str, Any]:
    """
    Build one raw, plan.yaml-shaped item ``dict`` for a test.

    ``status`` is a plain ``str``, not :class:`ItemStatus`: this mirrors exactly what
    ``yaml.safe_load`` hands back before any parsing into typed dataclasses happens,
    since ``find_items_to_correct`` and ``apply_status_corrections`` both work directly
    on that raw structure.
    """
    entry = {
        "id": identifier,
        "title": identifier,
        "branch": identifier,
        "track": "track-1",
        "status": status,
        "pull_request_number": pull_request_number,
    }
    if repository is not None:
        entry["repository"] = repository
    return entry


# %% find_items_to_correct


def test_finds_an_in_progress_item_whose_pull_request_is_merged():
    pull_requests_by_repository = {
        "owner/repo": {
            "1": PullRequestRecord(
                state=PullRequestState.CLOSED, merged_at=datetime(2026, 1, 1)
            )
        }
    }
    items = [item("a", "in_progress", pull_request_number=1)]
    corrections = find_items_to_correct(plan(items=items), pull_requests_by_repository)
    assert [entry["id"] for entry in corrections] == ["a"]


def test_ignores_an_item_already_marked_done():
    pull_requests_by_repository = {
        "owner/repo": {
            "1": PullRequestRecord(
                state=PullRequestState.CLOSED, merged_at=datetime(2026, 1, 1)
            )
        }
    }
    items = [item("a", "done", pull_request_number=1)]
    assert find_items_to_correct(plan(items=items), pull_requests_by_repository) == []


def test_ignores_an_item_whose_pull_request_is_still_open():
    pull_requests_by_repository = {
        "owner/repo": {"1": PullRequestRecord(state=PullRequestState.OPEN, draft=False)}
    }
    items = [item("a", "in_progress", pull_request_number=1)]
    assert find_items_to_correct(plan(items=items), pull_requests_by_repository) == []


def test_ignores_an_item_with_no_pull_request_yet():
    items = [item("a", "not_started")]
    assert find_items_to_correct(plan(items=items), {}) == []


def test_merged_via_out_of_band_label_is_also_corrected():
    pull_requests_by_repository = {
        "owner/repo": {
            "1": PullRequestRecord(state=PullRequestState.CLOSED, labels=["merged"])
        }
    }
    items = [item("a", "blocked", pull_request_number=1)]
    corrections = find_items_to_correct(plan(items=items), pull_requests_by_repository)
    assert [entry["id"] for entry in corrections] == ["a"]


def test_uses_the_item_repository_override_over_the_plan_default():
    pull_requests_by_repository = {
        "owner/other-repo": {
            "1": PullRequestRecord(
                state=PullRequestState.CLOSED, merged_at=datetime(2026, 1, 1)
            )
        }
    }
    items = [
        item("a", "in_progress", pull_request_number=1, repository="owner/other-repo")
    ]
    corrections = find_items_to_correct(plan(items=items), pull_requests_by_repository)
    assert [entry["id"] for entry in corrections] == ["a"]


# %% apply_status_corrections - real manifest text

# A real plan.yaml's raw text (two items, one due for correction) - loaded
# once and shared read-only by every test below, since apply_status_corrections
# patches text in place and each test asserts against its own fresh copy of
# the return value rather than mutating this constant.
MANIFEST_TEXT = (Path(__file__).parent / "fixtures" / "manifest.yaml").read_text()


def test_patches_only_the_targeted_items_status_line():
    data = yaml.safe_load(MANIFEST_TEXT)
    patched_text, corrections = apply_status_corrections(
        MANIFEST_TEXT, [data["items"][0]]
    )
    assert "    status: done" in patched_text
    assert "    status: not_started" in patched_text  # item b untouched
    assert [c.item_identifier for c in corrections] == ["a"]
    assert [c.previous_status for c in corrections] == [ItemStatus.IN_PROGRESS]


def test_patching_leaves_every_other_line_byte_for_byte_identical():
    data = yaml.safe_load(MANIFEST_TEXT)
    patched_text, _ = apply_status_corrections(MANIFEST_TEXT, [data["items"][0]])
    original_lines = MANIFEST_TEXT.split("\n")
    patched_lines = patched_text.split("\n")
    changed_line_pairs = [
        (before, after)
        for before, after in zip(original_lines, patched_lines)
        if before != after
    ]
    assert changed_line_pairs == [("    status: in_progress", "    status: done")]


def test_patched_text_still_parses_and_validates():
    data = yaml.safe_load(MANIFEST_TEXT)
    patched_text, _ = apply_status_corrections(MANIFEST_TEXT, [data["items"][0]])
    reparsed = yaml.safe_load(patched_text)
    assert reparsed["items"][0]["status"] == "done"
    assert reparsed["items"][0]["notes"] == data["items"][0]["notes"]


def test_no_items_to_correct_returns_original_text_unchanged():
    patched_text, corrections = apply_status_corrections(MANIFEST_TEXT, [])
    assert patched_text == MANIFEST_TEXT
    assert corrections == []


def test_raises_if_an_item_has_no_status_line():
    text = "- id: a\n  title: A\n  branch: a\n"
    with pytest.raises(MissingStatusLineError, match="no status: line"):
        apply_status_corrections(text, [{"id": "a", "branch": "a"}])


# %% main - manifest validation


def test_main_rejects_an_invalid_manifest_instead_of_crashing(
    tmp_path, monkeypatch, capsys
):
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("")  # yaml.safe_load("") -> None, not a mapping
    pr_data_path = tmp_path / "pr_data.json"
    pr_data_path.write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sync_manifest_status.py",
            "--plan",
            str(plan_path),
            "--pr-data",
            str(pr_data_path),
        ],
    )
    exit_code = main()
    assert exit_code == 1
    with pytest.raises(PlanValidationError) as expected_error:
        validate_plan(None)
    assert capsys.readouterr().err == (
        f"plan.yaml failed validation: {expected_error.value}\n"
    )


# %% main - file read/write round trip


def _write_merged_pull_request_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """
    Write the shared ``MANIFEST_TEXT`` fixture (item "a" in progress, pull request #1)
    plus a ``pr_data.json`` reporting #1 merged, to *tmp_path*.

    :param tmp_path: pytest's per-test temporary directory.
    :return:``(plan_path, pr_data_path)``.
    """
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(MANIFEST_TEXT)
    pr_data_path = tmp_path / "pr_data.json"
    pr_data_path.write_text(
        json.dumps(
            {
                "owner/repo": {
                    "1": {
                        "state": "closed",
                        "draft": False,
                        "merged_at": "2026-01-01T00:00:00+00:00",
                        "labels": [],
                    }
                }
            }
        )
    )
    return plan_path, pr_data_path


def test_main_corrects_the_plan_file_in_place_when_no_output_is_given(
    tmp_path, monkeypatch, capsys
):
    plan_path, pr_data_path = _write_merged_pull_request_fixture(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sync_manifest_status.py",
            "--plan",
            str(plan_path),
            "--pr-data",
            str(pr_data_path),
        ],
    )
    exit_code = main()
    assert exit_code == 0
    assert "    status: done" in plan_path.read_text()
    summary = json.loads(capsys.readouterr().out)
    assert summary == {"corrected": [{"id": "a", "previous_status": "in_progress"}]}


def test_main_writes_the_corrected_manifest_to_output_leaving_the_plan_file_untouched(
    tmp_path, monkeypatch
):
    plan_path, pr_data_path = _write_merged_pull_request_fixture(tmp_path)
    output_path = tmp_path / "corrected-plan.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sync_manifest_status.py",
            "--plan",
            str(plan_path),
            "--pr-data",
            str(pr_data_path),
            "--output",
            str(output_path),
        ],
    )
    exit_code = main()
    assert exit_code == 0
    assert plan_path.read_text() == MANIFEST_TEXT
    assert "    status: done" in output_path.read_text()
