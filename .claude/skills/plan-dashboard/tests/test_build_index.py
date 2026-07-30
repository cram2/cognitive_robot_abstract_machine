"""
Tests for build_index.py's master-index rendering.
"""

import json
import sys
from typing import Any

from build_index import PlanSummary, main, render_index_page


def summary(**overrides: Any) -> PlanSummary:
    """
    Build one :class:`PlanSummary` for a test, with sensible defaults for every field.

    :param overrides: Fields to replace on the returned summary.
    """
    fields = {
        "id": "plan-a",
        "title": "Plan A",
        "description": "desc",
        "done": 1,
        "total": 4,
        "dashboard_url": None,
    }
    fields.update(overrides)
    return PlanSummary(**fields)


# %% PlanSummary


def test_is_complete_requires_at_least_one_item():
    assert not summary(done=0, total=0).is_complete


def test_is_complete_when_done_equals_total():
    assert summary(done=4, total=4).is_complete


def test_completion_percentage_of_empty_plan_is_zero():
    assert summary(done=0, total=0).completion_percentage == 0.0


def test_completion_percentage_is_a_fraction_of_total():
    assert summary(done=1, total=4).completion_percentage == 25.0


def test_progress_label_for_empty_plan():
    assert summary(done=0, total=0).progress_label == "no items yet"


def test_progress_label_shows_done_over_total():
    assert summary(done=1, total=4).progress_label == "1 / 4 done"


def test_from_mapping_defaults_missing_description_to_empty_string():
    plan = PlanSummary.from_mapping({"id": "x", "title": "X", "done": 0, "total": 0})
    assert plan.description == ""
    assert plan.dashboard_url is None


def test_from_mapping_keeps_an_http_dashboard_url():
    plan = PlanSummary.from_mapping(
        {
            "id": "x",
            "title": "X",
            "done": 0,
            "total": 0,
            "dashboard_url": "https://claude.ai/code/artifact/abc",
        }
    )
    assert plan.dashboard_url == "https://claude.ai/code/artifact/abc"


def test_from_mapping_rejects_a_non_http_dashboard_url():
    plan = PlanSummary.from_mapping(
        {
            "id": "x",
            "title": "X",
            "done": 0,
            "total": 0,
            "dashboard_url": "javascript:alert(1)",
        }
    )
    assert plan.dashboard_url is None


def test_completion_percentage_label_is_formatted_to_one_decimal_place():
    assert summary(done=1, total=3).completion_percentage_label == "33.3%"


def test_css_class_is_plain_for_an_incomplete_plan():
    assert summary(done=1, total=4).css_class == "plan-card"


def test_css_class_marks_a_complete_plan():
    assert summary(done=4, total=4).css_class == "plan-card complete"


# %% render_index_page - per-card rendering


def test_card_links_to_dashboard_when_published():
    rendered = render_index_page(
        [summary(dashboard_url="https://claude.ai/code/artifact/abc")]
    )
    assert '<a class="plan-card' in rendered
    assert 'href="https://claude.ai/code/artifact/abc"' in rendered
    assert "Not published yet" not in rendered


def test_card_shows_unpublished_notice_when_no_dashboard_url():
    rendered = render_index_page([summary(dashboard_url=None)])
    assert '<div class="plan-card' in rendered
    assert "Not published yet" in rendered


def test_card_marks_complete_plans():
    rendered = render_index_page([summary(done=4, total=4)])
    assert "plan-card complete" in rendered


# %% render_index_page - full page


def test_index_page_shows_placeholder_when_no_plans():
    rendered = render_index_page([])
    assert "No plans found" in rendered


def test_index_page_renders_every_plan_card():
    rendered = render_index_page(
        [summary(id="a", title="Plan A"), summary(id="b", title="Plan B")]
    )
    assert "Plan A" in rendered
    assert "Plan B" in rendered


# %% main


def test_main_renders_every_plan_from_the_plans_json_file(tmp_path, monkeypatch):
    plans_path = tmp_path / "plans.json"
    plans_path.write_text(
        json.dumps(
            [
                {
                    "id": "a",
                    "title": "Plan A",
                    "description": "desc",
                    "done": 1,
                    "total": 2,
                    "dashboard_url": "https://claude.ai/code/artifact/abc",
                }
            ]
        )
    )
    output_path = tmp_path / "index.html"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_index.py",
            "--plans",
            str(plans_path),
            "--output",
            str(output_path),
        ],
    )
    exit_code = main()
    assert exit_code == 0
    assert "Plan A" in output_path.read_text()
