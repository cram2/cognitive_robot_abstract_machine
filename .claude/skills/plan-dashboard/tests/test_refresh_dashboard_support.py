"""
Tests for refresh_dashboard_support.py: the JSON-plumbing helpers refresh_dashboard.sh
calls between sync_manifest_status.py and build_dashboard.py.
"""

import json
import sys

import pytest

from refresh_dashboard_support import (
    SummaryKeyCollisionError,
    count_corrected,
    main,
    merge_summaries,
)

# %% count_corrected


def test_count_corrected_counts_the_corrected_list():
    summary = json.dumps({"corrected": [{"id": "a"}, {"id": "b"}]})
    assert count_corrected(summary) == 2


def test_count_corrected_zero_when_nothing_corrected():
    summary = json.dumps({"corrected": []})
    assert count_corrected(summary) == 0


# %% merge_summaries


def test_merge_summaries_combines_both_objects():
    sync_summary = json.dumps({"corrected": [{"id": "a"}]})
    build_summary = json.dumps({"status_counts": {"done": 1}, "drift_count": 0})
    merged = merge_summaries(sync_summary, build_summary)
    assert merged == {
        "corrected": [{"id": "a"}],
        "status_counts": {"done": 1},
        "drift_count": 0,
    }


def test_merge_summaries_raises_on_a_shared_key_instead_of_silently_dropping_one():
    sync_summary = json.dumps({"corrected": [{"id": "a"}], "drift_count": 1})
    build_summary = json.dumps({"drift_count": 0})
    with pytest.raises(SummaryKeyCollisionError, match="drift_count"):
        merge_summaries(sync_summary, build_summary)


# %% main - subcommand dispatch


def test_main_dispatches_count_corrected(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refresh_dashboard_support.py",
            "count-corrected",
            json.dumps({"corrected": [{"id": "a"}, {"id": "b"}]}),
        ],
    )
    exit_code = main()
    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "2"


def test_main_dispatches_merge_summaries(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refresh_dashboard_support.py",
            "merge-summaries",
            json.dumps({"corrected": [{"id": "a"}]}),
            json.dumps({"drift_count": 0}),
        ],
    )
    exit_code = main()
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "corrected": [{"id": "a"}],
        "drift_count": 0,
    }
