"""Unit tests for ``render_cases_side_by_side``."""

from __future__ import annotations

import dataclasses

import pytest

from krrood.entity_query_language.rdr.case_table import render_cases_side_by_side

from .animal import Animal, Species

# ---------------------------------------------------------------------------
# Minimal case fixtures — PatternName: DistinctFieldAnimal
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DistinctFieldAnimal:
    """A minimal Animal-like dataclass with two fields that are unique per instance.

    Used so tests can assert the *new* case's value and the *corner* case's value
    appear independently in the rendered output without guessing field positions.
    """

    name: str
    has_wings: bool
    leg_count: int


_NEW_CASE = DistinctFieldAnimal(name="eagle", has_wings=True, leg_count=2)
_CORNER_CASE = DistinctFieldAnimal(name="scorpion", has_wings=False, leg_count=8)


# ---------------------------------------------------------------------------
# Test 1 — custom new_label appears in the output
# ---------------------------------------------------------------------------


def test_render_cases_side_by_side_contains_new_label():
    """``render_cases_side_by_side`` emits the ``new_label`` string in its output."""
    result = render_cases_side_by_side(
        _NEW_CASE,
        _CORNER_CASE,
        new_label="My New Case",
        corner_label="My Corner Case",
        use_color=False,
    )
    assert "My New Case" in result


# ---------------------------------------------------------------------------
# Test 2 — custom corner_label appears in the output
# ---------------------------------------------------------------------------


def test_render_cases_side_by_side_contains_corner_label():
    """``render_cases_side_by_side`` emits the ``corner_label`` string in its output."""
    result = render_cases_side_by_side(
        _NEW_CASE,
        _CORNER_CASE,
        new_label="My New Case",
        corner_label="My Corner Case",
        use_color=False,
    )
    assert "My Corner Case" in result


# ---------------------------------------------------------------------------
# Test 3 — field values unique to each case both appear
# ---------------------------------------------------------------------------


def test_render_cases_side_by_side_contains_both_case_values():
    """Both a new-case-unique value and a corner-case-unique value appear in the output.

    ``_NEW_CASE.name == "eagle"`` and ``_CORNER_CASE.name == "scorpion"`` are distinct
    strings that cannot belong to the wrong side.
    """
    result = render_cases_side_by_side(
        _NEW_CASE,
        _CORNER_CASE,
        use_color=False,
    )
    # "eagle" is a field value present only in _NEW_CASE
    assert "eagle" in result
    # "scorpion" is a field value present only in _CORNER_CASE
    assert "scorpion" in result


# ---------------------------------------------------------------------------
# Test 4 — default labels ("New case" / "Corner case") appear when omitted
# ---------------------------------------------------------------------------


def test_render_cases_side_by_side_default_labels():
    """When called without label kwargs the output contains the default label strings."""
    result = render_cases_side_by_side(
        _NEW_CASE,
        _CORNER_CASE,
        use_color=False,
    )
    assert "New case" in result
    assert "Corner case" in result
