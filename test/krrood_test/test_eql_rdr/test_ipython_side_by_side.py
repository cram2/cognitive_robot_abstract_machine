"""Unit tests for ``IPythonInterface._case_table`` dispatching."""

from __future__ import annotations

import pytest

from krrood.entity_query_language.rdr.case_table import render_case_table
from krrood.entity_query_language.rdr.interactive import IPythonInterface
from krrood.entity_query_language.rdr.interface import CaseContext
from krrood.entity_query_language.rdr.single_class import EQLSingleClassRDR

from .animal import Animal, Species

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_animal(
    name: str,
    *,
    milk: bool = False,
    feathers: bool = False,
    fins: bool = False,
    backbone: bool = True,
    venomous: bool = False,
) -> Animal:
    """Pattern: DistinctAnimalCase — minimal animal with one discriminating feature.

    Copied verbatim from ``test_corner_case_population.py`` so this file is
    self-contained without a cross-test import.
    """
    return Animal(
        name=name,
        hair=milk,
        feathers=feathers,
        eggs=not milk,
        milk=milk,
        airborne=False,
        aquatic=fins,
        predator=False,
        toothed=backbone,
        backbone=backbone,
        breathes=not fins,
        venomous=venomous,
        fins=fins,
        legs=0 if fins else 4,
        tail=backbone,
        domestic=False,
        catsize=milk,
    )


def _make_context(
    rdr: EQLSingleClassRDR, case: Animal, *, corner_case: object = None
) -> CaseContext:
    """Build a ``CaseContext`` for ``case`` with an optional ``corner_case``."""
    return CaseContext(
        case_instance=case,
        case_variable=rdr.case_variable,
        corner_case=corner_case,
    )


# ---------------------------------------------------------------------------
# Test 1 — no corner case => output matches standalone render_case_table
# ---------------------------------------------------------------------------


def test_ipython_case_table_without_corner_case_uses_single_render():
    """With ``corner_case=None``, ``_case_table`` output equals ``render_case_table``
    output for the same case instance (no side-by-side wrapping applied)."""
    rdr = EQLSingleClassRDR(Animal, "species")
    case = _make_animal("mammal", milk=True)
    interface = IPythonInterface(use_color=False)

    context = _make_context(rdr, case, corner_case=None)

    result = interface._case_table(context)
    expected = render_case_table(case, use_color=False)

    assert result == expected


# ---------------------------------------------------------------------------
# Test 2 — corner_case set => output contains both labels
# ---------------------------------------------------------------------------


def test_ipython_case_table_with_corner_case_contains_both_labels():
    """With ``context.corner_case`` populated, ``_case_table`` output contains both
    the new-case label and the corner-case label (side-by-side render was used)."""
    rdr = EQLSingleClassRDR(Animal, "species")
    new_case = _make_animal("eagle", feathers=True)
    corner_case = _make_animal("mammal", milk=True)
    interface = IPythonInterface(use_color=False)

    context = _make_context(rdr, new_case, corner_case=corner_case)

    result = interface._case_table(context)

    # The exact default labels from render_cases_side_by_side must appear.
    assert "New case" in result
    assert "Corner case" in result
