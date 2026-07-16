"""
Tests for field display-name overrides: a dataclass field may register a
:attr:`GrammarMetadata.display_name` so it verbalizes as a chosen surface word instead
of its raw attribute name (*"beginning"* for a field named ``begin``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from krrood.entity_query_language.factories import an, entity, variable
from krrood.entity_query_language.verbalization.fragments.base import RoleFragment
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
from krrood.patterns.field_metadata import FieldMetadata, GrammarMetadata


@dataclass
class Date:
    """
    A calendar point used as a nested attribute in the period example.
    """

    month: int
    year: int


@dataclass
class Period:
    """
    A span whose ``begin`` field verbalizes as *"beginning"* via a display-name
    override.
    """

    begin: Date = field(
        metadata=FieldMetadata(
            other_metadata=[GrammarMetadata(display_name="beginning")]
        ).as_dict()
    )
    end: Date = field(default=None)


def test_display_name_overrides_attribute_name():
    """
    The registered display name replaces the raw field name in the surface word.
    """
    assert RoleFragment.for_attribute(Period, "begin").text == "beginning"


def test_absent_display_name_keeps_attribute_name():
    """
    A field with no display-name override keeps its attribute name.
    """
    assert RoleFragment.for_attribute(Period, "end").text == "end"


def test_explicit_text_override_wins_over_display_name():
    """
    An explicit *text* argument takes precedence over any registered display name.
    """
    assert RoleFragment.for_attribute(Period, "begin", text="start").text == "start"


def test_display_name_renders_in_a_verbalized_query():
    """
    The override surfaces end-to-end through the pipeline.
    """
    period = variable(Period, domain=None)
    query = an(entity(period).where(period.begin.month == 3))
    assert "beginning" in verbalize_expression(query)
