"""
Unit tests for the :class:`HasText` / :class:`HasNumber` fragment mixins — the single
definition of the ``text`` and ``number`` fields, shared by the leaf/phrase fragments
instead of being redeclared on each.

These pin the two properties that justify the mixins: (1) the ``text`` / ``number``
fields are contributed by the mixins (one definition, reused), and (2) the migration
preserves the existing construction contract — ``text`` stays positional, ``number`` is
keyword-only — so no call site changes.
"""

from __future__ import annotations

import dataclasses as dc

from krrood.entity_query_language.verbalization.fragments.base import (
    HasNumber,
    HasText,
    NounPhrase,
    RoleFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    GrammaticalNumber,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole


def _field(cls, name) -> dc.Field:
    return {f.name: f for f in dc.fields(cls)}[name]


def test_text_field_is_defined_once_on_hastext():
    """
    ``text`` is contributed by :class:`HasText` — both text-bearing fragments inherit
    it, none redeclares it.
    """
    assert HasText in WordFragment.__mro__
    assert HasText in RoleFragment.__mro__
    # The field exists on the concrete classes but is owned by the mixin, not redefined locally.
    assert "text" not in WordFragment.__dict__.get("__annotations__", {})
    assert "text" not in RoleFragment.__dict__.get("__annotations__", {})
    # A non-text fragment does not pick up HasText.
    assert HasText not in NounPhrase.__mro__


def test_number_field_is_defined_once_on_hasnumber():
    """
    ``number`` is contributed by :class:`HasNumber` — every fragment whose own text the
    morphology pass pluralises inherits it, none redeclares it.
    """
    for cls in (WordFragment, RoleFragment, NounPhrase):
        assert HasNumber in cls.__mro__
        assert "number" not in cls.__dict__.get("__annotations__", {})


def test_number_and_text_are_keyword_only():
    """
    Fragments are built by keyword, so ``text`` and the inherited ``number`` are both
    keyword-only and ``number`` never disturbs the field order.
    """
    assert _field(WordFragment, "text").kw_only is True
    assert _field(RoleFragment, "text").kw_only is True
    for cls in (WordFragment, RoleFragment, NounPhrase):
        assert _field(cls, "number").kw_only is True
        assert _field(cls, "number").default is GrammaticalNumber.SINGULAR


def test_keyword_construction_defaults_the_number():
    """
    A fragment built from its ``text`` alone takes the inherited ``number`` default,
    SINGULAR.
    """
    assert WordFragment(text="the").text == "the"
    assert WordFragment(text="the").number is GrammaticalNumber.SINGULAR
    role = RoleFragment(text="Robot", role=SemanticRole.VARIABLE)
    assert (role.text, role.role, role.number) == (
        "Robot",
        SemanticRole.VARIABLE,
        GrammaticalNumber.SINGULAR,
    )


def test_number_passed_by_keyword():
    """
    ``number=`` keyword construction reaches the inherited field on every fragment.
    """
    assert (
        WordFragment(text="x", number=GrammaticalNumber.PLURAL).number
        is GrammaticalNumber.PLURAL
    )
    assert (
        RoleFragment(
            text="a", role=SemanticRole.ATTRIBUTE, number=GrammaticalNumber.PLURAL
        ).number
        is GrammaticalNumber.PLURAL
    )
    assert (
        NounPhrase(head=WordFragment(text="x"), number=GrammaticalNumber.PLURAL).number
        is GrammaticalNumber.PLURAL
    )
