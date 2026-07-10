"""
Unit tests for the morphology realisation pass — the single place grammatical number is
*applied*.  Assemblers tag a leaf ``GrammaticalNumber.PLURAL``; this pass pluralises it.
"""

from __future__ import annotations

from krrood.entity_query_language.verbalization.fragments.base import (
    flatten_fragment_to_plain_text,
    PhraseFragment,
    RoleFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    GrammaticalNumber,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.rendering.morphology_processor import (
    MorphologyProcessor,
)
from krrood.entity_query_language.verbalization.rendering.realization import (
    realize_subtree,
)


def _text(fragment) -> str:
    return flatten_fragment_to_plain_text(fragment)


def test_pass_pluralises_only_plural_tagged_leaves():
    tree = PhraseFragment(
        parts=[
            RoleFragment(
                text="Robot",
                role=SemanticRole.VARIABLE,
                number=GrammaticalNumber.PLURAL,
            ),
            WordFragment(text="of"),
            RoleFragment(text="Cabinet", role=SemanticRole.VARIABLE),  # singular
        ]
    )
    assert _text(MorphologyProcessor().process(tree)) == "Robots of Cabinet"


def test_pass_leaves_singular_untouched():
    leaf = RoleFragment(text="Robot", role=SemanticRole.VARIABLE)
    assert _text(MorphologyProcessor().process(leaf)) == "Robot"


def test_pass_is_idempotent():
    leaf = RoleFragment(
        text="task", role=SemanticRole.ATTRIBUTE, number=GrammaticalNumber.PLURAL
    )
    once = MorphologyProcessor().process(leaf)
    twice = MorphologyProcessor().process(once)
    assert _text(twice) == "tasks"
    assert once.number is GrammaticalNumber.SINGULAR  # reset after inflection


def test_pass_preserves_block_structure():
    block = PhraseFragment(
        parts=[
            WordFragment(text="the"),
            RoleFragment(
                text="container",
                role=SemanticRole.ATTRIBUTE,
                number=GrammaticalNumber.PLURAL,
            ),
        ]
    )
    assert _text(MorphologyProcessor().process(block)) == "the containers"


def test_realize_subtree_runs_pass_then_flattens():
    leaf = RoleFragment(
        text="task", role=SemanticRole.ATTRIBUTE, number=GrammaticalNumber.PLURAL
    )
    assert realize_subtree(leaf) == "tasks"
