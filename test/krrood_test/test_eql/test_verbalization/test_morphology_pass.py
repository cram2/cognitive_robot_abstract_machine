"""
Unit tests for the morphology realisation pass — the single place grammatical number is
*applied*.

Assemblers tag a leaf ``GrammaticalNumber.PLURAL``; this pass pluralises it.
"""

from __future__ import annotations

from krrood.entity_query_language.verbalization.fragments.base import (
    CardinalNumber,
    CollectionElementNoun,
    flatten_fragment_to_plain_text,
    IndefiniteArticle,
    IndexOrdinal,
    PhraseFragment,
    VerbFormFragment,
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
from krrood.entity_query_language.verbalization.vocabulary.english import Articles


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
    assert _text(MorphologyProcessor().process(fragment=tree)) == "Robots of Cabinet"


def test_pass_leaves_singular_untouched():
    leaf = RoleFragment(text="Robot", role=SemanticRole.VARIABLE)
    assert _text(MorphologyProcessor().process(fragment=leaf)) == "Robot"


def test_pass_is_idempotent():
    leaf = RoleFragment(
        text="task", role=SemanticRole.ATTRIBUTE, number=GrammaticalNumber.PLURAL
    )
    once = MorphologyProcessor().process(fragment=leaf)
    twice = MorphologyProcessor().process(fragment=once)
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
    assert _text(MorphologyProcessor().process(fragment=block)) == "the containers"


def test_realize_subtree_runs_pass_then_flattens():
    leaf = RoleFragment(
        text="task", role=SemanticRole.ATTRIBUTE, number=GrammaticalNumber.PLURAL
    )
    assert realize_subtree(leaf) == "tasks"


# %% words decided while the clause was built


def test_indefinite_article_agrees_with_the_word_realised_after_it():
    """
    The article agrees with the ordinal the same pass realises, not with the noun.
    """
    tree = PhraseFragment(
        parts=[
            IndefiniteArticle(text=Articles.A.text),
            IndexOrdinal(index=7),
            WordFragment(text="Robot"),
        ]
    )
    assert _text(MorphologyProcessor().process(fragment=tree)) == "an eighth Robot"


def test_indefinite_article_before_a_consonant_stays_a():
    """
    An indefinite article before a consonant sound reads *a*.
    """
    tree = PhraseFragment(
        parts=[IndefiniteArticle(text=Articles.A.text), WordFragment(text="Robot")]
    )
    assert _text(MorphologyProcessor().process(fragment=tree)) == "a Robot"


def test_verb_form_is_reduced_to_its_lemma_before_negation():
    """
    A verb given as an inflected form negates with do-support on its lemma.
    """
    leaf = VerbFormFragment(text="connects", negated=True)
    assert MorphologyProcessor().rewrite(leaf=leaf).text == "does not connect"


def test_collection_element_noun_is_singularised():
    """
    A noun naming one element of a collection reads in the singular.
    """
    leaf = CollectionElementNoun(text="tasks", role=SemanticRole.ATTRIBUTE)
    assert MorphologyProcessor().rewrite(leaf=leaf).text == "task"


def test_counts_and_positions_are_realised_as_words():
    """
    A count reads as its cardinal and a negative index as its position from the end.
    """
    tree = PhraseFragment(parts=[CardinalNumber(count=7), IndexOrdinal(index=-2)])
    assert _text(MorphologyProcessor().process(fragment=tree)) == "seven second to last"
