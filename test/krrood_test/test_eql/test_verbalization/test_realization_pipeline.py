"""
The realisation passes share one ``RealizationPass`` contract and run as an ordered
pipeline (Phase 5).

The stateless leaf-rewriting passes (determiner, morphology) are ``RewritePass`` built
on ``map_fragment``, so they do not re-walk the tree by hand. This test pins the
contract and that ``realize_tree`` is exactly the ordered composition of those passes
(no behaviour change).
"""

from __future__ import annotations

from krrood.entity_query_language.verbalization.fragments.base import (
    flatten_fragment_to_plain_text,
    NounPhrase,
    PhraseFragment,
    RoleFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    GrammaticalNumber,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.rendering.coreference_processor import (
    CoreferenceProcessor,
)
from krrood.entity_query_language.verbalization.rendering.determiner_processor import (
    DeterminerProcessor,
)
from krrood.entity_query_language.verbalization.rendering.morphology_processor import (
    MorphologyProcessor,
)
from krrood.entity_query_language.verbalization.rendering.orthography_processor import (
    OrthographyProcessor,
)
from krrood.entity_query_language.verbalization.rendering.passes import (
    RealizationPass,
    RewritePass,
)
from krrood.entity_query_language.verbalization.rendering.realization import (
    realize_tree,
)
from krrood.entity_query_language.verbalization.vocabulary.english import Punctuation


def test_every_processor_is_a_realization_pass():
    for processor in (
        CoreferenceProcessor(),
        DeterminerProcessor(),
        MorphologyProcessor(),
        OrthographyProcessor(),
    ):
        assert isinstance(processor, RealizationPass)


def test_stateless_leaf_passes_are_rewrite_passes():
    assert isinstance(DeterminerProcessor(), RewritePass)
    assert isinstance(MorphologyProcessor(), RewritePass)


def _sample_tree() -> PhraseFragment:
    plural_indefinite = NounPhrase(
        head=RoleFragment(text="Cabinet", role=SemanticRole.VARIABLE),
        definiteness=Definiteness.INDEFINITE,
        number=GrammaticalNumber.PLURAL,
    )
    return PhraseFragment(
        parts=[
            plural_indefinite,
            Punctuation.COMMA.as_fragment(),
            RoleFragment(text="end", role=SemanticRole.VARIABLE),
        ]
    )


def test_realize_tree_is_the_ordered_pass_pipeline():
    """
    ``realize_tree`` equals coreference → determiner → morphology → orthography, applied
    in that order — exercised on a tree that needs every pass (determiner-drop +
    pluralise + comma glue).
    """
    tree = _sample_tree()
    manual = OrthographyProcessor().process(
        MorphologyProcessor().process(
            DeterminerProcessor().process(CoreferenceProcessor().process(tree))
        )
    )
    assert flatten_fragment_to_plain_text(
        realize_tree(tree)
    ) == flatten_fragment_to_plain_text(manual)
    # And it really produced the determiner-drop + plural + glued comma.
    assert flatten_fragment_to_plain_text(realize_tree(tree)) == "Cabinets, end"
