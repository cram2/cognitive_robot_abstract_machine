from __future__ import annotations

from typing_extensions import List

from krrood.entity_query_language.verbalization.chain_utils import PathStep
from krrood.entity_query_language.verbalization.fragments.base import (
    PhraseFragment,
    RoleFragment,
    Fragment,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Prepositions,
)


def _attribute_fragment(step: PathStep) -> RoleFragment:
    """:return: A role-tagged attribute fragment for *step*."""
    return RoleFragment(
        text=step.name,
        role=SemanticRole.ATTRIBUTE,
        source_reference=step.source_reference,
    )


def possessive_path(parts: List[PathStep], root_fragment: Fragment) -> Fragment:
    """:return: *"the <inner> of the <outer> of <root>"* (parts iterated innermost-first)."""
    if not parts:
        return root_fragment
    reversed_parts = list(reversed(parts))
    fragment_parts: List[Fragment] = [
        Articles.THE.as_fragment(),
        _attribute_fragment(reversed_parts[0]),
    ]
    for step in reversed_parts[1:]:
        fragment_parts.extend(
            [
                Prepositions.OF_THE.as_fragment(),
                _attribute_fragment(step),
            ]
        )
    fragment_parts.extend([Prepositions.OF.as_fragment(), root_fragment])
    return PhraseFragment(parts=fragment_parts)


def pronominal_path(parts: List[PathStep], pronoun: Fragment) -> Fragment:
    """:return: *"its attribute"* (single hop) or *"the attribute of its foo"* (multi-hop)."""
    if not parts:
        return pronoun
    reversed_parts = list(reversed(parts))
    last = len(reversed_parts) - 1
    fragment_parts: List[Fragment] = []
    for index, step in enumerate(reversed_parts):
        attribute_fragment = _attribute_fragment(step)
        if index == 0 and index != last:
            fragment_parts.extend([Articles.THE.as_fragment(), attribute_fragment])
        elif index == 0:  # single attribute → "its booking_date"
            fragment_parts.extend([pronoun, attribute_fragment])
        elif index == last:  # adjacent to the elided root → "of its amount_details"
            fragment_parts.extend(
                [Prepositions.OF.as_fragment(), pronoun, attribute_fragment]
            )
        else:
            fragment_parts.extend(
                [Prepositions.OF_THE.as_fragment(), attribute_fragment]
            )
    return PhraseFragment(parts=fragment_parts)
