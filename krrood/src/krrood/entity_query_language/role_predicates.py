"""
Entity query language predicates about :class:`~krrood.patterns.role.Role` objects.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any

from krrood.entity_query_language.predicate import Predicate, RenderedFields
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.vocabulary.english import Prepositions
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
    clause,
    Copula,
    Noun,
)
from krrood.patterns.role import Role


@dataclass(kw_only=True, eq=False)
class IsSameSemanticEntity(Predicate):
    """
    Whether two operands are the same underlying entity, as decided by
    :meth:`~krrood.patterns.role.Role.is_same_semantic_entity`, usable both as a direct
    boolean and symbolically inside a query.
    """

    first_entity: Any
    """
    The first operand; unwrapped to its root persistent entity when it is a role.
    """

    second_entity: Any
    """
    The second operand; unwrapped to its root persistent entity when it is a role.
    """

    def __call__(self) -> bool:
        return Role.is_same_semantic_entity(
            first_entity=self.first_entity, second_entity=self.second_entity
        )

    @classmethod
    def _verbalization_fragment_(
        cls, *, fields: RenderedFields
    ) -> VerbalizationFragment:
        """:return: the clause *"<first_entity> is the same semantic entity as
        <second_entity>"*."""
        return clause(
            Noun(content=fields["first_entity"]),
            Copula(),
            Noun.the(head="same semantic entity"),
            Prepositions.AS,
            Noun(content=fields["second_entity"]),
        )
