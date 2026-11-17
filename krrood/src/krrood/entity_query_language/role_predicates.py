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


@dataclass(eq=False, kw_only=True)
class IsSubClassOrRole(Predicate):
    """
    Predicate asserting that a class is a subclass of another in the sense of an
    ontology: it inherits from the other class, or it is a
    :class:`~krrood.patterns.role.Role` whose role takers are of a subclass of it.
    """

    sub_class: type
    """
    The class that is checked.
    """

    super_class: type | tuple[type, ...]
    """
    The class, or any of the classes, the checked class must be a subclass of.
    """

    def __call__(self) -> bool:
        return self.holds(sub_class=self.sub_class, super_class=self.super_class)

    @staticmethod
    def holds(*, sub_class: type, super_class: type | tuple[type, ...]) -> bool:
        """
        :return: Whether *sub_class* inherits from *super_class*, or is a role whose
            role takers are of a subclass of it, see :meth:`Role.is_role_for`.
        """
        if issubclass(sub_class, super_class):
            return True
        if not issubclass(sub_class, Role) or sub_class is Role:
            return False
        return sub_class.is_role_for(taker_class=super_class)

    @classmethod
    def _verbalization_fragment_(
        cls, *, fields: RenderedFields
    ) -> VerbalizationFragment:
        """:return: the clause *"<sub_class> is a subclass of or a role for
        <super_class>"*."""
        return clause(
            Noun(content=fields["sub_class"]),
            Copula(),
            Noun(content="subclass of or a role"),
            Prepositions.FOR,
            Noun(content=fields["super_class"]),
        )

    @classmethod
    def _example_operand_values_(cls) -> dict[str, Any]:
        """
        A class stands for :attr:`super_class`, see :meth:`~krrood.entity_query_language
        .predicate.SymbolicCallable._example_operand_values_`.
        """
        return {"super_class": object}
