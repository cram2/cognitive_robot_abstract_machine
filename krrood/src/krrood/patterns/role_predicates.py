"""
Predicates about roles and the entities they are roles of.
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
    Predicate asserting that two operands refer to the same underlying entity.

    Each operand is unwrapped to its :attr:`root_persistent_entity
    <krrood.patterns.role.Role.root_persistent_entity>` when it is a
    :class:`~krrood.patterns.role.Role` (otherwise used as-is), and the two resolved roots
    are compared by object identity. This sees through role chains, so a role, any role
    layered on top of it, and the root taker itself all count as the same entity.

    Like every :class:`~krrood.entity_query_language.predicate.Predicate` it can be used
    both as a direct boolean on concrete operands (``bool(IsSameSemanticEntity(a, b))``) and
    symbolically inside an entity-query-language ``where`` clause.
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
        return self._root_of(value=self.first_entity) is self._root_of(
            value=self.second_entity
        )

    @staticmethod
    def _root_of(*, value: Any) -> Any:
        """
        :return: ``value``'s root persistent entity when it is a role, otherwise ``value``.
        """
        return value.root_persistent_entity if isinstance(value, Role) else value

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
            role-taker type is a subclass or role of *super_class*.
        """
        if issubclass(sub_class, super_class):
            return True
        if issubclass(sub_class, Role) and sub_class is not Role:
            return IsSubClassOrRole.holds(
                sub_class=sub_class.get_role_taker_type(), super_class=super_class
            )
        return False

    @classmethod
    def _verbalization_fragment_(
        cls, *, fields: RenderedFields
    ) -> VerbalizationFragment:
        """:return: the clause *"<sub_class> is a subclass or role of
        <super_class>"*."""
        return clause(
            Noun(content=fields["sub_class"]),
            Copula(),
            Noun(content="subclass or role"),
            Prepositions.OF,
            Noun(content=fields["super_class"]),
        )

    @classmethod
    def _example_operand_values_(cls) -> dict[str, Any]:
        """
        A class stands for :attr:`super_class`, see :meth:`~krrood.entity_query_language
        .predicate.SymbolicCallable._example_operand_values_`.
        """
        return {"super_class": object}
