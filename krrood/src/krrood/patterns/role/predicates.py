from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Type

from krrood.entity_query_language.predicate import Predicate
from krrood.entity_query_language.utils import T
from krrood.patterns.role.role import Role


@dataclass(eq=False)
class HasRole(Predicate, Generic[T]):
    """
    Predicate that checks if an entity has a specific role type.
    """
    entity: T
    """
    The entity to check.
    """
    role: Type[Role[T]]
    """
    The role type to check.
    """

    def __call__(self) -> bool:
        return Role.has_role(self.entity, self.role)
