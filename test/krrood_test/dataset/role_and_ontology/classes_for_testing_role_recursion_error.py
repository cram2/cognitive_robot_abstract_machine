from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from krrood.entity_query_language.factories import variable_from
from krrood.patterns.role.role import Role, HasRoles
from krrood.patterns.role import HasRoles
from test.krrood_test.dataset.role_and_ontology.role_mixins.classes_for_testing_role_recursion_error_role_mixins import RoleForBaseForRoleRecursion, RoleForIntermediateForRoleRecursion, RoleForPersonForRoleRecursion

# ---------------------------------------------------------------------------
# Simple two-role / one-taker scenario
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class PersonForRoleRecursion(HasRoles):
    name: str

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, PersonForRoleRecursion) and self.name == other.name


@dataclass(eq=False)
class RoleForPersonForRoleRecursion(ABC):
    """Mixin that delegates PersonForRoleRecursion fields to the role taker."""

    @property
    @abstractmethod
    def role_taker(self) -> PersonForRoleRecursion: ...

    @property
    def name(self) -> str:
        return self.role_taker.name

    @name.setter
    def name(self, value: str):
        self.role_taker.name = value


@dataclass(eq=False)
class StudentForRoleRecursion(
    Role[PersonForRoleRecursion], RoleForPersonForRoleRecursion
):
    student_id: str
    person: PersonForRoleRecursion

    @classmethod
    def role_taker_attribute(cls) -> PersonForRoleRecursion:
        return variable_from(cls).person


@dataclass(eq=False)
class TeacherForRoleRecursion(
    Role[PersonForRoleRecursion], RoleForPersonForRoleRecursion
):
    employee_id: str
    person: PersonForRoleRecursion

    @classmethod
    def role_taker_attribute(cls) -> PersonForRoleRecursion:
        return variable_from(cls).person


# ---------------------------------------------------------------------------
# Chained-role scenario (three levels deep)
# ---------------------------------------------------------------------------


@dataclass
class BaseForRoleRecursion(HasRoles):
    base_attr: str = "base"


@dataclass(eq=False)
class RoleForBaseForRoleRecursion(ABC):
    """Mixin that delegates BaseForRoleRecursion fields to the role taker."""

    @property
    @abstractmethod
    def role_taker(self) -> BaseForRoleRecursion: ...

    @property
    def base_attr(self) -> str:
        return self.role_taker.base_attr

    @base_attr.setter
    def base_attr(self, value: str):
        self.role_taker.base_attr = value


@dataclass(eq=False)
class IntermediateForRoleRecursion(
    Role[BaseForRoleRecursion], RoleForBaseForRoleRecursion
):
    base: BaseForRoleRecursion
    inter_attr: str = "inter"

    @classmethod
    def role_taker_attribute(cls) -> BaseForRoleRecursion:
        return variable_from(cls).base


@dataclass(eq=False)
class RoleForIntermediateForRoleRecursion(RoleForBaseForRoleRecursion, ABC):
    """Mixin that delegates IntermediateForRoleRecursion fields to the role taker."""

    @property
    @abstractmethod
    def role_taker(self) -> IntermediateForRoleRecursion: ...

    @property
    def inter_attr(self) -> str:
        return self.role_taker.inter_attr

    @inter_attr.setter
    def inter_attr(self, value: str):
        self.role_taker.inter_attr = value


@dataclass(eq=False)
class TopForRoleRecursion(
    Role[IntermediateForRoleRecursion], RoleForIntermediateForRoleRecursion
):
    inter: IntermediateForRoleRecursion
    top_attr: str = "top"

    @classmethod
    def role_taker_attribute(cls) -> IntermediateForRoleRecursion:
        return variable_from(cls).inter
