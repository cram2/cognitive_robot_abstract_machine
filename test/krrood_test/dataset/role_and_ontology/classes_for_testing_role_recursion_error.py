from dataclasses import dataclass

from krrood.entity_query_language.factories import variable_from
from krrood.patterns.role.role import Role


@dataclass
class PersonForRoleRecursion:
    name: str


@dataclass(eq=False)
class StudentForRoleRecursion(Role[PersonForRoleRecursion]):
    student_id: str
    person: PersonForRoleRecursion

    @classmethod
    def role_taker_attribute(cls) -> PersonForRoleRecursion:
        return variable_from(cls).person


@dataclass(eq=False)
class TeacherForRoleRecursion(Role[PersonForRoleRecursion]):
    employee_id: str
    person: PersonForRoleRecursion

    @classmethod
    def role_taker_attribute(cls) -> PersonForRoleRecursion:
        return variable_from(cls).person


@dataclass
class BaseForRoleRecursion:
    base_attr: str = "base"

@dataclass(eq=False)
class IntermediateForRoleRecursion(Role[BaseForRoleRecursion]):
    base: BaseForRoleRecursion
    inter_attr: str = "inter"

    @classmethod
    def role_taker_attribute(cls) -> BaseForRoleRecursion:
        return variable_from(cls).base

@dataclass(eq=False)
class TopForRoleRecursion(Role[IntermediateForRoleRecursion]):
    inter: IntermediateForRoleRecursion
    top_attr: str = "top"

    @classmethod
    def role_taker_attribute(cls) -> IntermediateForRoleRecursion:
        return variable_from(cls).inter