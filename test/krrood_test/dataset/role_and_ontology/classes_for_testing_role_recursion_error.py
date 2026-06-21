from __future__ import annotations

from dataclasses import dataclass

from krrood.patterns.role import Role, role_taker_field

# ---------------------------------------------------------------------------
# Simple two-role / one-taker scenario
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class PersonForRoleRecursion:
    name: str

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, PersonForRoleRecursion) and self.name == other.name


@dataclass(eq=False)
class StudentForRoleRecursion(Role[PersonForRoleRecursion]):
    student_id: str
    person: PersonForRoleRecursion = role_taker_field()


@dataclass(eq=False)
class TeacherForRoleRecursion(Role[PersonForRoleRecursion]):
    employee_id: str
    person: PersonForRoleRecursion = role_taker_field()


# ---------------------------------------------------------------------------
# Chained-role scenario (three levels deep)
# ---------------------------------------------------------------------------


@dataclass
class BaseForRoleRecursion:
    base_attr: str = "base"


@dataclass(eq=False)
class IntermediateForRoleRecursion(Role[BaseForRoleRecursion]):
    base: BaseForRoleRecursion = role_taker_field()
    inter_attr: str = "inter"


@dataclass(eq=False)
class TopForRoleRecursion(Role[IntermediateForRoleRecursion]):
    inter: IntermediateForRoleRecursion = role_taker_field()
    top_attr: str = "top"
