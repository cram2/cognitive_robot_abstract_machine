from __future__ import annotations

from dataclasses import dataclass

from krrood.patterns.role import Role

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


@dataclass(eq=False)
class TeacherForRoleRecursion(Role[PersonForRoleRecursion]):
    employee_id: str


# ---------------------------------------------------------------------------
# Chained-role scenario (three levels deep)
# ---------------------------------------------------------------------------


@dataclass
class BaseForRoleRecursion:
    base_attr: str = "base"


@dataclass(eq=False)
class IntermediateForRoleRecursion(Role[BaseForRoleRecursion]):
    inter_attr: str = "inter"


@dataclass(eq=False)
class TopForRoleRecursion(Role[IntermediateForRoleRecursion]):
    top_attr: str = "top"
