from __future__ import annotations

from dataclasses import dataclass

from krrood.patterns.role import Role

# ---------------------------------------------------------------------------
# Simple two-role / one-taker scenario
# ---------------------------------------------------------------------------


@dataclass(kw_only=True, eq=False)
class PersonForRoleRecursion:
    name: str

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, PersonForRoleRecursion) and self.name == other.name


@dataclass(kw_only=True, eq=False)
class StudentForRoleRecursion(Role[PersonForRoleRecursion]):
    student_id: str


@dataclass(kw_only=True, eq=False)
class TeacherForRoleRecursion(Role[PersonForRoleRecursion]):
    employee_id: str


# ---------------------------------------------------------------------------
# Chained-role scenario (three levels deep)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class BaseForRoleRecursion:
    base_attr: str = "base"


@dataclass(kw_only=True, eq=False)
class IntermediateForRoleRecursion(Role[BaseForRoleRecursion]):
    inter_attr: str = "inter"


@dataclass(kw_only=True, eq=False)
class TopForRoleRecursion(Role[IntermediateForRoleRecursion]):
    top_attr: str = "top"
