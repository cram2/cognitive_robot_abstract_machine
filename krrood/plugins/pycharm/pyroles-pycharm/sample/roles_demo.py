"""
Demo file for the Python Role Lens plugin.

Open this in the sandbox IDE launched by `./gradlew runIde` and try:
  - typing a dot after one of the objects below to see delegated members in completion
  - hovering a delegated attribute to see its inferred type
  - Ctrl/Cmd+Click on a delegated attribute to jump to its real declaration

The plugin reads the taker type from the ``Role[...]`` base's generic argument, and detects
the ``Role`` base structurally (it defines ``__getattr__``), so this works even though this
``Role`` is ``roles_demo.Role`` rather than the krrood ``Role`` configured in the plugin.
The taker is held in a domain-named field, exactly as krrood roles do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

RoleTakerT = TypeVar("RoleTakerT")


@dataclass
class Role(Generic[RoleTakerT]):
    role_taker: RoleTakerT

    def __getattr__(self, name: str) -> Any:
        return getattr(self.role_taker, name)


# --- An inheritance chain on the taker side -----------------------------------
@dataclass
class Entity:
    id: int

    def describe(self) -> str:
        return f"<{type(self).__name__} {self.id}>"


@dataclass
class Person(Entity):
    name: str
    age: int


# --- A direct role ------------------------------------------------------------
@dataclass
class Teacher(Role[Person]):
    courses: list[str] = field(default_factory=list)


# --- A transitive role (subclass of a role) -----------------------------------
@dataclass
class SeniorTeacher(Teacher):
    tenure_years: int = 0


# --- A nested role (the taker is itself a role) -------------------------------
@dataclass
class Department(Role[Teacher]):
    budget: float = 0.0


def demo() -> None:
    teacher = Teacher(role_taker=Person(id=1, name="Ahmed", age=20), courses=["Math"])
    print(teacher.name)  # from Person            -> str
    print(teacher.age)  # from Person            -> int
    print(teacher.id)  # inherited from Entity  -> int
    print(teacher.describe())  # inherited method
    print(teacher.courses)  # the role's own field

    senior = SeniorTeacher(role_taker=Person(id=2, name="Sara", age=35), tenure_years=8)
    print(senior.name)  # transitive: resolved through Teacher -> Role[Person]
    print(senior.tenure_years)

    dept = Department(role_taker=teacher, budget=1_000.0)
    print(dept.courses)  # nested: Department -> Role[Teacher] (Teacher's own field)
    print(dept.name)  # nested + delegated: Teacher -> Role[Person] -> Person


if __name__ == "__main__":
    demo()
