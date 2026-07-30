"""
Fixture module for ``test_quantifier_overload_types``.

Not itself a pytest test module (no ``test_`` names, so pytest never
collects it): ``mypy`` alone consumes it. Each
``an()``/``a()``/``the()`` overload is exercised against a concrete call
shape and the statically-inferred type is checked with
:func:`typing_extensions.assert_type`, so a change to the overloads that
silently changes the inferred type fails ``mypy``, not just at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any, Callable, assert_type

from krrood.entity_query_language.factories import a, an, entity, the
from krrood.entity_query_language.query.match import Match
from krrood.entity_query_language.query.query import Entity


@dataclass
class Robot:
    name: str
    battery: int


def make_robot(name: str = "R2") -> Robot:
    return Robot(name, battery=100)


# %% Type[T] overload: builds a Match[T] from the type itself

assert_type(an(Robot), Match[Robot])
assert_type(a(Robot), Match[Robot])
assert_type(the(Robot), Match[Robot])

# %% Callable[..., T] overload: T inferred from the factory's own return annotation

assert_type(an(make_robot), Match[Robot])
assert_type(a(make_robot), Match[Robot])
assert_type(the(make_robot), Match[Robot])

# %% Callable[..., T] overload: explicit target_type when the factory can't be inferred

untyped_factory: Callable[..., Any] = make_robot

assert_type(an(untyped_factory, target_type=Robot), Match[Robot])
assert_type(a(untyped_factory, target_type=Robot), Match[Robot])
assert_type(the(untyped_factory, target_type=Robot), Match[Robot])


# %% bare T overload: quantifying an existing symbolic expression preserves its type


def quantify_value(robot: Robot) -> None:
    assert_type(an(robot), Robot)
    assert_type(a(robot), Robot)
    assert_type(the(robot), Robot)


def quantify_entity(robot: Robot) -> None:
    built = entity(robot)
    assert_type(built, Entity[Robot])
    assert_type(an(built), Entity[Robot])
    assert_type(a(built), Entity[Robot])
    assert_type(the(built), Entity[Robot])
