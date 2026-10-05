"""
Fixture module for ``test_quantifier_overload_types``.

Not itself a pytest test module (no ``test_`` names, so pytest never
collects it): ``mypy`` alone consumes it. Each
``an()``/``a()``/``the()`` overload is exercised against a concrete call
shape and the statically-inferred type is checked with
:func:`typing_extensions.assert_type`, so a change to the overloads that
silently changes the inferred type fails ``mypy``, not just at runtime.

The type/factory overloads return ``Union[T, Match[T]]``: the ``T`` member
is the same static lie :func:`~krrood.entity_query_language.factories.variable`
tells, so IDEs offer the matched class's own attributes, while the
``Match[T]`` member keeps the match-building methods available.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any, Callable, Union, assert_type

from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.entity_query_language.factories import a, an, entity, the
from krrood.entity_query_language.query.match import Match
from krrood.entity_query_language.query.query import Entity


@dataclass
class Robot:
    name: str
    battery: int


def make_robot(name: str = "R2") -> Robot:
    return Robot(name, battery=100)


# %% Type[T] overload: builds a match that reads like an instance of the class

assert_type(an(Robot), Union[Robot, Match[Robot]])
assert_type(a(Robot), Union[Robot, Match[Robot]])
assert_type(the(Robot), Union[Robot, Match[Robot]])

# %% attribute access resolves to the class's own field types

assert_type(an(Robot).battery, Union[int, Attribute[Robot]])
assert_type(the(Robot).name, Union[str, Attribute[Robot]])

# %% Callable[..., T] overload: T inferred from the factory's own return annotation

assert_type(an(make_robot), Union[Robot, Match[Robot]])
assert_type(a(make_robot), Union[Robot, Match[Robot]])
assert_type(the(make_robot), Union[Robot, Match[Robot]])

# %% Callable[..., T] overload: explicit target_type when the factory can't be inferred

untyped_factory: Callable[..., Any] = make_robot

assert_type(an(untyped_factory, target_type=Robot), Union[Robot, Match[Robot]])
assert_type(a(untyped_factory, target_type=Robot), Union[Robot, Match[Robot]])
assert_type(the(untyped_factory, target_type=Robot), Union[Robot, Match[Robot]])

# %% keyword-argument construction keeps the instance-like type

robot_match = an(Robot)
assert isinstance(robot_match, Match)
assert_type(robot_match(name="R2", battery=100), Union[Robot, Match[Robot]])

# %% match chaining stays available on the match side of the union

constrained = a(Robot)
assert isinstance(constrained, Match)
scoped = constrained(name="R2", battery=100)
assert isinstance(scoped, Match)
assert_type(scoped.from_([Robot("R2", 100)]), Match[Robot])
assert_type(scoped.where(scoped.battery == 100), Match[Robot])
assert_type(scoped.ordered_by(scoped.battery), Match[Robot])
assert_type(scoped.limit(1), Match[Robot])


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
