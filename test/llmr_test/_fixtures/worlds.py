"""Pytest fixtures that populate SymbolGraph with deterministic world objects.

These fixtures rely on the root ``cleanup_after_test`` autouse fixture to reset
SymbolGraph before/after each test, so every test starts with a known blank world.

Fixtures:
  :func:`simple_world`  — dict of lightweight ``SimpleNamespace`` objects for
                          duck-typed accessor tests (no SymbolGraph registration).
  :func:`symbol_world`  — populated SymbolGraph with bodies, annotations, a
                          structural link, and a robot-shaped annotation;
                          returned as a dict for targeted lookups in tests.
  :func:`robot_world`   — small SymbolGraph with a :class:`Manipulator` instance
                          registered so tests can exercise MRO-based routing.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing_extensions import Any, Dict

import pytest
from krrood.symbol_graph.symbol_graph import SymbolGraph

from .symbols import (
    FakeRobotAnnotation,
    Manipulator,
    MilkAnnotation,
    ParallelGripperLike,
    WorldBody,
)


@pytest.fixture
def simple_world() -> Dict[str, Any]:
    """Return duck-typed body stand-ins — useful for accessor / serialiser unit tests.

    Nothing is registered in SymbolGraph; the dict is only for tests that exercise
    body-level helpers (``body_display_name``, ``body_xyz``, ``body_bounding_box``)
    or grounding warnings against an empty graph.
    """
    milk = SimpleNamespace(name="milk")
    table = SimpleNamespace(name="table")
    fridge = SimpleNamespace(name="fridge")
    return {"milk": milk, "table": table, "fridge": fridge}


@pytest.fixture
def symbol_world() -> Dict[str, Any]:
    """Populate SymbolGraph with a small deterministic world and return handles to each instance.

    Contents:
      ``table``            plain body, no parent.
      ``counter``          plain body, no parent.
      ``milk_on_table``    body with ``table`` as its parent.
      ``milk_on_counter``  body with ``counter`` as its parent.
      ``red_cup``/``blue_cup``  ambiguous-by-name bodies for attribute-filter tests.
      ``structural``       body named ``base_link`` — hits ``_STRUCTURAL_SUFFIXES``.
      ``robot_owned``      body attached to a robot annotation (filtered out by default).
      ``annotation``       :class:`MilkAnnotation` over ``milk_on_table``.
      ``robot_annotation`` :class:`FakeRobotAnnotation` over ``robot_owned``.
    """
    graph = SymbolGraph()
    graph.clear()
    table = WorldBody("table")
    counter = WorldBody("counter")
    milk_on_table = WorldBody("milk_on_table", parent=table)
    milk_on_counter = WorldBody("milk_on_counter", parent=counter)
    red_cup = WorldBody("red_cup")
    blue_cup = WorldBody("blue_cup")
    structural = WorldBody("base_link")
    robot_owned = WorldBody("robot_base")
    annotation = MilkAnnotation(milk_on_table)
    robot_annotation = FakeRobotAnnotation(robot_owned)
    for instance in (
        table,
        counter,
        milk_on_table,
        milk_on_counter,
        red_cup,
        blue_cup,
        structural,
        robot_owned,
        annotation,
        robot_annotation,
    ):
        graph.ensure_wrapped_instance(instance)
    return {
        "table": table,
        "counter": counter,
        "milk_on_table": milk_on_table,
        "milk_on_counter": milk_on_counter,
        "red_cup": red_cup,
        "blue_cup": blue_cup,
        "structural": structural,
        "robot_owned": robot_owned,
        "annotation": annotation,
        "robot_annotation": robot_annotation,
        "body_type": WorldBody,
        "annotation_type": MilkAnnotation,
        "robot_annotation_type": FakeRobotAnnotation,
    }


@pytest.fixture
def robot_world() -> Dict[str, Any]:
    """Populate SymbolGraph with one :class:`Manipulator` and one :class:`ParallelGripperLike`.

    Lets tests exercise:
      - robot-annotation MRO guard in the serialiser.
      - Manipulator resolution in the grounder (expected-type path).
    """
    graph = SymbolGraph()
    graph.clear()
    left = Manipulator(name="left_hand")
    right = ParallelGripperLike(name="right_hand")
    for instance in (left, right):
        graph.ensure_wrapped_instance(instance)
    return {
        "left": left,
        "right": right,
        "manipulator_type": Manipulator,
        "gripper_type": ParallelGripperLike,
    }
