"""Conftest for world tests — provides simple world fixtures."""
from __future__ import annotations

from typing_extensions import Dict, Any
import pytest
from types import SimpleNamespace
from krrood.symbol_graph.symbol_graph import Symbol, SymbolGraph


@pytest.fixture
def simple_world() -> Dict[str, Any]:
    """Return a dict of mock world objects for grounding tests.

    Uses SimpleNamespace to simulate body objects with .name attributes.
    """
    milk = SimpleNamespace(name="milk")
    table = SimpleNamespace(name="table")
    fridge = SimpleNamespace(name="fridge")
    return {"milk": milk, "table": table, "fridge": fridge}


class WorldBody(Symbol):
    def __init__(self, name: str, parent: "WorldBody | None" = None):
        self.name = name
        self.parent_connection = (
            SimpleNamespace(parent=parent) if parent is not None else None
        )


class MilkAnnotation(Symbol):
    _synonyms = {"milk"}

    def __init__(self, *bodies: WorldBody):
        self.bodies = list(bodies)


class FakeRobotAnnotation(Symbol):
    def __init__(self, *bodies: WorldBody):
        self.bodies = list(bodies)
        self.root = bodies[0] if bodies else None
        self._robot = self


@pytest.fixture
def symbol_world() -> Dict[str, Any]:
    """Populate SymbolGraph with small deterministic world objects."""
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
