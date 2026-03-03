import os
from os.path import dirname

from krrood.ripple_down_rules import *
from krrood.ripple_down_rules.datastructures.tracked_object import X
from .datasets import (
    Drawer,
    Handle,
    Cabinet,
    View,
    WorldEntity,
    Body,
    Connection,
    Container,
)


def test_construct_class_hierarchy():
    isA.infer()
    Drawer.to_dot(os.path.join(os.path.dirname(__file__), "dependency_graph"))
    assert len(Drawer._dependency_graph.nodes()) == 20
    assert len(Drawer._dependency_graph.edges()) == 17


def test_construct_class_composition():
    isA.infer()
    has.infer()
    Drawer.to_dot(os.path.join(os.path.dirname(__file__), "dependency_graph"))
    assert len(Drawer._dependency_graph.nodes()) == 20
    assert len(Drawer._dependency_graph.edges()) == 22
    Drawer.to_dot(os.path.join(os.path.dirname(__file__), "dependency_graph"))


# @pytest.mark.skip("Not Implemented yet")
def test_construct_class_composition_and_dependency():
    assert next(has(Drawer, Handle))
    assert next(has(Cabinet, Drawer))
    assert list(has(Cabinet, X)) == [(Cabinet, Drawer), (Cabinet, Container)]
    assert list(has(Cabinet, X, recursive=True)) == [
        (Cabinet, Drawer),
        (Cabinet, Container),
        (Cabinet, Container),
        (Cabinet, Handle),
    ]
    assert list(has(X, Handle)) == [(Drawer, Handle)]
    assert list(has(X, Handle, recursive=True)) == [(Drawer, Handle), (Cabinet, Handle)]
    assert isA(Cabinet, View)
    assert isA(Cabinet, WorldEntity)
    assert not list(has(Cabinet, Handle))
    assert next(has(Cabinet, Handle, recursive=True))
    assert next(has(Cabinet, Body))
    assert next(has(Cabinet, WorldEntity))
    assert not list(has(Cabinet, Connection, recursive=True))
    assert list(has((Cabinet, Drawer), Handle)) == [(Drawer, Handle)]
    assert list(has((Cabinet, Drawer), Handle, recursive=True)) == [
        (Cabinet, Handle),
        (Drawer, Handle),
    ]
    assert list(has((Cabinet, Drawer), X)) == [
        (Cabinet, Drawer),
        (Cabinet, Container),
        (Drawer, Container),
        (Drawer, Handle),
    ]
    assert list(has((Cabinet, Drawer), (Handle, Container))) == [
        (Cabinet, Container),
        (Drawer, Handle),
        (Drawer, Container),
    ]
