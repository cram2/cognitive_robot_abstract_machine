from dataclasses import is_dataclass, fields

from krrood.class_diagrams.class_diagram import (
    ClassDiagram,
    WrappedSpecializedGeneric,
    make_specialized_dataclass,
)
from krrood.class_diagrams.utils import classes_of_module
from ..dataset import example_classes
from ..dataset.example_classes import Position, GenericClassAssociation, GenericClass


def test_class_diagram_visualization():
    classes = filter(
        is_dataclass,
        classes_of_module(example_classes),
    )
    diagram = ClassDiagram(classes)
    assert len(diagram.wrapped_classes) > 0
    assert len(diagram._dependency_graph.edges()) > 0
    associations = diagram.associations

    wrapped_pose = diagram.get_wrapped_class(example_classes.Pose)
    wrapped_position = diagram.get_wrapped_class(example_classes.Position)
    wrapped_positions = diagram.get_wrapped_class(example_classes.Positions)

    assert (
        len(
            [
                a
                for a in associations
                if a.source == wrapped_pose and a.target == wrapped_position
            ]
        )
        == 1
    )

    assert (
        len(
            [
                a
                for a in associations
                if a.source == wrapped_positions and a.target == wrapped_position
            ]
        )
        == 1
    )

    wrapped_positions_subclass = diagram.get_wrapped_class(
        example_classes.PositionsSubclassWithAnotherPosition
    )
    inheritances = diagram.inheritance_relations

    assert (
        len(
            [
                a
                for a in inheritances
                if a.source == wrapped_positions
                and a.target == wrapped_positions_subclass
            ]
        )
        == 1
    )


def test_underspecified_classes():

    classes = filter(
        is_dataclass,
        classes_of_module(example_classes),
    )
    diagram = ClassDiagram(classes)

    r = diagram.get_wrapped_class(example_classes.UnderspecifiedTypesContainer)
    assert r.clazz is example_classes.UnderspecifiedTypesContainer


def test_create_nodes_for_specialized_generic():
    classes = [Position, GenericClassAssociation, GenericClass]
    diagram = ClassDiagram(classes)
    generic_float: WrappedSpecializedGeneric = diagram.get_wrapped_class(
        GenericClass[float]
    )

    assert len(generic_float.fields) == 1

    float_field = generic_float.fields[0]
    assert float_field.type_endpoint is float

    generic_position = diagram.get_wrapped_class(GenericClass[Position])
    assert len(generic_position.fields) == 1
    position_field = generic_position.fields[0]
    assert position_field.type_endpoint is Position
