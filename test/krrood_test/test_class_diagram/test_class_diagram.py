from dataclasses import is_dataclass, fields, Field
from typing import Optional, List

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

    wrapped_generic_class = diagram.get_wrapped_class(GenericClass)

    generic_float: WrappedSpecializedGeneric = diagram.get_wrapped_class(
        GenericClass[float]
    )

    assert len(generic_float.fields) == 3

    float_field = generic_float.fields[0]
    assert float_field.type_endpoint is float

    generic_position = diagram.get_wrapped_class(GenericClass[Position])
    assert len(generic_position.fields) == 3
    position_field = generic_position.fields[0]
    assert position_field.type_endpoint is Position

    # get the inheritance relations that point to generic_float
    inheritance_relations_for_generic_float = [
        r for r in diagram.inheritance_relations if r.target is generic_float
    ]
    assert len(inheritance_relations_for_generic_float) == 1
    assert inheritance_relations_for_generic_float[0].source is wrapped_generic_class

    # get the inheritance relations that point to generic_position
    inheritance_relations_for_generic_position = [
        r for r in diagram.inheritance_relations if r.target is generic_position
    ]
    assert len(inheritance_relations_for_generic_position) == 1
    assert inheritance_relations_for_generic_position[0].source is wrapped_generic_class


def test_make_specialized_dataclass():
    type_ = GenericClass[float]
    result = make_specialized_dataclass(type_)
    value, optional_value, container = fields(result)
    assert value.type == float
    assert optional_value.type == Optional[float]
    assert container.type == List[float]
