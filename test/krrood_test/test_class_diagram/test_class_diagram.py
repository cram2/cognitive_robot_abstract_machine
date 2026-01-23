from dataclasses import is_dataclass

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.class_diagrams.utils import classes_of_module
from ..dataset import example_classes


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
