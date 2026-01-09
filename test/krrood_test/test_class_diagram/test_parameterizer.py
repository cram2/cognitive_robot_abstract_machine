from random_events.set import Set
from random_events.variable import Continuous, Integer, Symbolic
from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.class_diagrams.parameterizer import Parameterizer
from ..dataset.example_classes import Position, Orientation, Pose, Atom, Element


def test_parameterizer_with_example_classes():
    """
    Test the Parameterizer on example dataclasses:
    Position, Orientation, Pose, Atom.
    Ensures:
      - All dataclasses and enums are in ClassDiagram
      - Variables extracted correctly via WrappedClass
      - Nested dataclasses are handled recursively
    """
    diagram = ClassDiagram([
        Position,
        Orientation,
        Pose,
        Atom,
        Element,
    ])

    param = Parameterizer()

    position_wc = diagram.get_wrapped_class(Position)
    orientation_wc = diagram.get_wrapped_class(Orientation)
    pose_wc = diagram.get_wrapped_class(Pose)
    atom_wc = diagram.get_wrapped_class(Atom)

    position_variables = param(position_wc)
    orientation_variables = param(orientation_wc)
    pose_variables = param(pose_wc)
    atom_variables = param(atom_wc)

    expected_position_variables = [
        Continuous("Position.x"),
        Continuous("Position.y"),
        Continuous("Position.z"),
    ]

    expected_orientation_variables = [
        Continuous("Orientation.x"),
        Continuous("Orientation.y"),
        Continuous("Orientation.z"),
        Continuous("Orientation.w"),
    ]

    expected_pose_variables = [
        Continuous("Pose.position.x"),
        Continuous("Pose.position.y"),
        Continuous("Pose.position.z"),
        Continuous("Pose.orientation.x"),
        Continuous("Pose.orientation.y"),
        Continuous("Pose.orientation.z"),
        Continuous("Pose.orientation.w"),
    ]

    expected_atom_variables = [
        Symbolic("Atom.element", Set.from_iterable([Element.C, Element.H])),
        Integer("Atom.type"),
        Continuous("Atom.charge"),
    ]

    variables_for_pc = [Continuous(v.name) if isinstance(v, Integer) else v for v in atom_variables]
    pc = param.create_fully_factorized_distribution(variables_for_pc)

    assert position_variables == expected_position_variables
    assert orientation_variables == expected_orientation_variables
    assert pose_variables == expected_pose_variables
    assert [(type(v), v.name) for v in atom_variables] == [(type(v), v.name) for v in expected_atom_variables]
    assert set(pc.variables) == set(variables_for_pc)
