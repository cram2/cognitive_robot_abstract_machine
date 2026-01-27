from __future__ import annotations
import pytest
from random_events.set import Set
from random_events.variable import Continuous, Integer, Symbolic
from random_events.product_algebra import Event, SimpleEvent
from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.class_diagrams.parameterizer import Parameterizer
from pycram.datastructures.enums import TorsoState, Arms
from pycram.robot_plans import MoveTorsoAction
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.pick_up import PickUpAction
from pycram.robot_plans.actions.core.placing import PlaceAction
from pycram.datastructures.grasp import GraspDescription

from pycram.datastructures.pose import (
    PoseStamped,
    PyCramPose,
    PyCramVector3,
    PyCramQuaternion,
    Header,
)
from ..dataset.example_classes import (
    Position,
    Orientation,
    Pose,
    Atom,
    Element,
)


@pytest.fixture
def parameterizer() -> Parameterizer:
    """
    Fixture for the Parameterizer instance.
    """
    return Parameterizer()


def test_parameterize_position(parameterizer: Parameterizer):
    """
    Test parameterization of the Position class.
    """
    class_diagram = ClassDiagram([Position])
    wrapped_position = class_diagram.get_wrapped_class(Position)
    variables = parameterizer(wrapped_position)
    expected_variables = [
        Continuous("Position.x"),
        Continuous("Position.y"),
        Continuous("Position.z"),
    ]
    assert variables == expected_variables


def test_parameterize_orientation(parameterizer: Parameterizer):
    """
    Test parameterization of the Orientation class.
    """
    class_diagram = ClassDiagram([Orientation])
    wrapped_orientation = class_diagram.get_wrapped_class(Orientation)
    variables = parameterizer(wrapped_orientation)
    expected_variables = [
        Continuous("Orientation.x"),
        Continuous("Orientation.y"),
        Continuous("Orientation.z"),
        Continuous("Orientation.w"),
    ]

    assert variables == expected_variables


def test_parameterize_pose(parameterizer: Parameterizer):
    """
    Test parameterization of the Pose class.
    """
    class_diagram = ClassDiagram([Pose, Position, Orientation])
    wrapped_pose = class_diagram.get_wrapped_class(Pose)
    variables = parameterizer(wrapped_pose)
    expected_variables = [
        Continuous("Pose.position.x"),
        Continuous("Pose.position.y"),
        Continuous("Pose.position.z"),
        Continuous("Pose.orientation.x"),
        Continuous("Pose.orientation.y"),
        Continuous("Pose.orientation.z"),
        Continuous("Pose.orientation.w"),
    ]

    assert variables == expected_variables


def test_parameterize_atom(parameterizer: Parameterizer):
    """
    Test parameterization of the Atom class.
    """
    class_diagram = ClassDiagram([Atom, Element])
    wrapped_atom = class_diagram.get_wrapped_class(Atom)
    variables = parameterizer(wrapped_atom)
    expected_variables = [
        Symbolic("Atom.element", Set.from_iterable(Element)),
        Integer("Atom.type"),
        Continuous("Atom.charge"),
    ]

    assert [(type(v), v.name) for v in variables] == [
        (type(v), v.name) for v in expected_variables
    ]


def test_create_fully_factorized_distribution(parameterizer: Parameterizer):
    """
    Test for a fully factorized distribution.
    """
    variables = [
        Continuous("Variable.A"),
        Continuous("Variable.B"),
    ]
    probabilistic_circuit = parameterizer.create_fully_factorized_distribution(
        variables
    )
    assert len(probabilistic_circuit.variables) == 2
    assert set(probabilistic_circuit.variables) == set(variables)


def test_parameterize_movetorse_navigate(parameterizer: Parameterizer):
    """
    Test parameterization of a potential robot plan consisting of: MoveTorso - Navigate - MoveTorso.

    This test verifies:
    1. Parameterization of simple robot action plan
    2. Sampling from the constrained distribution and validation of constraints.
    """

    plan_classes = [
        MoveTorsoAction, NavigateAction, PoseStamped, PyCramPose,
        PyCramVector3, PyCramQuaternion, Header
    ]
    class_diagram = ClassDiagram(plan_classes)
    wrapped_move_torso = class_diagram.get_wrapped_class(MoveTorsoAction)
    wrapped_navigate = class_diagram.get_wrapped_class(NavigateAction)

    movetorso_variables1 = parameterizer.parameterize(wrapped_move_torso, prefix="MoveTorsoAction_1")
    navigate_variables = parameterizer(wrapped_navigate)
    movetorso_variables2 = parameterizer.parameterize(wrapped_move_torso, prefix="MoveTorsoAction_2")

    all_variables = movetorso_variables1 + navigate_variables + movetorso_variables2
    variables = {v.name: v for v in all_variables}

    expected_names = {
        "MoveTorsoAction_1.torso_state", "MoveTorsoAction_2.torso_state",
        "NavigateAction.keep_joint_states", "NavigateAction.target_location.header.sequence",
        "NavigateAction.target_location.pose.position.x", "NavigateAction.target_location.pose.position.y",
        "NavigateAction.target_location.pose.position.z", "NavigateAction.target_location.pose.orientation.x",
        "NavigateAction.target_location.pose.orientation.y", "NavigateAction.target_location.pose.orientation.z",
        "NavigateAction.target_location.pose.orientation.w",
    }

    assert set(variables.keys()) == expected_names

    probabilistic_circuit = parameterizer.create_fully_factorized_distribution(all_variables)

    expected_distribution_names = expected_names - {"NavigateAction.target_location.header.sequence"}
    assert {v.name for v in probabilistic_circuit.variables} == expected_distribution_names

    torso_1 = variables["MoveTorsoAction_1.torso_state"]
    torso_2 = variables["MoveTorsoAction_2.torso_state"]

    consistency_events = [SimpleEvent({torso_1: [state], torso_2: [state]}) for state in TorsoState]
    restricted_distribution, _ = probabilistic_circuit.truncated(Event(*consistency_events))
    restricted_distribution.normalize()

    pose_constraints = {
        variables["NavigateAction.target_location.pose.position.x"]: 1.5,
        variables["NavigateAction.target_location.pose.position.y"]: -2.0,
        variables["NavigateAction.target_location.pose.orientation.x"]: 0.0,
        variables["NavigateAction.target_location.pose.orientation.y"]: 0.0,
        variables["NavigateAction.target_location.pose.orientation.z"]: 0.0,
        variables["NavigateAction.target_location.pose.orientation.w"]: 1.0,
    }
    
    final_distribution, _ = restricted_distribution.conditional(pose_constraints)
    final_distribution.normalize()

    target_x, target_y = 1.5, -2.0
    nav_x = variables["NavigateAction.target_location.pose.position.x"]
    nav_y = variables["NavigateAction.target_location.pose.position.y"]

    for sample_values in final_distribution.sample(10):
        sample = dict(zip(final_distribution.variables, sample_values))
        assert sample[torso_1] == sample[torso_2]
        assert sample[nav_x] == target_x
        assert sample[nav_y] == target_y


def test_parameterize_pickup_navigate_place(parameterizer: Parameterizer):
    """
    Test parameterization of a potential robot plan consisting of: PickUp - Navigate - Place.

    This test verifies:
    1. Parameterization of pick up, navigate, placing robot action plan
    2. Creating and sampling from a constrained distribution over the plan variables.
    """

    plan_classes = [
        PickUpAction, NavigateAction, PlaceAction,
        GraspDescription, PoseStamped, PyCramPose,
        PyCramVector3, PyCramQuaternion, Header
    ]
    class_diagram = ClassDiagram(plan_classes)

    wrapped_pickup = class_diagram.get_wrapped_class(PickUpAction)
    wrapped_navigate = class_diagram.get_wrapped_class(NavigateAction)
    wrapped_place = class_diagram.get_wrapped_class(PlaceAction)

    pickup_variables = parameterizer.parameterize(wrapped_pickup, prefix="PickUpAction")
    navigate_variables = parameterizer.parameterize(wrapped_navigate, prefix="NavigateAction")
    place_variables = parameterizer.parameterize(wrapped_place, prefix="PlaceAction")

    all_variables = pickup_variables + navigate_variables + place_variables
    variables = {v.name: v for v in all_variables}

    expected_variables = {
        "PickUpAction.arm",
        "PickUpAction.grasp_description.approach_direction",
        "PickUpAction.grasp_description.vertical_alignment",
        "PickUpAction.grasp_description.rotate_gripper",
        "NavigateAction.keep_joint_states",
        "NavigateAction.target_location.header.sequence",
        "NavigateAction.target_location.pose.position.x",
        "NavigateAction.target_location.pose.position.y",
        "NavigateAction.target_location.pose.position.z",
        "NavigateAction.target_location.pose.orientation.x",
        "NavigateAction.target_location.pose.orientation.y",
        "NavigateAction.target_location.pose.orientation.z",
        "NavigateAction.target_location.pose.orientation.w",
        "PlaceAction.arm",
        "PlaceAction.target_location.header.sequence",
        "PlaceAction.target_location.pose.position.x",
        "PlaceAction.target_location.pose.position.y",
        "PlaceAction.target_location.pose.position.z",
        "PlaceAction.target_location.pose.orientation.x",
        "PlaceAction.target_location.pose.orientation.y",
        "PlaceAction.target_location.pose.orientation.z",
        "PlaceAction.target_location.pose.orientation.w",
    }

    assert set(variables.keys()) == expected_variables

    probabilistic_distribution = parameterizer.create_fully_factorized_distribution(all_variables)

    expected_distribution = expected_variables - {"NavigateAction.target_location.header.sequence", "PlaceAction.target_location.header.sequence"}
    assert {v.name for v in probabilistic_distribution.variables} == expected_distribution

    arm_pickup = variables["PickUpAction.arm"]
    arm_place = variables["PlaceAction.arm"]

    arm_consistency_events = [SimpleEvent({arm_pickup: [arm], arm_place: [arm]}) for arm in Arms]
    restricted_dist, _ = probabilistic_distribution.truncated(Event(*arm_consistency_events))
    restricted_dist.normalize()

    nav_target_x = 2.0
    nav_target_y = 3.0
    pose_constraints = {
        variables["NavigateAction.target_location.pose.position.x"]: nav_target_x,
        variables["NavigateAction.target_location.pose.position.y"]: nav_target_y,
    }

    final_distribution, _ = restricted_dist.conditional(pose_constraints)
    final_distribution.normalize()

    v_nav_x = variables["NavigateAction.target_location.pose.position.x"]
    v_nav_y = variables["NavigateAction.target_location.pose.position.y"]

    for sample_values in final_distribution.sample(10):
        sample = dict(zip(final_distribution.variables, sample_values))
        assert sample[arm_pickup] == sample[arm_place]
        assert sample[v_nav_x] == nav_target_x
        assert sample[v_nav_y] == nav_target_y


