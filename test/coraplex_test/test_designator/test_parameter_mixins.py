"""
Tests that the coraplex actions take their parameters from one mixin per parameter.
"""

from __future__ import annotations

import dataclasses
import inspect
from types import ModuleType

import pytest

from coraplex.robot_plans import mixins
from coraplex.robot_plans.actions.base import Action
from coraplex.robot_plans.actions.core.container import OpenAction
from coraplex.robot_plans.actions.core.misc import MoveToReach
from coraplex.robot_plans.actions.core.navigation import (
    DrivesBase,
    NavigateAction,
    PathPlanningNavigateAction,
)
from coraplex.robot_plans.actions.core.pick_up import GraspingAction, PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveFingersToWidthAction,
    MoveManipulatorAction,
    ParkArmsAction,
    SetGripperAction,
)
from coraplex.robot_plans.mixins import (
    HasGraspApproach,
    HasApproachClearanceParameter,
    HasArmParameter,
    HasGraspCandidateParameter,
    HasGripperConfigurationParameter,
    HasPlacingCruiseLinearVelocityParameter,
    HasPlanarGoalPoseParameter,
    HasRetreatDistanceParameter,
    MovesToolCenterPoint,
)
from giskardpy.motion_statechart.tasks.drive_tasks import DrivenBaseGoal
from giskardpy.motion_statechart.tasks.gripper import MoveFingers, MoveGripper
from giskardpy.motion_statechart import mixins as node_mixins
from giskardpy.motion_statechart.mixins import (
    HasFingerStallMinimumTimeParameter,
    HasGoalPoseParameter,
    HasMaxJointVelocityParameter,
    HasOrientationThresholdParameter,
    HasPositionThresholdParameter,
    HasThresholdParameter,
    HasTolerateFingerStallParameter,
    HasTwoFingerGripperParameter,
    HasWidthParameter,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from krrood.class_diagrams.utils import get_type_hints_of_object
from krrood.code_generation.naming import to_snake_case
from krrood.patterns.mixin import ParameterMixin
from krrood_spatial_math.spatial_types import Pose3Dimensional
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk


def parameter_mixins(designator_type: type) -> set[type]:
    """
    :return: The parameter mixins the designator inherits.
    """
    return {
        base
        for base in designator_type.__mro__
        if base.__module__ == mixins.__name__
        and issubclass(base, ParameterMixin)
        and not issubclass(base, Action)
    }


def field_types(designator_type: type) -> dict:
    """
    :return: The declared type of each of the designator's dataclass fields, by name.
    """
    return {
        parameter.name: parameter.type
        for parameter in dataclasses.fields(designator_type)
    }


def field_names(designator_type: type) -> set[str]:
    """
    :return: The names of the designator's dataclass fields.
    """
    return {parameter.name for parameter in dataclasses.fields(designator_type)}


# %% one mixin per parameter


def mixins_defined_in(module: ModuleType) -> set[type]:
    """
    :return: The parameter mixins `module` defines, recognised by their names.
    """
    return {
        cls
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module.__name__
        and cls.__name__.startswith("Has")
        and cls.__name__.endswith(("Parameter", "Parameters"))
    }


def all_parameter_mixins() -> set[type]:
    """
    :return: The parameter mixins of the actions and of the motion statechart nodes.
    """
    return mixins_defined_in(mixins) | mixins_defined_in(node_mixins)


def parameter_mixins_named(suffix: str) -> list[type]:
    """
    :return: The parameter mixins whose names end in `suffix`, in a fixed order.
    """
    return sorted(
        (mixin for mixin in all_parameter_mixins() if mixin.__name__.endswith(suffix)),
        key=lambda mixin: mixin.__qualname__,
    )


def own_fields(mixin: type) -> set[str]:
    """
    :return: The names of the fields `mixin` declares itself.
    """
    return set(vars(mixin).get("__annotations__", {}))


def combined_mixins(mixin: type) -> set[type]:
    """
    :return: The other parameter mixins `mixin` inherits.
    """
    return {base for base in mixin.__mro__[1:] if base in all_parameter_mixins()}


def parameter_name(mixin: type) -> str:
    """
    :param mixin: A mixin named ``Has{Parameter}Parameter``.
    :return: The field name its class name spells, in snake case.
    """
    return to_snake_case(mixin.__name__.removeprefix("Has").removesuffix("Parameter"))


@pytest.mark.parametrize(
    "mixin",
    parameter_mixins_named("Parameter"),
)
def test_a_parameter_mixin_declares_the_one_field_it_is_named_for(mixin: type):
    """
    The name of a mixin spells the field it adds, and it adds nothing it inherits.
    """
    assert own_fields(mixin) == {parameter_name(mixin)}
    assert combined_mixins(mixin) == set()


def test_no_parameter_mixin_bundles_several_parameters():
    """
    Parameters that always come together are one parameter of a type holding them.
    """
    assert parameter_mixins_named("Parameters") == []


@pytest.mark.parametrize(
    "mixin",
    sorted(all_parameter_mixins(), key=lambda mixin: mixin.__qualname__),
)
def test_every_parameter_mixin_is_a_parameter_mixin(mixin: type):
    """
    Every class in the mixin module named for a parameter is marked as a parameter
    mixin.
    """
    assert issubclass(mixin, ParameterMixin)


def test_action_inherits_parameter_mixins():
    """
    A pick-up takes its arm, grasp, clearance and retreat from their mixins, and its
    approach from the grasp approach behaviour.
    """
    assert parameter_mixins(PickUpAction) >= {
        HasArmParameter,
        HasGraspCandidateParameter,
        HasApproachClearanceParameter,
        HasRetreatDistanceParameter,
    }
    assert issubclass(PickUpAction, HasGraspApproach)
    assert issubclass(PickUpAction, HasPositionThresholdParameter)


def test_pick_up_action_takes_its_grasp_and_arm(pr2_apartment_context):
    """
    A pick-up records the grasp and the arm it was given as its designator parameters.
    """
    world, robot, extensions = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    grasp = milk.grasp_candidates()[0]
    arm = robot.left_arm

    action = PickUpAction(grasp_candidate=grasp, arm=arm)

    parameters = action.designator_parameter
    assert parameters["arm"] is arm
    assert parameters["grasp_candidate"] is grasp


def test_open_action_takes_hold_of_its_handle_by_a_grasp():
    """
    Opening takes hold of its handle by a grasp, with an arm.
    """
    assert parameter_mixins(OpenAction) >= {
        HasGraspCandidateParameter,
        HasArmParameter,
    }


@pytest.mark.parametrize(
    "action_type", [MoveManipulatorAction, MoveToReach, PlaceAction]
)
def test_an_action_moving_a_tool_center_point_takes_an_arm(action_type: type):
    """
    Each of them moves the tool center point of the arm it is given, rather than taking
    an end effector or working out an arm of its own.
    """
    assert issubclass(action_type, HasArmParameter)


def test_move_to_reach_takes_no_approach_it_would_not_use():
    """
    It moves straight to the tool frame goal of its grasp frame, which neither a
    clearance nor a retreat distance shapes.
    """
    assert not issubclass(MoveToReach, HasGraspApproach)


def test_set_gripper_action_takes_the_configuration_it_reaches():
    """
    Setting a gripper takes only the configuration the gripper is brought into.
    """
    assert parameter_mixins(SetGripperAction) == {HasGripperConfigurationParameter}


# %% runtime resolution of the inherited field types


def test_inherited_parameters_keep_their_declared_types():
    """
    The fields a designator inherits from the mixins carry the types the mixins declare
    as type objects, not as strings the designator's own module would have to resolve.
    """
    types = field_types(PlaceAction)

    assert types["arm"] is HasArmParameter.__annotations__["arm"]
    assert (
        types["placing_cruise_linear_velocity"]
        == HasPlacingCruiseLinearVelocityParameter.__annotations__[
            "placing_cruise_linear_velocity"
        ]
    )


# %% tuning carried by the behaviour it tunes


def test_behaviours_driving_a_tool_center_point_carry_their_own_tolerances():
    """
    The goal tolerances belong to driving an arm to a goal, so an action that does that
    has them without naming a second mixin.
    """
    assert issubclass(MovesToolCenterPoint, HasPositionThresholdParameter)
    assert issubclass(MovesToolCenterPoint, HasOrientationThresholdParameter)


def test_a_tool_center_point_comes_closer_to_its_goal_than_a_pose_goal_requires():
    """
    The tool center point is brought closer to its goal than a pose goal requires by
    default, so an approach does not stop short of a small object.
    """
    assert MovesToolCenterPoint.position_threshold < CartesianPose.position_threshold


def test_behaviours_without_a_tool_center_point_goal_have_no_tolerances():
    """
    Parking the arms and setting a gripper drive no tool center point, so they carry no
    goal tolerances.
    """
    for action_type in (ParkArmsAction, SetGripperAction):
        assert not issubclass(action_type, HasPositionThresholdParameter)
        assert "position_threshold" not in field_names(action_type)


def test_navigation_actions_take_the_same_goal():
    """
    Driving the base, whether straight or along a planned path, takes the same goal.
    """
    for navigation in (NavigateAction, PathPlanningNavigateAction):
        assert issubclass(navigation, HasPlanarGoalPoseParameter)
        assert (
            field_types(navigation)["planar_goal_pose"]
            == HasPlanarGoalPoseParameter.__annotations__["planar_goal_pose"]
        )


# %% parameters shared with the motion statechart nodes


@pytest.mark.parametrize(
    "action_type, node_type, mixin",
    [
        (MoveFingersToWidthAction, MoveGripper, HasTwoFingerGripperParameter),
        (MoveFingersToWidthAction, MoveGripper, HasWidthParameter),
        (MoveFingersToWidthAction, MoveGripper, HasThresholdParameter),
        (PickUpAction, MoveFingers, HasTolerateFingerStallParameter),
        (PickUpAction, MoveFingers, HasFingerStallMinimumTimeParameter),
        (GraspingAction, MoveFingers, HasTolerateFingerStallParameter),
        (GraspingAction, MoveFingers, HasFingerStallMinimumTimeParameter),
        (ParkArmsAction, MoveFingers, HasMaxJointVelocityParameter),
        (DrivesBase, DrivenBaseGoal, HasThresholdParameter),
        (MoveManipulatorAction, CartesianPose, HasGoalPoseParameter),
    ],
)
def test_an_action_shares_the_parameter_mixin_of_the_node_it_builds(
    action_type: type, node_type: type, mixin: type
):
    """
    An action and the node it passes a parameter to take that parameter from the same
    mixin.
    """
    assert issubclass(action_type, mixin)
    assert issubclass(node_type, mixin)


def test_parameters_inherited_from_node_mixins_resolve_to_their_types():
    """
    The node mixins declare their types as strings, which resolve against the module of
    the mixin rather than that of the action.
    """
    hints = get_type_hints_of_object(PickUpAction)
    assert hints["position_threshold"] is float
    assert hints["tolerate_finger_stall"] is bool
    assert get_type_hints_of_object(PlaceAction)["goal_pose"] is Pose3Dimensional


def test_a_place_takes_its_goal_as_a_pose_goal_does():
    """
    Where a place puts the object down is the same kind of goal a pose goal drives to.
    """
    assert issubclass(PlaceAction, HasGoalPoseParameter)
