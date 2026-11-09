"""
The parameters motion statechart nodes share come from one mixin per concept, so that a
parameter has the same name and type in every node taking it.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil
import sys
from types import ModuleType

import pytest

from giskardpy.motion_statechart import goals, mixins, monitors, tasks
from giskardpy.motion_statechart.goals.base_pose_goals import (
    DrivenBaseGoal,
    HolonomicBaseGoal,
)
from giskardpy.motion_statechart.goals.cartesian_goals import CartesianPoseStraight
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    SelfCollisionAvoidance,
)
from giskardpy.motion_statechart.goals.gripper import (
    MoveFingers,
    MoveFingersToJointState,
    MoveGripper,
)
from giskardpy.motion_statechart.goals.grasped_mechanism import (
    ArticulateGraspedMechanism,
)
from giskardpy.motion_statechart.mixins import (
    GoalType,
    HasAngularVelocityLimitParameter,
    HasFingerStallToleranceParameters,
    HasGoalParameter,
    HasJointVelocityLimitParameter,
    HasKinematicChainParameters,
    HasLinearVelocityLimitParameter,
    HasPoseCruiseVelocityParameters,
    HasPoseThresholdParameters,
    HasCruiseVelocityParameter,
    HasRootLinkParameter,
    HasThresholdParameter,
    HasTipLinkParameter,
    HasTwoFingerGripperParameters,
    HasWeightParameter,
)
from giskardpy.motion_statechart.monitors.cartesian_monitors import VectorsAligned
from giskardpy.motion_statechart.monitors.feature_monitors import (
    AngleMonitor,
    DistanceMonitor,
    HeightMonitor,
    PerpendicularMonitor,
)
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianOrientation,
    CartesianPose,
    CartesianPosition,
    CartesianPositionTrajectory,
    CartesianPositionVelocityLimit,
    CartesianRotationVelocityLimit,
    CartesianVelocityLimit,
    DistanceAlongAxis,
    HoldOrientation,
    HoldPose,
)
from giskardpy.motion_statechart.tasks.feature_functions import (
    AlignPerpendicular,
    AngleGoal,
    DistanceGoal,
    HeightGoal,
    ReachPoint,
)
from giskardpy.motion_statechart.tasks.grasp_bar import GraspBar
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointPositionList,
    JointVelocityLimit,
)
from giskardpy.motion_statechart.tasks.pointing import Pointing, PointingCone
from giskardpy.motion_statechart.tasks.weight_scaling_goals import MaxManipulability
from giskardpy.motion_statechart.tasks.wiggle_insert import WiggleInsert
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from krrood_spatial_math.spatial_types import Point3, Pose, RotationMatrix, Vector3
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

# %% collecting the node classes


def _classes_defined_in(package: ModuleType) -> list[type]:
    """
    :param package: A package of motion statechart nodes.
    :return: Every dataclass defined in a module of `package`.
    """
    classes = []
    for module_info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{module_info.name}")
        classes.extend(
            cls
            for _, cls in inspect.getmembers(module, inspect.isclass)
            if cls.__module__ == module.__name__ and dataclasses.is_dataclass(cls)
        )
    return classes


def _node_classes() -> list[type]:
    """
    :return: Every dataclass among the tasks, goals and monitors.
    """
    return [
        cls
        for package in (tasks, goals, monitors)
        for cls in _classes_defined_in(package)
    ]


def _parameter_mixins() -> list[type]:
    """
    :return: Every parameter mixin.
    """
    return [
        cls
        for _, cls in inspect.getmembers(mixins, inspect.isclass)
        if cls.__module__ == mixins.__name__
    ]


def _declared_type(cls: type, field_name: str) -> object:
    """
    :param cls: A dataclass with a field named `field_name`.
    :param field_name: The name of the field.
    :return: The type of the field, resolved in the module of the class declaring it.
    """
    declaring_class = next(
        base
        for base in cls.__mro__
        if field_name in vars(base).get("__annotations__", {})
    )
    field_type = next(
        field.type for field in dataclasses.fields(cls) if field.name == field_name
    )
    if not isinstance(field_type, str):
        return field_type
    return eval(field_type, vars(sys.modules[declaring_class.__module__]))


def _feature_parameters(cls: type) -> dict[str, object]:
    """
    :param cls: A node class.
    :return: The type of every point and vector `cls` takes, by the name it takes it as.
    """
    return {
        field.name: _declared_type(cls, field.name)
        for field in dataclasses.fields(cls)
        if field.init and _declared_type(cls, field.name) in (Point3, Vector3)
    }


# %% one mixin per concept


def test_only_the_kinematic_chain_combines_parameter_mixins():
    combined_mixins = {
        mixin: {
            other
            for other in _parameter_mixins()
            if other is not mixin and issubclass(mixin, other)
        }
        for mixin in _parameter_mixins()
    }
    assert {mixin: others for mixin, others in combined_mixins.items() if others} == {
        HasKinematicChainParameters: {HasRootLinkParameter, HasTipLinkParameter}
    }


@pytest.mark.parametrize(
    "retired_name",
    [
        "goal_point",
        "goal_points",
        "goal_orientation",
        "goal_pose",
        "goal_state",
        "goal_joint_state",
        "goal_normal",
        "manipulability_threshold",
        "maximum_velocity",
        "down_velocity",
        "translation_threshold",
        "up_axis",
        "finger_velocity",
        "tolerate_stall",
        "stall_minimum_time",
    ],
)
def test_one_concept_has_one_name(retired_name: str):
    nodes_with_retired_name = [
        cls
        for package in (tasks, goals)
        for cls in _classes_defined_in(package)
        if retired_name in {field.name for field in dataclasses.fields(cls)}
    ]
    assert nodes_with_retired_name == []


@pytest.mark.parametrize(
    "node_class, mixin",
    [
        (CartesianPosition, HasCruiseVelocityParameter),
        (CartesianOrientation, HasCruiseVelocityParameter),
        (Pointing, HasCruiseVelocityParameter),
        (PointingCone, HasCruiseVelocityParameter),
        (JointPositionList, HasCruiseVelocityParameter),
        (AngleGoal, HasCruiseVelocityParameter),
        (WiggleInsert, HasCruiseVelocityParameter),
        (HoldOrientation, HasCruiseVelocityParameter),
        (ExternalCollisionAvoidance, HasCruiseVelocityParameter),
        (SelfCollisionAvoidance, HasCruiseVelocityParameter),
        (CartesianPose, HasPoseCruiseVelocityParameters),
        (CartesianPoseStraight, HasPoseCruiseVelocityParameters),
        (HoldPose, HasPoseCruiseVelocityParameters),
        (GraspBar, HasPoseCruiseVelocityParameters),
        (CartesianPositionVelocityLimit, HasLinearVelocityLimitParameter),
        (CartesianVelocityLimit, HasLinearVelocityLimitParameter),
        (CartesianRotationVelocityLimit, HasAngularVelocityLimitParameter),
        (CartesianVelocityLimit, HasAngularVelocityLimitParameter),
        (HoldOrientation, HasThresholdParameter),
        (DrivenBaseGoal, HasThresholdParameter),
        (MoveGripper, HasThresholdParameter),
        (MoveGripper, HasTwoFingerGripperParameters),
        (MoveFingers, HasFingerStallToleranceParameters),
        (MoveFingers, HasJointVelocityLimitParameter),
        (CartesianPose, HasPoseThresholdParameters),
        (HoldPose, HasPoseThresholdParameters),
        (CartesianPose, HasWeightParameter),
        (CartesianVelocityLimit, HasWeightParameter),
        (DrivenBaseGoal, HasWeightParameter),
    ],
)
def test_a_shared_parameter_comes_from_its_mixin(node_class: type, mixin: type):
    assert issubclass(node_class, mixin)


def test_a_joint_velocity_limit_names_its_cap_as_the_cartesian_limits_do():
    assert {field.name for field in dataclasses.fields(JointVelocityLimit)} >= {
        "max_joint_velocity"
    }
    assert "max_velocity" not in {
        field.name for field in dataclasses.fields(JointVelocityLimit)
    }


# %% links of the kinematic chain


def test_every_root_and_tip_link_is_a_kinematic_structure_entity():
    link_types = {
        (cls, field.name): _declared_type(cls, field.name)
        for cls in _node_classes()
        for field in dataclasses.fields(cls)
        if field.name in ("root_link", "tip_link")
    }
    assert {
        key: link_type
        for key, link_type in link_types.items()
        if link_type not in (KinematicStructureEntity, KinematicStructureEntity | None)
    } == {}


def test_every_node_taking_a_root_and_a_tip_link_takes_them_as_a_kinematic_chain():
    assert [
        cls
        for cls in _node_classes()
        if issubclass(cls, HasRootLinkParameter)
        and issubclass(cls, HasTipLinkParameter)
        and not issubclass(cls, HasKinematicChainParameters)
    ] == []


# %% goals and their reference frames


@pytest.mark.parametrize(
    "node_class, goal_type",
    [
        (CartesianPosition, Point3),
        (CartesianOrientation, RotationMatrix),
        (CartesianPositionTrajectory, list[Point3]),
        (Pointing, Point3),
        (PointingCone, Point3),
        (CartesianPose, Pose),
        (CartesianPoseStraight, Pose),
        (HolonomicBaseGoal, Pose),
        (JointPositionList, JointState),
        (MoveFingersToJointState, JointState),
        (ArticulateGraspedMechanism, float),
        (MaxManipulability, float),
    ],
)
def test_a_node_binds_the_type_of_its_goal(node_class: type, goal_type: object):
    assert issubclass(node_class, HasGoalParameter)
    assert node_class.get_type_of_generic_parameter(GoalType) == goal_type


def test_the_goal_reference_frame_is_the_frame_of_the_goal():
    goal_frame = Body(name=PrefixedName("goal_frame"))
    task = CartesianPosition(
        root_link=Body(name=PrefixedName("root")),
        tip_link=Body(name=PrefixedName("tip")),
        goal=Point3(reference_frame=goal_frame),
    )
    assert task.goal_reference_frame is goal_frame


def test_nodes_with_generic_parameters_are_told_apart_by_identity():
    root, tip = Body(name=PrefixedName("root")), Body(name=PrefixedName("tip"))
    goal = Point3(reference_frame=root)
    first = CartesianPosition(root_link=root, tip_link=tip, goal=goal)
    second = CartesianPosition(root_link=root, tip_link=tip, goal=goal)
    assert first != second
    assert len({first, second}) == 2


# %% pairs of geometric features


@pytest.mark.parametrize(
    "node_class, feature_parameters",
    [
        (AlignPlanes, {"tip_normal": Vector3, "reference_normal": Vector3}),
        (AlignPerpendicular, {"tip_normal": Vector3, "reference_normal": Vector3}),
        (AngleGoal, {"tip_vector": Vector3, "reference_vector": Vector3}),
        (DistanceGoal, {"tip_point": Point3, "reference_point": Point3}),
        (HeightGoal, {"tip_point": Point3, "reference_point": Point3}),
        (ReachPoint, {"tip_point": Point3, "reference_point": Point3}),
        (
            DistanceAlongAxis,
            {"tip_point": Point3, "reference_point": Point3, "axis": Vector3},
        ),
    ],
)
def test_a_node_takes_its_features_by_what_they_are(
    node_class: type, feature_parameters: dict[str, type]
):
    assert _feature_parameters(node_class) == feature_parameters


@pytest.mark.parametrize(
    "task, monitor",
    [
        (AlignPlanes, VectorsAligned),
        (AlignPerpendicular, PerpendicularMonitor),
        (AngleGoal, AngleMonitor),
        (DistanceGoal, DistanceMonitor),
        (HeightGoal, HeightMonitor),
    ],
)
def test_a_feature_task_takes_its_features_as_its_monitor_does(
    task: type, monitor: type
):
    assert _feature_parameters(task) == _feature_parameters(monitor)
