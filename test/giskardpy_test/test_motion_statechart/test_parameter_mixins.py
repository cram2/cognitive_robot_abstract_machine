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

from giskardpy.motion_statechart import mixins, monitors, tasks
from giskardpy.motion_statechart.tasks.drive_tasks import (
    BaseHeading,
    BaseOrientation,
    BaseTurnTowardsGoalPosition,
    DrivenBaseGoal,
    HolonomicBaseGoal,
)
from giskardpy.motion_statechart.tasks.collision_avoidance import (
    ExternalCollisionAvoidance,
    SelfCollisionAvoidance,
)
from giskardpy.motion_statechart.tasks.gripper import (
    MoveFingers,
    MoveFingersToJointState,
    MoveGripper,
)
from giskardpy.motion_statechart.tasks.grasped_mechanism import (
    ArticulateGraspedMechanism,
)
from giskardpy.motion_statechart.mixins import (
    HasTolerateFingerStallParameter,
    HasFingerStallMinimumTimeParameter,
    HasGoalBindingPolicyParameter,
    HasGoalOrientationParameter,
    HasGoalPointParameter,
    HasGoalPointsParameter,
    HasGoalPoseParameter,
    HasGoalStateParameter,
    HasMaxJointVelocityParameter,
    HasKinematicChainParameter,
    HasCruiseLinearVelocityParameter,
    HasCruiseAngularVelocityParameter,
    HasPositionThresholdParameter,
    HasOrientationThresholdParameter,
    HasCruiseVelocityParameter,
    HasThresholdParameter,
    HasTwoFingerGripperParameter,
    HasWidthParameter,
    HasWeightParameter,
    HasTipNormalParameter,
    HasReferenceNormalParameter,
    HasTipPointParameter,
    HasReferencePointParameter,
    HasTipVectorParameter,
    HasReferenceVectorParameter,
    HasLowerLimitParameter,
    HasUpperLimitParameter,
    HasLowerAngleParameter,
    HasUpperAngleParameter,
    HasPointingAxisParameter,
    HasHolePointParameter,
    HasHoleNormalParameter,
)
from giskardpy.motion_statechart.forward_kinematics_binding import (
    ForwardKinematicsBinding,
)
from giskardpy.motion_statechart.monitors.cartesian_monitors import (
    OrientationReached,
    PointingAt,
    PoseReached,
    PositionReached,
    VectorsAligned,
)
from giskardpy.motion_statechart.monitors.feature_monitors import (
    AngleMonitor,
    DistanceMonitor,
    HeightMonitor,
    PerpendicularMonitor,
)
from giskardpy.motion_statechart.tasks.align_planes import (
    AlignPlanes,
    KeepHeldBodyUpright,
)
from giskardpy.motion_statechart.tasks.cylinder_insertion import InsertCylinder
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianOrientation,
    CartesianPose,
    CartesianPoseStraight,
    CartesianPosition,
    CartesianPositionTrajectory,
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
from semantic_digital_twin.datastructures.alignment import AlignmentPair
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from krrood_spatial_math.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)
from semantic_digital_twin.world_description.kinematic_chain import KinematicChain

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
    :return: Every dataclass among the tasks and monitors.
    """
    return [
        cls for package in (tasks, monitors) for cls in _classes_defined_in(package)
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
    :param cls: A dataclass.
    :return: The type of every point and vector `cls` takes, by the name it takes it as.
    """
    return {
        field.name: _declared_type(cls, field.name)
        for field in dataclasses.fields(cls)
        if field.init and _declared_type(cls, field.name) in (Point3, Vector3)
    }


# %% one mixin per concept


def test_no_parameter_mixin_combines_another():
    """
    A parameter mixin declares one parameter, so none inherits another; a class needing
    several inherits each of them.
    """
    combined_mixins = {
        mixin: {
            other
            for other in _parameter_mixins()
            if other is not mixin and issubclass(mixin, other)
        }
        for mixin in _parameter_mixins()
    }
    assert {mixin: others for mixin, others in combined_mixins.items() if others} == {}


@pytest.mark.parametrize(
    "retired_name",
    [
        "goal",
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
        "binding_policy",
        "gripper",
    ],
)
def test_one_concept_has_one_name(retired_name: str):
    """
    A name retired for the one every node now uses is no field of any node.
    """
    nodes_with_retired_name = [
        cls
        for cls in _node_classes()
        if retired_name in {field.name for field in dataclasses.fields(cls)}
    ]
    assert nodes_with_retired_name == []


@pytest.mark.parametrize(
    "node_class, mixin",
    [
        (CartesianPosition, HasCruiseVelocityParameter),
        (CartesianOrientation, HasCruiseAngularVelocityParameter),
        (Pointing, HasCruiseAngularVelocityParameter),
        (PointingCone, HasCruiseAngularVelocityParameter),
        (JointPositionList, HasCruiseVelocityParameter),
        (AngleGoal, HasCruiseAngularVelocityParameter),
        (WiggleInsert, HasCruiseVelocityParameter),
        (HoldOrientation, HasCruiseAngularVelocityParameter),
        (ExternalCollisionAvoidance, HasCruiseVelocityParameter),
        (SelfCollisionAvoidance, HasCruiseVelocityParameter),
        (CartesianPose, HasCruiseLinearVelocityParameter),
        (CartesianPose, HasCruiseAngularVelocityParameter),
        (CartesianPoseStraight, HasCruiseLinearVelocityParameter),
        (CartesianPoseStraight, HasCruiseAngularVelocityParameter),
        (HoldPose, HasCruiseLinearVelocityParameter),
        (HoldPose, HasCruiseAngularVelocityParameter),
        (GraspBar, HasCruiseLinearVelocityParameter),
        (GraspBar, HasCruiseAngularVelocityParameter),
        (HoldOrientation, HasThresholdParameter),
        (DrivenBaseGoal, HasThresholdParameter),
        (MoveGripper, HasThresholdParameter),
        (MoveGripper, HasTwoFingerGripperParameter),
        (MoveGripper, HasWidthParameter),
        (MoveFingers, HasTolerateFingerStallParameter),
        (MoveFingers, HasFingerStallMinimumTimeParameter),
        (MoveFingers, HasMaxJointVelocityParameter),
        (CartesianPose, HasPositionThresholdParameter),
        (CartesianPose, HasOrientationThresholdParameter),
        (HoldPose, HasPositionThresholdParameter),
        (HoldPose, HasOrientationThresholdParameter),
        (CartesianPose, HasWeightParameter),
        (DrivenBaseGoal, HasWeightParameter),
    ],
)
def test_a_shared_parameter_comes_from_its_mixin(node_class: type, mixin: type):
    """
    A node taking a shared parameter inherits the mixin declaring it instead of
    declaring a copy of its own.
    """
    assert issubclass(node_class, mixin)


def _mixin_declaring(field_name: str) -> type | None:
    """
    :return: The parameter mixin declaring a field named `field_name`, if any does.
    """
    return next(
        (
            mixin
            for mixin in _parameter_mixins()
            if field_name in vars(mixin).get("__annotations__", {})
        ),
        None,
    )


def test_every_parameter_a_mixin_declares_comes_from_that_mixin():
    """
    A node taking a parameter under the name a mixin gives it means what that mixin
    means by it.
    """
    assert [
        (cls, field.name)
        for cls in _node_classes()
        for field in dataclasses.fields(cls)
        if _mixin_declaring(field.name) is not None
        and not issubclass(cls, _mixin_declaring(field.name))
    ] == []


@pytest.mark.parametrize(
    "node_class",
    [
        AlignPlanes,
        KeepHeldBodyUpright,
        AlignPerpendicular,
        AngleGoal,
        CartesianOrientation,
        HoldOrientation,
        Pointing,
        PointingCone,
        BaseOrientation,
        BaseHeading,
        BaseTurnTowardsGoalPosition,
    ],
)
def test_a_task_with_an_error_in_radians_approaches_its_goal_at_an_angular_velocity(
    node_class: type,
):
    """
    A task whose error is an angle takes its cruise velocity in rad/s, never the m/s
    :attr:`cruise_velocity`.
    """
    assert issubclass(node_class, HasCruiseAngularVelocityParameter)
    assert not issubclass(node_class, HasCruiseVelocityParameter)


def test_a_forward_kinematics_binding_takes_its_chain_from_its_mixin():
    """
    A forward kinematics binding is told its chain the way the nodes are.
    """
    assert issubclass(ForwardKinematicsBinding, HasKinematicChainParameter)


def test_a_joint_velocity_limit_takes_its_cap_as_a_max_joint_velocity():
    """
    A joint velocity limit takes its cap as ``max_joint_velocity``, not under the
    retired ``max_velocity``.
    """
    assert {field.name for field in dataclasses.fields(JointVelocityLimit)} >= {
        "max_joint_velocity"
    }
    assert "max_velocity" not in {
        field.name for field in dataclasses.fields(JointVelocityLimit)
    }


# %% links of the kinematic chain


def test_every_root_and_tip_link_is_a_kinematic_structure_entity():
    """
    A link a node is still given apart from a chain is typed as any entity of the
    kinematic structure, not as a body.
    """
    link_types = {
        (cls, field.name): _declared_type(cls, field.name)
        for cls in _node_classes()
        for field in dataclasses.fields(cls)
        if field.name in ("root_link", "tip_link")
    }
    assert {
        key: link_type
        for key, link_type in link_types.items()
        if link_type is not KinematicStructureEntity
    } == {}


def test_no_node_takes_a_root_and_a_tip_link_apart():
    """
    No node takes a root and a tip link as two parameters; it takes a kinematic chain.
    """
    assert [
        cls
        for cls in _node_classes()
        if {"root_link", "tip_link"}
        <= {field.name for field in dataclasses.fields(cls) if field.init}
    ] == []


def test_every_kinematic_chain_comes_from_its_mixin():
    """
    Every node taking a kinematic chain inherits it from its mixin.
    """
    assert [
        cls
        for cls in _node_classes()
        if "kinematic_chain" in {field.name for field in dataclasses.fields(cls)}
        and not issubclass(cls, HasKinematicChainParameter)
    ] == []


# %% goals and their reference frames


@pytest.mark.parametrize(
    "node_class, mixin",
    [
        (CartesianPosition, HasGoalPointParameter),
        (Pointing, HasGoalPointParameter),
        (PointingCone, HasGoalPointParameter),
        (CartesianOrientation, HasGoalOrientationParameter),
        (CartesianPositionTrajectory, HasGoalPointsParameter),
        (CartesianPose, HasGoalPoseParameter),
        (CartesianPoseStraight, HasGoalPoseParameter),
        (HolonomicBaseGoal, HasGoalPoseParameter),
        (JointPositionList, HasGoalStateParameter),
        (MoveFingersToJointState, HasGoalStateParameter),
    ],
)
def test_a_goal_comes_from_the_mixin_of_its_kind(node_class: type, mixin: type):
    """
    A node names its goal for what kind of goal it is, as every node of that kind does.
    """
    assert issubclass(node_class, mixin)


def _goal_mixins_of(node_class: type) -> set[type]:
    """
    :return: The mixins of the goals `node_class` takes.
    """
    return {
        mixin
        for mixin in (
            HasGoalPointParameter,
            HasGoalPointsParameter,
            HasGoalOrientationParameter,
            HasGoalPoseParameter,
            HasGoalStateParameter,
        )
        if issubclass(node_class, mixin)
    }


@pytest.mark.parametrize(
    "task, monitor",
    [
        (CartesianPose, PoseReached),
        (CartesianPosition, PositionReached),
        (CartesianOrientation, OrientationReached),
        (Pointing, PointingAt),
    ],
)
def test_a_monitor_takes_its_goal_as_its_task_does(task: type, monitor: type):
    """
    A monitor observing whether a task's goal is reached is given that goal the same
    way.
    """
    assert _goal_mixins_of(monitor) == _goal_mixins_of(task)


@pytest.mark.parametrize(
    "task, monitor",
    [
        (CartesianPose, PoseReached),
        (CartesianPosition, PositionReached),
        (CartesianOrientation, OrientationReached),
    ],
)
def test_a_monitor_reads_its_goal_frame_as_its_task_does(task: type, monitor: type):
    """
    A monitor reads a goal frame that may move when its task does.
    """
    assert issubclass(monitor, HasGoalBindingPolicyParameter) == issubclass(
        task, HasGoalBindingPolicyParameter
    )


def test_a_manipulability_goal_names_what_it_drives_towards():
    """
    The value is the manipulability the task drives towards, not a threshold.
    """
    assert "goal_manipulability" in {
        field.name for field in dataclasses.fields(MaxManipulability)
    }


def test_a_grasped_mechanism_names_the_position_it_drives_to():
    """
    The value is a position of the degree of freedom, not a joint state.
    """
    assert "goal_position" in {
        field.name for field in dataclasses.fields(ArticulateGraspedMechanism)
    }


def test_the_goal_reference_frame_is_the_frame_of_the_goal():
    """
    A task's goal reference frame is the frame its goal is expressed in, not the root of
    its chain.
    """
    goal_frame = Body(name=PrefixedName("goal_frame"))
    task = CartesianPosition(
        kinematic_chain=KinematicChain(
            root=Body(name=PrefixedName("root")), tip=Body(name=PrefixedName("tip"))
        ),
        goal_point=Point3(reference_frame=goal_frame),
    )
    assert task.goal_reference_frame is goal_frame


def test_nodes_with_equal_parameters_are_told_apart_by_identity():
    """
    Two nodes built from equal parameters are still two nodes, in comparisons and in
    sets.
    """
    root, tip = Body(name=PrefixedName("root")), Body(name=PrefixedName("tip"))
    goal = Point3(reference_frame=root)
    first = CartesianPosition(
        kinematic_chain=KinematicChain(root=root, tip=tip), goal_point=goal
    )
    second = CartesianPosition(
        kinematic_chain=KinematicChain(root=root, tip=tip), goal_point=goal
    )
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
    """
    A node names each geometric feature it takes after the kind of feature, such as a
    point or a normal, and types it accordingly.
    """
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
    """
    A feature task and the monitor observing the same relation take the same features
    under the same names.
    """
    assert _feature_parameters(task) == _feature_parameters(monitor)


def test_an_alignment_pair_names_its_normals_as_align_planes_takes_them():
    """
    An alignment pair names its normals as :class:`AlignPlanes` does, so one can be
    handed on to the other.
    """
    assert _feature_parameters(AlignmentPair) == _feature_parameters(AlignPlanes)


@pytest.mark.parametrize(
    "node_class, mixin",
    [
        (AlignPlanes, HasTipNormalParameter),
        (AlignPlanes, HasReferenceNormalParameter),
        (AlignPerpendicular, HasTipNormalParameter),
        (AlignPerpendicular, HasReferenceNormalParameter),
        (VectorsAligned, HasTipNormalParameter),
        (VectorsAligned, HasReferenceNormalParameter),
        (PerpendicularMonitor, HasTipNormalParameter),
        (PerpendicularMonitor, HasReferenceNormalParameter),
        (AngleGoal, HasTipVectorParameter),
        (AngleGoal, HasReferenceVectorParameter),
        (AngleGoal, HasLowerAngleParameter),
        (AngleGoal, HasUpperAngleParameter),
        (AngleMonitor, HasTipVectorParameter),
        (AngleMonitor, HasReferenceVectorParameter),
        (AngleMonitor, HasLowerAngleParameter),
        (AngleMonitor, HasUpperAngleParameter),
        (DistanceGoal, HasTipPointParameter),
        (DistanceGoal, HasReferencePointParameter),
        (DistanceGoal, HasLowerLimitParameter),
        (DistanceGoal, HasUpperLimitParameter),
        (DistanceMonitor, HasTipPointParameter),
        (DistanceMonitor, HasReferencePointParameter),
        (DistanceMonitor, HasLowerLimitParameter),
        (DistanceMonitor, HasUpperLimitParameter),
        (HeightGoal, HasTipPointParameter),
        (HeightGoal, HasReferencePointParameter),
        (HeightGoal, HasLowerLimitParameter),
        (HeightGoal, HasUpperLimitParameter),
        (HeightMonitor, HasTipPointParameter),
        (HeightMonitor, HasReferencePointParameter),
        (HeightMonitor, HasLowerLimitParameter),
        (HeightMonitor, HasUpperLimitParameter),
        (ReachPoint, HasTipPointParameter),
        (ReachPoint, HasReferencePointParameter),
        (DistanceAlongAxis, HasTipPointParameter),
        (DistanceAlongAxis, HasReferencePointParameter),
        (Pointing, HasPointingAxisParameter),
        (PointingCone, HasPointingAxisParameter),
        (PointingAt, HasPointingAxisParameter),
        (WiggleInsert, HasHolePointParameter),
        (InsertCylinder, HasHolePointParameter),
        (WiggleInsert, HasHoleNormalParameter),
        (InsertCylinder, HasHoleNormalParameter),
    ],
)
def test_a_feature_comes_from_its_mixin(node_class: type, mixin: type):
    """
    A node taking a geometric feature inherits the mixin declaring it.
    """
    assert issubclass(node_class, mixin)
