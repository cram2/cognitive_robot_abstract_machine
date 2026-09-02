from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Optional

from coraplex.config.action_conf import ActionConfig
from coraplex.datastructures.enums import Arms, MovementType
from coraplex.datastructures.grasp import GraspDescription
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.robots.robot_parts import Camera, EndEffector
from semantic_digital_twin.semantic_annotations.mixins import IsGraspable
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
)


@dataclass(eq=False)
class UsedArm:
    """
    Mixin for behaviours that operate one of the robot's arms.
    """

    arm: Arms = field(kw_only=True)
    """
    The arm the behaviour uses.
    """


@dataclass(eq=False)
class ObjectActedOn:
    """
    Mixin for behaviours that act on a single graspable object.
    """

    target_object: IsGraspable = field(kw_only=True)
    """
    The graspable annotation the behaviour acts on; its :attr:`root` body is used where the
    underlying kinematic body is required.
    """


@dataclass(eq=False)
class HandleOperatedOn:
    """
    Mixin for behaviours that grasp and articulate a handle, such as opening or closing a
    container.
    """

    handle: Handle = field(kw_only=True)
    """
    The handle annotation the behaviour operates; its :attr:`root` body is used where the
    underlying kinematic body is required.
    """


@dataclass(eq=False)
class UsedGraspDescription:
    """
    Mixin for behaviours that approach a body with a defined grasp.
    """

    grasp_description: GraspDescription = field(kw_only=True)
    """
    The grasp the behaviour uses to approach the body.
    """


@dataclass(eq=False)
class JointStatesKept:
    """
    Mixin for behaviours that can preserve the robot's joint states while moving the base.
    """

    keep_joint_states: bool = field(default=False, kw_only=True)
    """
    Whether the joint states are kept unchanged during the behaviour.
    """


@dataclass(eq=False)
class GripperCollisionAllowed:
    """
    Mixin for behaviours that may permit the gripper to collide with the environment.
    """

    allow_gripper_collision: Optional[bool] = field(default=None, kw_only=True)
    """
    Whether the gripper is allowed to collide during the behaviour.
    """


@dataclass(eq=False)
class TargetLocationMovedTo:
    """
    Mixin for behaviours that move the robot or an object to a destination pose.
    """

    target_location: Pose = field(kw_only=True)
    """
    The destination pose the behaviour moves to.
    """


@dataclass(eq=False)
class TargetPoseReached:
    """
    Mixin for behaviours that drive an end effector to a target pose.
    """

    target_pose: Pose = field(kw_only=True)
    """
    The pose the end effector reaches.
    """


@dataclass(eq=False)
class TargetLookedAt:
    """
    Mixin for behaviours that orient the robot toward a pose.
    """

    look_at_target: Pose = field(kw_only=True)
    """
    The pose the behaviour orients toward.
    """


@dataclass(eq=False)
class StandingPositionMovedTo:
    """
    Mixin for behaviours that first move the robot's base to a standing pose.
    """

    standing_position: Pose = field(kw_only=True)
    """
    The pose the robot stands at before manipulating.
    """


@dataclass(eq=False)
class UsedMovementType:
    """
    Mixin for behaviours whose Cartesian motion follows a selectable movement type.
    """

    movement_type: MovementType = field(default=MovementType.CARTESIAN, kw_only=True)
    """
    The type of Cartesian movement the behaviour performs.
    """


@dataclass(eq=False)
class GripperStateSet:
    """
    Mixin for behaviours that set the gripper to an open or closed state.
    """

    motion: GripperState = field(kw_only=True)
    """
    The gripper state the behaviour sets.
    """


@dataclass(eq=False)
class UsedEndEffector:
    """
    Mixin for behaviours that act through a specific end effector.
    """

    end_effector: EndEffector = field(kw_only=True)
    """
    The end effector the behaviour uses.
    """


@dataclass(eq=False)
class UsedCamera:
    """
    Mixin for behaviours that point a camera.
    """

    camera: Optional[Camera] = field(default=None, kw_only=True)
    """
    The camera the behaviour points; ``None`` selects the robot's default camera.
    """


@dataclass(eq=False)
class UsedTool:
    """
    Mixin for behaviours that manipulate an object with a held tool.
    """

    tool: SemanticAnnotation = field(kw_only=True)
    """
    The tool the behaviour uses.
    """


@dataclass(eq=False)
class UsedTechnique:
    """
    Mixin for behaviours that can be parametrised by a named technique.
    """

    technique: Optional[str] = field(default=None, kw_only=True)
    """
    The technique the behaviour applies.
    """


@dataclass(eq=False)
class UsedGraspingPreposeDistance:
    """
    Mixin for behaviours that approach a handle from a prepose offset before grasping.
    """

    grasping_prepose_distance: float = field(
        default=ActionConfig.grasping_prepose_distance, kw_only=True
    )
    """
    The distance in meters between the gripper and the handle before approaching to grasp.
    """


@dataclass(eq=False)
class PoseSequenceReversed:
    """
    Mixin for behaviours whose pose sequence can be reversed to move away instead of toward
    the target.
    """

    reverse_pose_sequence: bool = field(default=False, kw_only=True)
    """
    Whether the pose sequence is reversed.
    """


@dataclass(eq=False)
class TorsoStateSet:
    """
    Mixin for behaviours that set the torso to a defined state.
    """

    torso_state: TorsoState = field(kw_only=True)
    """
    The torso state the behaviour sets.
    """


@dataclass(eq=False)
class LinkAlignmentApplied:
    """
    Mixin for behaviours that can align an end-effector link with a goal axis.

    .. note:: The directional axes differ in representation between behaviours (axis identifier
        versus normal vector) and therefore stay declared on the concrete classes.
    """

    align: Optional[bool] = field(default=False, kw_only=True)
    """
    Whether the end effector is aligned with a goal axis.
    """

    tip_link: Optional[str] = field(default=None, kw_only=True)
    """
    The name of the tip link to align.
    """

    root_link: Optional[str] = field(default=None, kw_only=True)
    """
    The name of the root link to align against.
    """


################################################
######### Higher-order Parameter################
################################################


@dataclass(eq=False)
class ObjectManipulationParameters(ObjectActedOn, UsedArm):
    """
    Bundle of the parameters shared by every behaviour that manipulates an object with an arm:
    the object and the arm acting on it.
    """


@dataclass(eq=False)
class GraspParameters(ObjectManipulationParameters, UsedGraspDescription):
    """
    Bundle of the parameters for grasping an object: the object, the arm, and the grasp the
    arm approaches it with.
    """


@dataclass(eq=False)
class ToolUsageParameters(ObjectManipulationParameters, UsedTool, UsedTechnique):
    """
    Bundle of the parameters for acting on an object with a held tool: the object, the arm,
    the tool, and the technique.
    """


@dataclass(eq=False)
class MobileManipulationParameters(
    ObjectManipulationParameters, StandingPositionMovedTo, JointStatesKept
):
    """
    Bundle of the parameters for manipulating an object after first driving the base to a
    standing pose: the object, the arm, the standing pose, and whether joint states are kept.
    """


@dataclass(eq=False)
class HandleOperationParameters(HandleOperatedOn, UsedArm):
    """
    Bundle of the parameters for articulating a handle with an arm: the handle and the arm.
    """


@dataclass(eq=False)
class EndEffectorPoseParameters(
    UsedEndEffector, TargetPoseReached, GripperCollisionAllowed
):
    """
    Bundle of the parameters for driving an end effector to a target pose: the end effector,
    the target pose, and whether gripper collision is allowed.
    """


@dataclass(eq=False)
class CameraTargetParameters(UsedCamera, TargetLookedAt):
    """
    Bundle of the parameters for pointing a camera at a target: the camera and the pose it is
    pointed at.
    """


@dataclass(eq=False)
class NavigationParameters(TargetLocationMovedTo, JointStatesKept):
    """
    Bundle of the parameters for navigating the base to a destination: the destination and
    whether joint states are kept.
    """


@dataclass(eq=False)
class GripperActuationParameters(GripperStateSet, UsedArm):
    """
    Bundle of the parameters for setting a gripper to an open or closed state: the gripper
    state and the arm.
    """
