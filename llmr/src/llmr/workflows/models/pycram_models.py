"""Pydantic models for PyCRAM action designators and robot actions."""

from dataclasses import field
from enum import Enum, auto
from typing_extensions import Annotated, Dict, List, Literal, Optional, Type, Union

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class Arms(int, Enum):
    """Enum for Arms."""

    LEFT = 0
    RIGHT = 1
    BOTH = 2


class GripperState(Enum):
    """Enum for the different motions of the gripper."""

    OPEN = auto()
    CLOSE = auto()


class TorsoState(Enum):
    """Enum for the different states of the torso."""

    HIGH = auto()
    MID = auto()
    LOW = auto()


class Grasp(str, Enum):
    FRONT = "front"
    TOP = "top"


class DetectionTechnique(int, Enum):
    """Enum for techniques for detection tasks."""

    ALL = 0
    HUMAN = 1
    TYPES = 2
    REGION = 3
    HUMAN_ATTRIBUTES = 4
    HUMAN_WAVING = 5


class DetectionState(int, Enum):
    """Enum for the state of the detection task."""

    START = 0
    STOP = 1
    PAUSE = 2


# ── Common Models ──────────────────────────────────────────────────────────────

class PoseStampedModel(BaseModel):
    position: List[float]  # [x, y, z]
    orientation: Optional[List[float]] = None  # [x, y, z, w]


class GraspDescription(BaseModel):
    approach_direction: Grasp = Field(
        description="The primary approach direction. Must be one of {Grasp.FRONT, Grasp.TOP}."
    )
    vertical_alignment: Optional[Grasp] = Field(
        default=None,
        description="The vertical alignment when grasping the pose, or None if not applicable.",
    )
    rotate_gripper: bool = Field(
        default=False, description="Indicates if the gripper should be rotated by 90 degrees."
    )


class Vector3(BaseModel):
    x: float = 0
    y: float = 0
    z: float = 0

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]


class Quaternion(BaseModel):
    x: float = 0
    y: float = 0
    z: float = 0
    w: float = 1

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.w]


class Header(BaseModel):
    frame_id: str = "map"
    stamp: Optional[str] = None
    sequence: int = 0


class Pose(BaseModel):
    position: Vector3 = field(default_factory=Vector3)
    orientation: Quaternion = field(default_factory=Quaternion)

    def to_list(self) -> List:
        return [self.position.to_list(), self.orientation.to_list()]


class PoseStamped(BaseModel):
    pose: Pose = field(default_factory=Pose)
    header: Header = field(default_factory=Header)

    @property
    def position(self) -> Vector3:
        return self.pose.position

    @property
    def orientation(self) -> Quaternion:
        return self.pose.orientation

    @property
    def frame_id(self) -> str:
        return self.header.frame_id

    def to_list(self) -> List:
        return [self.pose.to_list(), self.frame_id]


class ObjectModel(BaseModel):
    name: str = Field(description="The name of the object.")
    concept: str = Field(description="Representing Type[PhysicalObject] as string")
    path: Optional[str] = Field(description="The path to the object.", default=None)
    pose: Optional[PoseStamped] = Field(description="The pose of the object.", default=None)
    world: Optional[str] = Field(
        description="The world of the object.", default=None
    )
    color: Optional[str] = Field(description="The color of the object.", default=None)
    ignore_cached_files: Optional[bool] = False
    scale_mesh: Optional[float] = 1.0
    mesh_transform: Optional[Dict[str, Union[List[float], str]]] = None


class Location(BaseModel):
    ...


class PhysicalObject(BaseModel):
    ...


class Agent(BaseModel):
    ...


class ActionDescription(BaseModel):
    """The performable designator with a single element for each list of possible parameters."""

    robot_position: Optional[PoseStampedModel] = Field(
        description="The position of the robot at the start of the action.", default=None
    )
    robot_torso_height: Optional[float] = Field(
        description="The torso height of the robot at the start of the action.", default=None
    )
    robot_type: Optional[Type[Agent]] = Field(
        description="The type of the robot at the start of the action.", default=None
    )


# ── Action Models ──────────────────────────────────────────────────────────────

class MoveTorsoAction(ActionDescription):
    """Move the torso of the robot up and down."""

    action_type: str = "MoveTorsoAction"
    torso_state: TorsoState = Field(description="The state of the torso that should be set")


class SetGripperAction(ActionDescription):
    """Set the gripper state of the robot."""

    action_type: str = "SetGripperAction"
    gripper: Arms = Field(description="The gripper that should be set")
    motion: GripperState = Field(description="The motion that should be set on the gripper")


class GripAction(ActionDescription):
    """Grip an object with the robot."""

    action_type: str = "GripAction"
    object_designator: ObjectModel = Field(description="The object that should be gripped")
    gripper: Arms = Field(description="The gripper that should be used to grip the object")
    effort: float = Field(description="The effort that should be used to grip the object")


class ParkArmsActionDescription(ActionDescription):
    """Park the arms of the robot."""

    action_type: Literal["ParkArmsAction"] = "ParkArmsAction"
    arm: Arms = Field(description="Entry from the enum for which arm should be parked")


class NavigateActionDescription(ActionDescription):
    """Navigate the Robot to a position."""

    action_type: Literal["NavigateAction"] = "NavigateAction"
    target_location: PoseStampedModel = Field(
        description="Location to which the robot should be navigated"
    )
    keep_joint_states: Optional[bool] = Field(
        default=True,
        description="Keep the joint states of the robot the same during the navigation.",
    )


class PickUpActionDescription(ActionDescription):
    """Let the robot pick up an object."""

    action_type: Literal["PickUpAction"] = "PickUpAction"
    object_designator: ObjectModel = Field(
        description="Object designator describing the object that should be picked up"
    )
    arm: Arms = Field(description="The arm that should be used for pick up")
    grasp_description: GraspDescription = Field(
        description="The GraspDescription that should be used for picking up the object"
    )


class PlaceActionDescription(ActionDescription):
    """Place an Object at a position using an arm."""

    action_type: Literal["PlaceAction"] = "PlaceAction"
    object_designator: ObjectModel = Field(
        description="Object designator describing the object that should be placed"
    )
    target_location: PoseStampedModel = Field(
        description="Pose in the world at which the object should be placed"
    )
    arm: Arms = Field(description="Arm that is currently holding the object")


class TransportActionDescription(ActionDescription):
    """Transport an object to a position using an arm."""

    action_type: Literal["TransportAction"] = "TransportAction"
    object_designator: ObjectModel = Field(
        description="Object designator describing the object that should be transported."
    )
    target_location: PoseStampedModel = Field(
        description="Target Location to which the object should be transported"
    )
    arm: Arms = Field(description="Arm that should be used")


class ReachToPickUpAction(ActionDescription):
    """Let the robot reach a specific pose."""

    action_type: str = "ReachToPickUpAction"
    object_designator: ObjectModel = Field(
        description="Object designator describing the object that should be picked up"
    )
    arm: Arms = Field(description="The arm that should be used for pick up")
    grasp_description: GraspDescription = Field(
        description="The grasp description that should be used for picking up the object"
    )


class LookAtAction(ActionDescription):
    """Lets the robot look at a position."""

    action_type: str = "LookAtAction"
    target: PoseStampedModel = Field(
        description="Position at which the robot should look, given as 6D pose"
    )


class OpenAction(ActionDescription):
    """Opens a container like object."""

    action_type: str = "OpenAction"
    object_designator: ObjectModel = Field(
        description="Object designator describing the object that should be opened"
    )
    arm: Arms = Field(description="Arm that should be used for opening the container")
    grasping_prepose_distance: float = Field(
        description="The distance in meters the gripper should be at in the x-axis away from the handle."
    )


class CloseAction(ActionDescription):
    """Closes a container like object."""

    action_type: str = "CloseAction"
    object_designator: ObjectModel = Field(
        description="Object designator describing the object that should be closed"
    )
    arm: Arms = Field(description="Arm that should be used for closing")
    grasping_prepose_distance: Optional[float] = Field(
        description="The distance in meters between the gripper and the handle before approaching to grasp."
    )


class GraspingAction(ActionDescription):
    """Grasps an object described by the given Object Designator description."""

    action_type: str = "GraspingAction"
    object_designator: ObjectModel = Field(
        description="Object Designator for the object that should be grasped"
    )
    arm: Arms = Field(description="Arm that should be used for grasping")
    prepose_distance: float = Field(
        description="The distance in meters the gripper should be at before grasping the object"
    )


class MoveAndPickUpAction(ActionDescription):
    """Navigate to standing_position, then turn towards the object and pick it up."""

    action_type: str = "MoveAndPickUpAction"
    standing_position: PoseStampedModel = Field(
        description="The pose to stand before trying to pick up the object"
    )
    object_designator: ObjectModel = Field(description="The object to pick up")
    arm: Arms = Field(description="The arm to use")
    grasp: Grasp = Field(description="The grasp to use")
    keep_joint_states: bool = Field(
        description="Keep the joint states of the robot the same during the navigation."
    )


class MoveAndPlaceAction(ActionDescription):
    """Navigate to standing_position, then place the object."""

    action_type: str = "MoveAndPlaceAction"
    standing_position: PoseStampedModel = Field(
        description="The pose to stand before trying to place the object"
    )
    object_designator: ObjectModel = Field(description="The object to place")
    target_location: PoseStampedModel = Field(description="The location to place the object.")
    arm: Arms = Field(description="The arm to use")
    keep_joint_states: bool = Field(
        description="Keep the joint states of the robot the same during the navigation."
    )


class FaceAtAction(ActionDescription):
    """Turn the robot chassis such that it faces the pose and after that perform a look at action."""

    action_type: str = "FaceAtAction"
    pose: PoseStampedModel = Field(description="The pose to face")
    keep_joint_states: bool = Field(
        description="Keep the joint states of the robot the same during the navigation."
    )


class DetectAction(ActionDescription):
    """Detects an object that fits the object description."""

    action_type: str = "DetectAction"
    technique: DetectionTechnique = Field(
        description="The technique that should be used for detection"
    )
    state: DetectionState = Field(
        description="The state of the detection, e.g Start Stop for continues perception",
        default=None,
    )
    object_designator: Optional[ObjectModel] = Field(
        default=None,
        description="The type of the object that should be detected, only considered if technique is equal to Type",
    )
    region: Location = Field(
        default=None,
        description="The region in which the object should be detected",
    )


class SearchAction(ActionDescription):
    """Searches for a target object around the given location."""

    action_type: str = "SearchAction"
    target_location: PoseStampedModel = Field(
        description="Location around which to look for a target object."
    )
    object_type: str = Field(
        description="SOMA - PhysicalObject concept of the object which is searched for."
    )


# ── PyCRAM models ──────────────────────────────────────────────────────────────

class ActionNames(BaseModel):
    model_names: List[
        Literal[
            "PickUpActionDescription",
            "PlaceActionDescription",
            "NavigateAction",
            "ParkArmsAction",
            "TransportActionDescription",
        ]
    ] = Field(description="Action model names")


class Actions(BaseModel):
    models: List[
        Annotated[
            Union[
                TransportActionDescription,
                PickUpActionDescription,
                PlaceActionDescription,
                NavigateActionDescription,
                ParkArmsActionDescription,
            ],
            Field(discriminator="action_type"),
        ]
    ]


class GroundedCramPlans(BaseModel):
    grounded_plans: List[str] = Field(
        description="List of grounded CRAM plans with tagged entities replaced by concrete entity references"
    )
