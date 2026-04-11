from __future__ import annotations

import logging
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Any

from typing_extensions import (
    TYPE_CHECKING,
    Optional,
    Self,
    List,
)

from krrood.class_diagrams.attribute_introspector import (
    DataclassOnlyIntrospector,
    DiscoveredAttribute,
)
from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.entity_query_language.factories import (
    variable,
    a,
    entity,
    contains,
)
from semantic_digital_twin.datastructures.definitions import JointStateType
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    NoJointStateWithType,
    DoesNotBelongToAWorldError,
    DuplicateRobotAssignmentsError,
)
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import (
    Vector3,
    Quaternion,
    RotationMatrix,
)
from semantic_digital_twin.world_description.connections import (
    ActiveConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import BoundingBox, Scale
from semantic_digital_twin.world_description.world_entity import (
    Body,
    Connection,
    WorldEntity,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)
from semantic_digital_twin.world_description.world_modification import (
    synchronized_attribute_modification,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World
    from semantic_digital_twin.robots.abstract_robot import AbstractRobot


logger = logging.getLogger(__name__)


@dataclass(eq=False)
class AggregatesRobotParts(ABC):
    @property
    def _robot_parts(self) -> list[RobotPart]:
        """
        Serves as a generic interface to access all robot parts assigned to a robot part.
        Returns a list of all robot parts assigned directly to this robot part.
        """
        wrapped_class = WrappedClass(self.__class__)
        introspector = DataclassOnlyIntrospector()
        robot_parts = []
        for field_ in introspector.discover(self.__class__):
            value = getattr(self, field_.public_name)
            wrapped_field = WrappedField(wrapped_class, field_.field)

            if isinstance(value, (list, set)) and issubclass(
                wrapped_field.contained_type, RobotPart
            ):
                robot_parts.extend(value)
                for robot_part in value:
                    robot_parts.extend(robot_part._robot_parts)
            elif isinstance(value, RobotPart):
                robot_parts.append(value)
                robot_parts.extend(value._robot_parts)

        return robot_parts


@dataclass(eq=False)
class RobotPart(HasRootBody, AggregatesRobotParts, ABC):
    """
    Represents a collection of connected robot bodies, starting from a root body, and ending in a unspecified collection
    of tip bodies.
    """

    joint_states: list[JointState] = field(default_factory=list)
    """
    Fixed joint states that are defined for this robot annotation. 
    """

    sensors: List[Sensor] = field(default_factory=list)
    """
    A collection of sensors in the kinematic chain, such as cameras or other sensors.
    """

    def __post_init__(self):
        introspector = DataclassOnlyIntrospector()
        for field_ in introspector.discover(self.__class__):
            value = getattr(self, field_.public_name)
            if isinstance(value, (list, set)):
                for v in value:
                    self._raise_if_not_in_world(v)
            else:
                self._raise_if_not_in_world(value)

    def _raise_if_not_in_world(self, object_of_interest: Any):
        if (
            isinstance(object_of_interest, WorldEntity)
            and object_of_interest._world is None
        ):
            raise DoesNotBelongToAWorldError(object_of_interest)

    @synchronized_attribute_modification
    def add_joint_state(self, joint_state: JointState):
        """
        Adds a joint state to this semantic annotation.
        """
        if not self.is_controlled:
            raise NotImplementedError(
                "Adding joint states is only supported for robot parts that can be controlled."
            )
        self.joint_states.append(joint_state)
        joint_state.assign_to_robot(self._robot)

    @synchronized_attribute_modification
    def add_sensors(self, sensors: List[Sensor]):
        self.sensors.extend(sensors)

    def get_joint_state_by_type(self, state_type: JointStateType) -> JointState:
        """
        Returns a JointState for a given joint state type.
        :param state_type: The state type to search for
        :return: The joint state with the given type
        """
        for j in self.joint_states:
            if j.state_type == state_type:
                return j
        raise NoJointStateWithType(state_type)

    @property
    def is_controlled(self) -> bool:
        return any((c for c in self.connections if c.is_controlled))

    @classmethod
    def create_with_new_body_in_world(
        cls,
        name: PrefixedName,
        world: World,
        world_root_T_self: Optional[HomogeneousTransformationMatrix] = None,
        connection_limits: Optional[DegreeOfFreedomLimits] = None,
        active_axis: Optional[Vector3] = None,
        connection_multiplier: float = 1.0,
        connection_offset: float = 0.0,
        scale: Scale = None,
        **kwargs,
    ) -> Self:
        raise NotImplementedError(
            "The bodies needed for RobotParts should already exist in the world, by parsing a URDF"
        )

    @classmethod
    @abstractmethod
    def create_and_add_to_world(
        cls,
        name: PrefixedName,
        world: World,
        **kwargs,
    ) -> Self:
        """
        Creates a new instance of the RobotPart and adds it to the world. The specific parameters needed to create the
        RobotPart are added in the actual implementation of the method.
        The primary purpose if this method is to allow the programmer to enforce some order in which the RobotParts are
        created and added to the world when the user implements a new AbstractRobot.
        """

    def _log_missing_fields(self):
        """
        Logs any fields that are empty, which could indicate missing information in the robot annotation.
        Primarily used for manual validation purposes.
        """
        wrapped_class = WrappedClass(self.__class__)
        introspector = DataclassOnlyIntrospector()
        for field_ in introspector.discover(self.__class__):
            self._process_field(wrapped_class, field_)

    def _process_field(self, wrapped_class: WrappedClass, field: DiscoveredAttribute):
        """
        Processes a single field of the dataclass, checking if it is empty, and logs a warning if it is.

        :param wrapped_class: The wrapped class of the dataclass.
        :param field: The discovered attribute of the dataclass.
        """
        value = getattr(self, field.public_name)
        wrapped_field = WrappedField(wrapped_class, field.field)
        type_endpoint = wrapped_field.type_endpoint

        if isinstance(value, (list, set)) and issubclass(
            wrapped_field.contained_type, RobotPart
        ):
            if not value:
                self._log_missing_field(field)
                return

            for robot_part in value:
                robot_part._log_missing_fields()

        elif issubclass(type_endpoint, RobotPart) and value is None:
            self._log_missing_field(field)

    def _log_missing_field(self, field: DiscoveredAttribute):
        logger.info(
            f"The field {field.public_name} of {self.__class__.__name__} is empty. Please confirm that this is intentional."
        )

    @property
    def _robot(self) -> Optional[AbstractRobot]:
        from semantic_digital_twin.robots.abstract_robot import AbstractRobot

        robot_variable = variable(AbstractRobot, self._world.semantic_annotations)
        robot = (
            a(entity(robot_variable))
            .where(contains(robot_variable._robot_parts, self))
            .tolist()
        )
        if len(robot) == 0:
            return None
        elif len(robot) > 1:
            raise DuplicateRobotAssignmentsError(robot_part=self, robots=robot)
        return robot[0]


@dataclass(eq=False)
class KinematicChain(RobotPart, ABC):
    """
    A kinematic chain in a robot, starting from a root body, and ending in a specific tip body.
    A kinematic chain can have multiple sensors. There are no assumptions about the
    position of the manipulator or sensors in the kinematic chain
    """

    tip: Body = field(kw_only=True)
    """
    The tip body of the kinematic chain, which is the last body in the chain.
    """

    @property
    def kinematic_structure_entities(self) -> list[KinematicStructureEntity]:
        """
        Returns itself as a kinematic chain of bodies.
        """
        kinematic_structure_entities = [
            entity
            for entity in self._world.compute_chain_of_kinematic_structure_entities(
                self.root, self.tip
            )
        ]

        for sensor in self.sensors:
            kinematic_structure_entities.extend(sensor.kinematic_structure_entities)

        return kinematic_structure_entities

    @property
    def connections(self) -> list[Connection]:
        """
        Returns the connections of the kinematic chain.
        This is a list of connections between the bodies in the kinematic chain
        """
        if self.root == self.tip:
            return [self.root.parent_connection]
        return self._world.compute_chain_of_connections(self.root, self.tip)

    @property
    def active_connections(self) -> list[ActiveConnection]:
        return [
            connection
            for connection in self.connections
            if isinstance(connection, ActiveConnection)
        ]


@dataclass(eq=False)
class Arm(KinematicChain):
    """
    Represents an arm of a robot, which is a kinematic chain with a manipulator.
    """

    manipulator: Optional[Manipulator] = field(default=None)
    """
    The manipulator of the kinematic chain, if it exists. This is usually a gripper or similar device.
    """

    @synchronized_attribute_modification
    def add_manipulator(self, manipulator: Manipulator):
        self.manipulator = manipulator

    @property
    def kinematic_structure_entities(self) -> list[KinematicStructureEntity]:
        """
        Returns itself as a kinematic chain of bodies.
        """
        kinematic_structure_entities = [
            entity
            for entity in self._world.compute_chain_of_kinematic_structure_entities(
                self.root, self.tip
            )
        ]
        if self.manipulator is not None:
            kinematic_structure_entities.extend(
                self.manipulator.kinematic_structure_entities
            )

        for sensor in self.sensors:
            kinematic_structure_entities.extend(sensor.kinematic_structure_entities)

        return kinematic_structure_entities

    @classmethod
    def create_and_add_to_world(
        cls,
        name: PrefixedName,
        world: World,
        root_name: str,
        tip_name: str,
        manipulator: Manipulator,
        sensors: List[Sensor] = None,
    ) -> Self:
        if manipulator._world is not world:
            raise ValueError(
                "The manipulator must be part of the given world, but it is not."
            )
        self = cls(
            name=name,
            root=world.get_body_by_name(root_name),
            tip=world.get_body_by_name(tip_name),
        )
        world.add_semantic_annotation(self)
        self.add_manipulator(manipulator)
        if sensors is not None:
            self.add_sensors(sensors)
        return self


@dataclass(eq=False)
class Manipulator(RobotPart, ABC):
    """
    Abstract base class of robot manipulators. Always has a tool frame.
    """

    tool_frame: Body = field(kw_only=True)
    """
    The tool frame or tool center point of the manipulator. Usually the point the robot tries to align with the object.
    """

    front_facing_orientation: Quaternion = field(kw_only=True)
    """
    The orientation of the manipulator's tool frame, which is usually the front-facing orientation.
    """

    front_facing_axis: Vector3 = field(init=False)
    """
    The axis of the manipulator's tool frame that is facing forward.
    """

    is_controlled = True

    def __post_init__(self):
        super().__post_init__()
        rotation_matrix = RotationMatrix.from_quaternion(self.front_facing_orientation)
        self.front_facing_axis = Vector3.from_iterable(rotation_matrix[:3, 0])


@dataclass(eq=False)
class Finger(KinematicChain):
    """
    A finger is a kinematic chain, since it should have an unambiguous tip body, and may contain sensors.
    """

    finger_tip_frame: Optional[Body] = None
    """
    The frame of the finger tip. Could be used to align the finger with, for example, a button.
    """

    @classmethod
    def create_and_add_to_world(
        cls,
        name: PrefixedName,
        world: World,
        root_name: str,
        tip_name: str,
        finger_tip_frame_name: Optional[str] = None,
        sensors: List[Sensor] = None,
    ) -> Self:
        finger_tip_frame = None
        if finger_tip_frame_name is not None:
            finger_tip_frame = world.get_body_by_name(finger_tip_frame_name)
        self = cls(
            name=name,
            root=world.get_body_by_name(root_name),
            tip=world.get_body_by_name(tip_name),
            finger_tip_frame=finger_tip_frame,
        )
        world.add_semantic_annotation(self)
        if sensors is not None:
            self.add_sensors(sensors)
        return self


@dataclass(eq=False)
class ParallelGripper(Manipulator):
    """
    Represents a parallel gripper of a robot. Contains a finger and a thumb. The thumb is a specific finger
    that always needs to touch an object when grasping it, ensuring a stable grasp.
    """

    thumb: Optional[Finger] = field(default=None)
    """
    The thumb of the parallel gripper, which is the part that always needs to touch an object when grasping it.
    """

    finger: Optional[Finger] = field(default=None)
    """
    The finger of the parallel gripper, which is the part that moves in parallel to the thumb to grasp objects.
    """

    @synchronized_attribute_modification
    def add_finger(self, finger: Finger):
        self.finger = finger

    @synchronized_attribute_modification
    def add_thumb(self, thumb: Finger):
        self.thumb = thumb

    @classmethod
    def create_and_add_to_world(
        cls,
        name: PrefixedName,
        world: World,
        root_name: str,
        tool_frame_name: str,
        front_facing_orientation: Quaternion,
        finger: Finger,
        thumb: Finger,
        sensors: List[Sensor] = None,
    ) -> Self:
        if finger._world is not world or thumb._world is not world:
            raise ValueError(
                "The finger and thumb must be part of the given world, but they are not."
            )
        self = cls(
            name=name,
            root=world.get_body_by_name(root_name),
            tool_frame=world.get_body_by_name(tool_frame_name),
            front_facing_orientation=front_facing_orientation,
        )
        world.add_semantic_annotation(self)
        self.add_thumb(thumb)
        self.add_finger(finger)
        if sensors is not None:
            self.add_sensors(sensors)
        return self


@dataclass(eq=False)
class HumanoidGripper(Manipulator):
    """
    Represents a human-like gripper of a robot. Contains a collection of fingers and a thumb. The thumb is a specific finger
    that always needs to touch an object when grasping it, ensuring a stable grasp.
    """

    thumb: Finger = field(default=None)
    """
    The thumb of the humanoid gripper, which is the part that always needs to touch an object when grasping it.
    """

    fingers: List[Finger] = field(default_factory=list)
    """
    The fingers of the humanoid gripper, which are the parts that move in parallel to the thumb to grasp objects.
    """

    @synchronized_attribute_modification
    def add_fingers(self, fingers: List[Finger]):
        self.fingers.extend(fingers)

    @synchronized_attribute_modification
    def add_thumb(self, thumb: Finger):
        self.thumb = thumb

    @classmethod
    def create_and_add_to_world(
        cls,
        name: PrefixedName,
        world: World,
        root_name: str,
        tool_frame_name: str,
        front_facing_orientation: Quaternion,
        fingers: List[Finger],
        thumb: Finger,
        sensors: List[Sensor] = None,
    ) -> Self:
        if thumb._world is not world or any(f._world is not world for f in fingers):
            raise ValueError(
                "The fingers and thumb must be part of the given world, but they are not."
            )
        self = cls(
            name=name,
            root=world.get_body_by_name(root_name),
            tool_frame=world.get_body_by_name(tool_frame_name),
            front_facing_orientation=front_facing_orientation,
        )
        world.add_semantic_annotation(self)
        self.add_thumb(thumb)
        self.add_fingers(fingers)
        if sensors is not None:
            self.add_sensors(sensors)
        return self


@dataclass(eq=False)
class Sensor(RobotPart, ABC):
    """
    Abstract base class for any kind of sensor in a robot.
    """


@dataclass
class FieldOfView:
    """
    Represents the field of view of a camera sensor, defined by the vertical and horizontal angles of the camera's view.
    """

    vertical_angle: float
    horizontal_angle: float


@dataclass(eq=False)
class Camera(Sensor):
    """
    Represents a camera sensor in a robot.
    """

    forward_facing_axis: Vector3 = field(kw_only=True)
    field_of_view: FieldOfView = field(kw_only=True)
    default_camera: bool = False

    # these should be calculated values i think?
    minimal_height: float = 0.0
    maximal_height: float = 1.0

    @classmethod
    def create_and_add_to_world(
        cls,
        name: PrefixedName,
        world: World,
        root_name: str,
        forward_facing_axis: Vector3,
        field_of_view: FieldOfView,
        minimal_height: float,
        maximal_height: float,
        default_camera: bool,
    ) -> Self:
        self = cls(
            name=name,
            root=world.get_body_by_name(root_name),
            forward_facing_axis=forward_facing_axis,
            field_of_view=field_of_view,
            default_camera=default_camera,
            minimal_height=minimal_height,
            maximal_height=maximal_height,
        )
        world.add_semantic_annotation(self)
        return self


@dataclass(eq=False)
class Torso(KinematicChain):
    """
    A Torso is a kinematic chain connecting the base of the robot with a collection of other kinematic chains.
    """

    @classmethod
    def create_and_add_to_world(
        cls,
        name: PrefixedName,
        world: World,
        root_name: str,
        tip_name: str,
        sensors: List[Sensor] = None,
    ) -> Self:
        self = cls(
            name=name,
            root=world.get_body_by_name(root_name),
            tip=world.get_body_by_name(tip_name),
        )
        world.add_semantic_annotation(self)
        if sensors is not None:
            self.add_sensors(sensors)
        return self


@dataclass(eq=False)
class MobileBase(RobotPart):
    """
    The base of a robot
    """

    main_axis: Vector3 = field(default_factory=Vector3.X)
    """
    Axis along which the robot manipulates
    """

    full_body_controlled: bool = field(default=False, kw_only=True)
    """
    If True, the robot can move its entire body during a motion. 
    If False, only the robot will always stand still when moving an arm.
    """

    @property
    def bounding_box(self) -> BoundingBox:
        return self.root.collision.as_bounding_box_collection_in_frame(
            self._world.root
        ).bounding_box()

    @classmethod
    def create_and_add_to_world(
        cls,
        name: PrefixedName,
        world: World,
        root_name: str,
        main_axis: Vector3,
        full_body_controlled: bool,
        sensors: List[Sensor] = None,
    ) -> Self:
        self = cls(
            name=name,
            root=world.get_body_by_name(root_name),
            main_axis=main_axis,
            full_body_controlled=full_body_controlled,
        )
        if sensors is not None:
            self.add_sensors(sensors)
        world.add_semantic_annotation(self)
        return self
