from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Self, TYPE_CHECKING, Set
from uuid import UUID

from krrood.adapters.json_serializer import list_like_classes
from krrood.class_diagrams.attribute_introspector import (
    DataclassOnlyIntrospector,
    DiscoveredAttribute,
)
from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.entity_query_language.factories import variable, contains, a, entity
from semantic_digital_twin.datastructures.definitions import JointStateType
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    NoJointStateWithType,
    UselessConceptError,
    DuplicateRobotAssignmentsError,
)
from semantic_digital_twin.robots.abstract_robot import (
    HasFingers,
    HasTwoFingers,
    HasEndEffector,
    HasCameras,
    HasRobotParts,
)
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.spatial_types import (
    Quaternion,
    Vector3,
    RotationMatrix,
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.connections import ActiveConnection
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import BoundingBox, Scale
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
    Connection,
)
from semantic_digital_twin.world_description.world_modification import (
    synchronized_attribute_modification,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World
    from semantic_digital_twin.robots.abstract_robot import AbstractRobot


logger = logging.getLogger("semantic_digital_twin")


@dataclass(eq=False)
class AbstractRobotPart(HasRootBody, HasRobotParts, ABC):

    joint_states: list[JointState] = field(default_factory=list)

    @synchronized_attribute_modification
    def add_joint_state(self, joint_state: JointState):
        """
        Adds a joint state to this semantic annotation.
        """
        self.joint_states.append(joint_state)
        joint_state.assign_to_robot(self._robot)

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
        raise UselessConceptError(
            message="The bodies needed for RobotParts should already exist in the world, by parsing a URDF"
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

    def _default_hardware_interface_setup(self):
        """
        Sets up a default hardware interface for the robot part by setting the has_hardware_interface flag to True for
         all active connections of all robot parts in this robot part
        """
        for robot_part in self._robot_parts:
            for connection in robot_part.active_connections:
                connection.has_hardware_interface = True

    @property
    def active_connections(self) -> list[ActiveConnection]:
        return [
            connection
            for connection in self.connections
            if isinstance(connection, ActiveConnection)
        ]


@dataclass(eq=False)
class KinematicChain(AbstractRobotPart, ABC):

    tip: Body = field(kw_only=True)

    def _kinematic_structure_entities(
        self, visited: Set[int]
    ) -> list[KinematicStructureEntity]:
        if id(self) in visited:
            return []
        visited.add(id(self))
        kinematic_structure_entities = [
            entity
            for entity in self._world.compute_chain_of_kinematic_structure_entities(
                self.root, self.tip
            )
        ]

        for robot_part in self._robot_parts:
            kinematic_structure_entities.extend(
                robot_part._kinematic_structure_entities(visited=visited)
            )

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


@dataclass
class FieldOfView:
    """
    Represents the field of view of a camera sensor, defined by the vertical and horizontal angles of the camera's view.
    """

    vertical_angle: float
    horizontal_angle: float


@dataclass(eq=False)
class Sensor(AbstractRobotPart, ABC): ...


@dataclass(eq=False)
class Camera(Sensor, ABC):
    forward_facing_axis: Vector3 = field(kw_only=True)
    field_of_view: FieldOfView = field(kw_only=True)
    default_camera: bool = False

    # these should be calculated values i think?
    minimal_height: float = 0.0
    maximal_height: float = 1.0


@dataclass(eq=False)
class Finger(KinematicChain, ABC):

    finger_tip_frame: Optional[Body] = None
    """
    The frame of the finger tip. Could be used to align the finger with, for example, a button.
    """


@dataclass(eq=False)
class EndEffector(AbstractRobotPart, ABC):
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

    def __post_init__(self):
        super().__post_init__()
        rotation_matrix = RotationMatrix.from_quaternion(self.front_facing_orientation)
        self.front_facing_axis = Vector3.from_iterable(rotation_matrix[:3, 0])


@dataclass(eq=False)
class MechanicalGripper(EndEffector, ABC): ...


@dataclass(eq=False)
class ParallelGripper(MechanicalGripper, HasTwoFingers, ABC): ...


@dataclass(eq=False)
class HumanoidHand(MechanicalGripper, HasFingers, ABC): ...


@dataclass(eq=False)
class Torso(KinematicChain, ABC): ...


@dataclass(eq=False)
class Arm(KinematicChain, HasEndEffector, ABC): ...


@dataclass(eq=False)
class Neck(KinematicChain, HasCameras, ABC): ...


@dataclass(eq=False)
class MobileBase(AbstractRobotPart, ABC):
    """
    The base of a robot
    """

    forward_axis: Vector3 = field(default_factory=Vector3.X)
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
