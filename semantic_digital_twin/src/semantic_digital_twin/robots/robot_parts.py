from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, Self, TYPE_CHECKING, Set, List, DefaultDict
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
    MissingDefaultCameraError,
)
from semantic_digital_twin.robots.robot_part_mixins import (
    HasFingers,
    HasTwoFingers,
    HasEndEffector,
    HasCameras,
)
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import Agent
from semantic_digital_twin.spatial_types import (
    Quaternion,
    Vector3,
    RotationMatrix,
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world_description.connections import (
    ActiveConnection,
    Drive,
    ActiveConnection1DOF,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
    DegreeOfFreedom,
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

logger = logging.getLogger("semantic_digital_twin")


@dataclass(eq=False)
class HasRobotParts(ABC):
    @property
    def _robot_parts(self) -> list[AbstractRobotPart]:
        """
        Serves as a generic interface to access all robot parts assigned to a robot part.
        Returns a list of all robot parts assigned directly to this robot part.
        """
        return self._aggregate_robot_parts(set())

    def _aggregate_robot_parts(self, seen: Set[UUID]) -> list[AbstractRobotPart]:
        """
        Recursively aggregates all robot parts assigned to this robot part, including itself if it is a robot part.
         Uses a set of seen UUIDs to avoid infinite recursion in case of cyclic references and duplicates.
        """
        wrapped_class = WrappedClass(self.__class__)
        introspector = DataclassOnlyIntrospector()
        robot_parts = []

        if isinstance(self, AbstractRobotPart):
            if self.id in seen:
                return []
            seen.add(self.id)
            robot_parts.append(self)

        for field_ in introspector.discover(self.__class__):
            value = getattr(self, field_.public_name)
            wrapped_field = WrappedField(wrapped_class, field_.field)

            if isinstance(value, list_like_classes) and issubclass(
                wrapped_field.contained_type, HasRobotParts
            ):
                for robot_part in value:
                    robot_parts.extend(robot_part._aggregate_robot_parts(seen))
            elif isinstance(value, HasRobotParts):
                robot_parts.extend(value._aggregate_robot_parts(seen))

        return robot_parts

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

        if isinstance(value, list_like_classes) and issubclass(
            wrapped_field.contained_type, HasRobotParts
        ):
            if not value:
                self._log_missing_field(field)
                return

            for robot_part in value:
                robot_part._log_missing_fields()

        elif issubclass(type_endpoint, HasRobotParts) and value is None:
            self._log_missing_field(field)

    def _log_missing_field(self, field: DiscoveredAttribute):
        logger.info(
            f"The field {field.public_name} of {self.__class__.__name__} is empty."
        )


@dataclass(eq=False)
class AbstractRobotPart(HasRootBody, HasRobotParts, ABC):
    """
    Abstract base class for all robot parts.
    """

    joint_states: list[JointState] = field(default_factory=list)
    """
    Common joint states for the current robot part.
    """

    @classmethod
    @abstractmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        """
        Sets up a default configuration of this robot part in the world, below the given robot root.
        This is used to set up a default configuration of the robot part in the world after parsing a URDF.
        """

    @abstractmethod
    def setup_hardware_interfaces(self):
        """
        Sets up a default hardware interface for this robot part by setting the has_hardware_interface flag to True for
         relevant connections of this robot part. Implement as "pass" if this robot part does not have any hardware interfaces.
        """

    @abstractmethod
    def setup_joint_states(self):
        """
        Sets up default joint states for this robot part. Implement as "pass" if this robot part does not have any joint states.
        """

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
            message="The bodies needed for RobotParts should already exist in the world after parsing a URDF"
        )

    @property
    def _robot(self) -> Optional[AbstractRobot]:
        """
        Computes backreference to the robot this robot part belongs to.
        """
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

    def _setup_hardware_interfaces_for_active_connections(self):
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
    """
    A kinematic chain is a robot part that consists of a chain of bodies and connections between them.
    It has a root body and a tip body, and the connections between them can be computed using the world description.
    """

    tip: Body = field(kw_only=True)
    """
    The body at the end of the kinematic chain.
    """

    def _kinematic_structure_entities(
        self, visited: Set[int]
    ) -> list[KinematicStructureEntity]:
        """
        Computes the kinematic structure entities of this kinematic chain, which are the bodies and connections that
        make up the kinematic chain, including the bodies of any robot parts that are part of this kinematic chain.

        """
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
    """
    The vertical angle of the camera's field of view, in radians.
    """

    horizontal_angle: float
    """
    The horizontal angle of the camera's field of view, in radians.
    """


@dataclass(eq=False)
class Sensor(AbstractRobotPart, ABC):
    """
    Abstract base class for all sensors. A sensor is a robot part that can perceive the environment.
    """


@dataclass(eq=False)
class Camera(Sensor, ABC):
    """
    A camera is a sensor that captures images of the environment.
    """

    forward_facing_axis: Vector3 = field(kw_only=True)
    """
    The axis of the camera that is facing forward.
    """

    field_of_view: FieldOfView = field(kw_only=True)
    """
    The field of view of the camera, defined by the vertical and horizontal angles of the camera's view.
    """

    default_camera: bool = False
    """
    Whether this camera is the default camera of the robot. Used for quick access.
    """

    minimal_height: float = 0.0
    """
    The minimal height of the camera above the ground, in meters.
    """

    maximal_height: float = 1.0
    """
    The maximal height of the camera above the ground, in meters.
    """


@dataclass(eq=False)
class Finger(KinematicChain, ABC):
    """
    A finger is a kinematic chain attached to a gripper to manipulate objects.
    """

    finger_tip_frame: Optional[Body] = None
    """
    The frame of the finger tip. Could be used to align the finger with, for example, a button.
    """


@dataclass(eq=False)
class EndEffector(AbstractRobotPart, ABC):
    """
    Abstract base class of robot end effector. Always has a tool frame.
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
class ParallelGripper(EndEffector, HasTwoFingers, ABC):
    """
    A parallel gripper is an end effector that has two fingers that move in parallel to grasp objects.
    """


@dataclass(eq=False)
class HumanoidHand(EndEffector, HasFingers, ABC):
    """
    A humanoid hand is an end effector that has multiple fingers that can move independently to grasp objects.
    """


@dataclass(eq=False)
class Torso(KinematicChain, ABC):
    """
    The torso of a robot, which is a kinematic chain providing additional shared degrees of freedom to its
    attachments, such as arms or the neck.
    """


@dataclass(eq=False)
class Arm(KinematicChain, HasEndEffector, ABC):
    """
    An arm is a kinematic chain that has an end effector attached to it.
    """


@dataclass(eq=False)
class Neck(KinematicChain, HasCameras, ABC):
    """
    The neck of a robot, which is a kinematic chain that has a camera attached to it.
    """


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


@dataclass(eq=False)
class AbstractRobot(Agent, HasRobotParts, ABC):
    """
    Specification of an abstract robot
    """

    @classmethod
    @abstractmethod
    def get_ros_file_path(cls) -> str:
        """
        Returns a ROS file path pointing to the description of this robot, for example a URDF file.
        """

    @classmethod
    @abstractmethod
    def _get_root_body_name(cls) -> str:
        """
        Returns the name of the root body of the robot in the world, which serves as the entry point for traversing the
        robot's kinematic structure.
        """

    @abstractmethod
    def setup_robot_part_semantic_annotations(self):
        """
        Sets up the semantic annotations for all robot parts of this robot.
        """

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates a robot from a world.
        """
        return cls.from_branch_in_world(world.root)

    @classmethod
    def from_branch_in_world(cls, branch_root: KinematicStructureEntity) -> Self:
        """
        Creates a robot from a branch in a world.
        This is useful when you have multiple of the same robots in the same world, which would normally cause naming conflicts.
        """
        world = branch_root._world
        robot_root = world.get_body_in_branch_by_name(
            branch_root=branch_root, name=cls._get_root_body_name()
        )
        with world.modify_world():
            self = cls(
                root=robot_root,
            )
            world.add_semantic_annotation(self)
            self.setup_robot_part_semantic_annotations()
            for robot_part in self._robot_parts:
                robot_part.setup_hardware_interfaces()
                robot_part.setup_joint_states()
            return self

    @property
    def controlled_connections(self) -> list[ActiveConnection]:
        """
        A subset of the robot's connections that are controlled by a controller.
        """
        return [
            connection
            for connection in self.connections
            if isinstance(connection, ActiveConnection) and connection.is_controlled
        ]

    @property
    def degrees_of_freedom_with_hardware_interface(self) -> List[DegreeOfFreedom]:
        """
        The number of degrees of freedom of the robot, which is the sum of the degrees of freedom of all its manipulators.
        """
        dofs = []
        for connection in self.connections:
            dofs.extend(connection.controlled_dofs)
        return dofs

    def validate(self) -> bool:
        """
        Validates the robot semantic annotation.
            The validation process includes:
            1. Printing out missing fields of any robot part, so that the user can check if they are intentionally left blank.
            2. Deepcopy the resulting world to ensure that all parts of the robot are initialized in the correct order
            3. Assert that the copied world is the same as the original world
            4. Assert that the robot semantic annotation has a default camera.

        :return: True if the robot semantic annotation is valid, False otherwise.
        """

        for robot_part in self._robot_parts:
            robot_part._log_missing_fields()

        self_world_copy = deepcopy(self._world)

        assert set(self_world_copy._world_entity_hash_table.keys()) == set(
            self._world._world_entity_hash_table.keys()
        )

        assert (
            self_world_copy.get_semantic_annotations_by_type(AbstractRobot)[
                0
            ].get_default_camera()
            is not None
        )

        return True

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(
            lambda: 1.0,
        )
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

    @property
    def drive(self) -> Optional[Drive]:
        """
        The connection which the robot uses for driving.
        """
        try:
            parent_connection = self.root.parent_connection
            if isinstance(parent_connection, Drive):
                return parent_connection
        except AttributeError:
            pass

    def tighten_dof_velocity_limits_of_1dof_connections(
        self,
        new_limits: DefaultDict[ActiveConnection1DOF, float],
    ):
        """
        Convenience method for tightening the velocity limits of all one degree-of-freedom (1DOF)
        active connections in the system.

        The method iterates through all connections of type `ActiveConnection1DOF`
        and configures their velocity limits by overwriting the existing
        lower and upper limit values with the provided ones.

        :param new_limits: A dictionary linking 1DOF connections to their corresponding
            new velocity limits. The keys are of type `ActiveConnection1DOF`, and the
            values represent the new velocity limits specific to each connection.
        """
        for connection in self._world.get_connections_by_type(ActiveConnection1DOF):
            connection.raw_dof._overwrite_dof_limits(
                new_lower_limits=DerivativeMap(
                    None, -new_limits[connection], None, None
                ),
                new_upper_limits=DerivativeMap(
                    None, new_limits[connection], None, None
                ),
            )

    def get_default_camera(self) -> Camera:
        """
        Returns the default camera of the robot.
        """
        for robot_part in self._robot_parts:
            if isinstance(robot_part, Camera) and robot_part.default_camera:
                return robot_part
        raise MissingDefaultCameraError(type(self))
