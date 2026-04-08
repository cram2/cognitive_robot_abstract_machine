from __future__ import annotations

import logging
from abc import abstractmethod, ABC
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable

from typing_extensions import (
    TYPE_CHECKING,
    Optional,
    Self,
    DefaultDict,
    List,
    assert_never,
)

from krrood.class_diagrams.attribute_introspector import DataclassOnlyIntrospector
from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.class_diagrams.wrapped_field import WrappedField
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import JointStateType
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import NoJointStateWithType
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import Agent
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.spatial_types.spatial_types import (
    Vector3,
    Quaternion,
    RotationMatrix,
)
from semantic_digital_twin.world_description.connections import (
    ActiveConnection,
    OmniDrive,
    ActiveConnection1DOF,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import BoundingBox, Scale
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    Connection,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)
from semantic_digital_twin.world_description.world_modification import (
    synchronized_attribute_modification,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World

logger = logging.getLogger(__name__)

_REQUIRED_FOR_ROBOT_SETUP_KEY = "__required_for_robot_setup__"


def required_for_robot_setup(function: Callable) -> Callable:
    setattr(function, _REQUIRED_FOR_ROBOT_SETUP_KEY, True)
    return function


@dataclass(eq=False)
class RobotPart(HasRootBody, ABC):
    """
    Represents a collection of connected robot bodies, starting from a root body, and ending in a unspecified collection
    of tip bodies.
    """

    _robot: AbstractRobot = field(init=False, default=None, repr=False)
    """
    The robot this semantic annotation belongs to
    """

    joint_states: list[JointState] = field(default_factory=list)
    """
    Fixed joint states that are defined for this robot annotation. 
    """

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
    ) -> Self: ...

    def _print_out_missing_fields(self):
        wrapped_class = WrappedClass(self.__class__)
        introspector = DataclassOnlyIntrospector()
        for field_ in introspector.discover(self.__class__):
            value = getattr(self, field_.public_name)
            wrapped_field = WrappedField(wrapped_class, field_.field)
            type_endpoint = wrapped_field.type_endpoint

            if isinstance(value, (list, set)) and issubclass(
                wrapped_field.contained_type, RobotPart
            ):

                if not value:
                    logger.info(
                        f"The field {field_.public_name} of {self.__class__.__name__} is empty. Please confirm that this is intentional."
                    )
                else:
                    for robot_part in value:
                        robot_part._print_out_missing_fields()

            elif issubclass(type_endpoint, RobotPart) and value is None:
                logger.info(
                    f"The field {field_.public_name} of {self.__class__.__name__} is empty. Please confirm that this is intentional."
                )


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

    sensors: List[Sensor] = field(default_factory=list)
    """
    A collection of sensors in the kinematic chain, such as cameras or other sensors.
    """

    @synchronized_attribute_modification
    def add_sensors(self, sensors: List[Sensor]):
        self.sensors.extend(sensors)

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


@dataclass(eq=False)
class Arm(KinematicChain):
    """
    Represents an arm of a robot, which is a kinematic chain with a manipulator.
    """

    manipulator: Optional[Manipulator] = field(init=False, default=None, repr=False)
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

    front_facing_axis: Vector3 = field(kw_only=True, init=False)
    """
    The axis of the manipulator's tool frame that is facing forward.
    """

    is_controlled = True

    def __post_init__(self):
        rotation_matrix = RotationMatrix.from_quaternion(self.front_facing_orientation)
        # raise NotImplementedError("Luca Implement this correctly!")
        self.front_facing_axis = rotation_matrix[:2, 0]


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

    thumb: Finger = field(init=False, default=None, repr=False)
    """
    The thumb of the parallel gripper, which is the part that always needs to touch an object when grasping it.
    """

    finger: Finger = field(init=False, default=None, repr=False)
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
        default_camera: bool = False,
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
class Base(KinematicChain):
    """
    The base of a robot
    """

    main_axis: Vector3 = field(default_factory=Vector3.X)
    """
    Axis along which the robot manipulates
    """

    @property
    def bounding_box(self) -> BoundingBox:
        bounding_boxes = [
            kse.collision.as_bounding_box_collection_in_frame(
                self._world.root
            ).bounding_box()
            for kse in self._world.compute_chain_of_kinematic_structure_entities(
                self.root, self.tip
            )
            if kse.collision is not None
        ]
        bb_collection = BoundingBoxCollection(
            bounding_boxes, reference_frame=self._world.root
        )
        return bb_collection.bounding_box()

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
class AbstractRobot(Agent, ABC):
    """
    Specification of an abstract robot. A robot consists of:
    - a root body, which is the base of the robot
    - an optional torso, which is a kinematic chain (usually without a manipulator) connecting the base with a collection
        of other kinematic chains
    - an optional collection of manipulator chains, each containing a manipulator, such as a gripper
    - an optional collection of sensor chains, each containing a sensor, such as a camera
    => If a kinematic chain contains both a manipulator and a sensor, it will be part of both collections
    """

    # torso: Optional[Torso] = None
    # """
    # The torso of the robot, which is a kinematic chain connecting the base with a collection of other kinematic chains.
    # """

    # base: Optional[Base] = None
    # """
    # The base of the robot, the part closes to the floor
    # """

    # manipulators: List[Manipulator] = field(default_factory=list)
    # """
    # A collection of manipulators in the robot, such as grippers.
    # """

    # sensors: List[Sensor] = field(default_factory=list)
    # """
    # A collection of sensors in the robot, such as cameras.
    # """
    #
    # manipulator_chains: List[KinematicChain] = field(default_factory=list)
    # """
    # A collection of all kinematic chains containing a manipulator, such as a gripper.
    # """
    #
    # sensor_chains: List[KinematicChain] = field(default_factory=list)
    # """
    # A collection of all kinematic chains containing a sensor, such as a camera.
    # """

    # full_body_controlled: bool = field(default=False, kw_only=True)
    # """
    # Whether this robots needs full-body control to be able to operate effectively
    # """

    def _get_robot_setup_methods(self):
        names = set()

        for base in type(self).__mro__[1:]:
            for name, obj in vars(base).items():
                if getattr(obj, _REQUIRED_FOR_ROBOT_SETUP_KEY, None) is not None:
                    names.add(name)

        return {name: getattr(self, name) for name in names}

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

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates a robot semantic annotation from the given world.
        This method constructs the robot semantic annotation by identifying and organizing the various semantic components of the robot,
        such as manipulators, sensors, and kinematic chains. It is expected to be implemented in subclasses.

        :param world: The world from which to create the robot semantic annotation.

        :return: A robot semantic annotation.
        """
        with world.modify_world():
            robot_root_body = cls._get_structural_root_body(world)
            robot = cls(
                name=PrefixedName(cls.__name__, world.name),
                root=robot_root_body,
            )
            world.add_semantic_annotation(robot)

            setup_methods = robot._get_robot_setup_methods()
            for name, setup_method in setup_methods.items():
                setup_method()

            robot._setup_collision_rules()
            robot._setup_velocity_limits()
            robot._setup_hardware_interfaces()
            robot._setup_joint_states()
        return robot

    @classmethod
    def mock_from_urdf_file_and_validate(cls, urdf_file: str):
        world = URDFParser.from_file(urdf_file).parse(mock_geometry=True)
        self = cls.from_world(world)
        self.validate()

    def validate(self) -> bool:

        wrapped_class = WrappedClass(self.__class__)
        introspector = DataclassOnlyIntrospector()
        for field_ in introspector.discover(self.__class__):
            value = getattr(self, field_.public_name)
            wrapped_field = WrappedField(wrapped_class, field_.field)
            type_endpoint = wrapped_field.type_endpoint

            if isinstance(value, (list, set)) and issubclass(
                wrapped_field.contained_type, RobotPart
            ):
                if not value:
                    logger.info(
                        f"The field {field_.public_name} of {self.__class__.__name__} is empty. Please confirm that this is intentional."
                    )
                else:
                    for robot_part in value:
                        robot_part._print_out_missing_fields()
            elif issubclass(type_endpoint, RobotPart):
                logger.info(
                    f"The field {field_.public_name} of {self.__class__.__name__} is empty. Please confirm that this is intentional."
                )

        self_world_copy = deepcopy(self._world)

        assert all(
            (original_b.id == copy_b.id)
            for original_b, copy_b in zip(self_world_copy.bodies, self._world.bodies)
        )
        assert all(
            (original_s.id == copy_s.id)
            for original_s, copy_s in zip(
                self_world_copy.semantic_annotations, self._world.semantic_annotations
            )
        )
        assert all(
            (hash(original_c) == hash(copy_c))
            for original_c, copy_c in zip(
                self_world_copy.connections, self._world.connections
            )
        )

        return True

    @classmethod
    @abstractmethod
    def _get_structural_root_body(cls, world: World) -> Body: ...

    @abstractmethod
    def _setup_semantic_annotations(self): ...

    @abstractmethod
    def _setup_collision_rules(self): ...

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(
            lambda: 1.0,
        )
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

    @abstractmethod
    def _setup_hardware_interfaces(self): ...

    @abstractmethod
    def _setup_joint_states(self): ...

    @property
    def drive(self) -> Optional[OmniDrive]:
        """
        The connection which the robot uses for driving.
        """
        try:
            parent_connection = self.root.parent_connection
            if isinstance(parent_connection, OmniDrive):
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

    def add_manipulator(self, manipulator: Manipulator):
        """
        Adds a manipulator to the robot's collection of manipulators.
        """
        self.manipulators.append(manipulator)
        self._semantic_annotations.add(manipulator)
        manipulator.assign_to_robot(self)

    def add_sensor(self, sensor: Sensor):
        """
        Adds a sensor to the robot's collection of sensors.
        """
        self.sensors.append(sensor)
        self._semantic_annotations.add(sensor)
        sensor.assign_to_robot(self)

    def add_kinematic_chain(self, kinematic_chain: KinematicChain):
        """
        Adds a kinematic chain to the robot's collection of kinematic chains.
        This can be either a manipulator chain or a sensor chain.
        """
        if kinematic_chain.manipulator is None and not kinematic_chain.sensors:
            logging.warning(
                f"Kinematic chain {kinematic_chain.name} has no manipulator or sensors, so it was skipped. Did you mean to add it to the torso?"
            )
            return
        if kinematic_chain.manipulator is not None:
            self.manipulator_chains.append(kinematic_chain)
        if kinematic_chain.sensors:
            self.sensor_chains.append(kinematic_chain)
        self._semantic_annotations.add(kinematic_chain)
        kinematic_chain.assign_to_robot(self)

    def get_default_camera(self) -> Camera:
        for sensor in self.sensors:
            if isinstance(sensor, Camera) and sensor.default_camera:
                return sensor
        return [s for s in self.sensors if isinstance(s, Camera)][0]
