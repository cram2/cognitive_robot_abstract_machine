from __future__ import annotations

import logging
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, Any

from typing_extensions import (
    Iterable,
    Set,
    TYPE_CHECKING,
    Optional,
    Self,
    DefaultDict,
    List,
)

from krrood.adapters.json_serializer import SubclassJSONSerializer
from ..collision_checking.collision_detector import CollisionCheck
from ..spatial_types.derivatives import DerivativeMap
from ..spatial_types.spatial_types import (
    Vector3,
    Quaternion,
)
from ..world_description.connections import (
    ActiveConnection,
    OmniDrive,
    ActiveConnection1DOF,
)
from ..world_description.geometry import BoundingBox
from ..world_description.shape_collection import BoundingBoxCollection
from ..world_description.world_entity import (
    Body,
    RootedSemanticAnnotation,
    Agent,
    Connection,
    CollisionCheckingConfig,
)
from ..world_description.world_entity import (
    KinematicStructureEntity,
    Region,
)

if TYPE_CHECKING:
    from ..world import World


@dataclass
class SemanticRobotAnnotation(RootedSemanticAnnotation, ABC):
    """
    Represents a collection of connected robot bodies, starting from a root body, and ending in a unspecified collection
    of tip bodies.
    """

    _robot: AbstractRobot = field(default=None)
    """
    The robot this semantic annotation belongs to
    """

    def __post_init__(self):
        if self._world is not None:
            self._world.add_semantic_annotation(self)

    @abstractmethod
    def assign_to_robot(self, robot: AbstractRobot):
        """
        This method assigns the robot to the current semantic annotation, and then iterates through its own fields to call the
        appropriate methods to att them to the robot.

        :param robot: The robot to which this semantic annotation should be assigned.
        """
        ...


@dataclass
class KinematicChain(SemanticRobotAnnotation, ABC):
    """
    Abstract base class for kinematic chain in a robot, starting from a root body, and ending in a specific tip body.
    A kinematic chain can contain both a manipulator and sensors at the same time. There are no assumptions about the
    position of the manipulator or sensors in the kinematic chain
    """

    tip: Body = field(default=None)
    """
    The tip body of the kinematic chain, which is the last body in the chain.
    """

    manipulator: Optional[Manipulator] = None
    """
    The manipulator of the kinematic chain, if it exists. This is usually a gripper or similar device.
    """

    sensors: Set[Sensor] = field(default_factory=set)
    """
    A collection of sensors in the kinematic chain, such as cameras or other sensors.
    """

    @property
    def bodies(self) -> Iterable[Body]:
        """
        Returns itself as a kinematic chain of bodies.
        """
        return [
            entity
            for entity in self._world.compute_chain_of_kinematic_structure_entities(
                self.root, self.tip
            )
            if isinstance(entity, Body)
        ]

    @property
    def kinematic_structure_entities(self) -> Iterable[KinematicStructureEntity]:
        """
        Returns itself as a kinematic chain of KinematicStructureEntity.
        """
        return self._world.compute_chain_of_kinematic_structure_entities(
            self.root, self.tip
        )

    @property
    def regions(self) -> Iterable[Region]:
        """
        Returns itself as a kinematic chain of KinematicStructureEntity.
        """
        return [
            entity
            for entity in self._world.compute_chain_of_kinematic_structure_entities(
                self.root, self.tip
            )
            if isinstance(entity, Region)
        ]

    @property
    def connections(self) -> Iterable[Connection]:
        """
        Returns the connections of the kinematic chain.
        This is a list of connections between the bodies in the kinematic chain
        """
        return self._world.compute_chain_of_connections(self.root, self.tip)

    def assign_to_robot(self, robot: AbstractRobot):
        """
        Assigns the kinematic chain to the given robot. This method ensures that the kinematic chain is only assigned
        to one robot at a time, and raises an error if it is already assigned to another robot.
        """
        if self._robot is not None and self._robot != robot:
            raise ValueError(
                f"Kinematic chain {self.name} is already part of another robot: {self._robot.name}."
            )
        if self._robot is not None:
            return
        self._robot = robot
        if self.manipulator is not None:
            robot.add_manipulator(self.manipulator)
        for sensor in self.sensors:
            robot.add_sensor(sensor)

    def __hash__(self):
        """
        Returns the hash of the kinematic chain, which is based on the root and tip bodies.
        This allows for proper comparison and storage in sets or dictionaries.
        """
        return hash((self.name, self.root, self.tip))


@dataclass
class Arm(KinematicChain):
    """
    Represents an arm of a robot, which is a kinematic chain with a specific tip body.
    An arm has a manipulators and potentially sensors.
    """

    def __hash__(self):
        """
        Returns the hash of the kinematic chain, which is based on the root and tip bodies.
        This allows for proper comparison and storage in sets or dictionaries.
        """
        return hash((self.name, self.root, self.tip))


@dataclass
class Manipulator(SemanticRobotAnnotation, ABC):
    """
    Abstract base class of robot manipulators. Always has a tool frame.
    """

    tool_frame: Body = field(default=None)

    front_facing_orientation: Quaternion = field(default=None)
    """
    The orientation of the manipulator's tool frame, which is usually the front-facing orientation.
    """

    front_facing_axis: Vector3 = field(default=None)
    """
    The axis of the manipulator's tool frame that is facing forward.
    """

    def assign_to_robot(self, robot: AbstractRobot):
        """
        Assigns the manipulator to the given robot. This method ensures that the manipulator is only assigned
        to one robot at a time, and raises an error if it is already assigned to another robot.
        """
        if self._robot is not None and self._robot != robot:
            raise ValueError(
                f"Manipulator {self.name} is already part of another robot: {self._robot.name}."
            )
        if self._robot is not None:
            return
        self._robot = robot

    def __hash__(self):
        """
        Returns the hash of the kinematic chain, which is based on the root and tip bodies.
        This allows for proper comparison and storage in sets or dictionaries.
        """
        return hash((self.name, self.root, self.tool_frame))


@dataclass
class Finger(KinematicChain):
    """
    A finger is a kinematic chain, since it should have an unambiguous tip body, and may contain sensors.
    """

    def __hash__(self):
        """
        Returns the hash of the kinematic chain, which is based on the root and tip bodies.
        This allows for proper comparison and storage in sets or dictionaries.
        """
        return hash((self.name, self.root, self.tip))


@dataclass
class ParallelGripper(Manipulator):
    """
    Represents a gripper of a robot. Contains a collection of fingers and a thumb. The thumb is a specific finger
    that always needs to touch an object when grasping it, ensuring a stable grasp.
    """

    finger: Finger = field(default=None)
    thumb: Finger = field(default=None)

    def assign_to_robot(self, robot: AbstractRobot):
        """
        Assigns the parallel gripper to the given robot and calls the appropriate methods for the its finger and thumb.
         This method ensures that the parallel gripper is only assigned to one robot at a time, and raises an error if
         it is already assigned to another
        """
        if self._robot is not None and self._robot != robot:
            raise ValueError(
                f"ParallelGripper {self.name} is already part of another robot: {self._robot.name}."
            )
        if self._robot is not None:
            return
        self._robot = robot

    def __hash__(self):
        """
        Returns the hash of the kinematic chain, which is based on the root and tip bodies.
        This allows for proper comparison and storage in sets or dictionaries.
        """
        return hash((self.name, self.root, self.tool_frame))


@dataclass
class Sensor(SemanticRobotAnnotation, ABC):
    """
    Abstract base class for any kind of sensor in a robot.
    """

    def assign_to_robot(self, robot: AbstractRobot):
        """
        Assigns the sensor to the given robot. This method ensures that the sensor is only assigned
        to one robot at a time, and raises an error if it is already assigned to another robot.
        """
        if self._robot is not None and self._robot != robot:
            raise ValueError(
                f"Sensor {self.name} is already part of another robot: {self._robot.name}."
            )
        if self._robot is not None:
            return
        self._robot = robot


@dataclass
class FieldOfView(SubclassJSONSerializer):
    """
    Represents the field of view of a camera sensor, defined by the vertical and horizontal angles of the camera's view.
    """

    vertical_angle: float
    horizontal_angle: float

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "vertical_angle": self.vertical_angle,
            "horizontal_angle": self.horizontal_angle,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            vertical_angle=data["vertical_angle"],
            horizontal_angle=data["horizontal_angle"],
        )


@dataclass
class Camera(Sensor):
    """
    Represents a camera sensor in a robot.
    """

    forward_facing_axis: Vector3 = field(default=None)
    field_of_view: FieldOfView = field(default=None)
    minimal_height: float = 0.0
    maximal_height: float = 1.0

    def __hash__(self):
        """
        Returns the hash of the kinematic chain, which is based on the root and tip bodies.
        This allows for proper comparison and storage in sets or dictionaries.
        """
        return hash((self.name, self.root))


@dataclass
class Neck(KinematicChain):
    """
    Represents a special kinematic chain that connects the head of a robot with a collection of sensors, such as cameras
    and which does not have a manipulator.
    """

    pitch_body: Optional[Body] = None
    """
    The body that allows pitch movement in the neck, if it exists.
    """
    yaw_body: Optional[Body] = None
    """
    The body that allows yaw movement in the neck, if it exists.
    """

    def __hash__(self):
        """
        Returns the hash of the kinematic chain, which is based on the root and tip bodies.
        This allows for proper comparison and storage in sets or dictionaries.
        """
        return hash((self.name, self.root, self.tip))


@dataclass
class Torso(KinematicChain):
    """
    A Torso is a kinematic chain connecting the base of the robot with a collection of other kinematic chains.
    """

    def assign_to_robot(self, robot: AbstractRobot):
        """
        Assigns the torso to the given robot and calls the appropriate method for each of its attached kinematic chains.
         This method ensures that the torso is only assigned to one robot at a time, and raises an error if it is
         already assigned to another robot.
        """
        if self._robot is not None and self._robot != robot:
            raise ValueError(
                f"Torso {self.name} is already part of another robot: {self._robot.name}."
            )
        if self._robot is not None:
            return
        self._robot = robot

    def __hash__(self):
        """
        Returns the hash of the kinematic chain, which is based on the root and tip bodies.
        This allows for proper comparison and storage in sets or dictionaries.
        """
        return hash((self.name, self.root, self.tip))


@dataclass
class Base(KinematicChain):
    """
    The base of a robot
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

    def __hash__(self):
        return hash((self.name, self.root, self.tip))


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

    torso: Optional[Torso] = None
    """
    The torso of the robot, which is a kinematic chain connecting the base with a collection of other kinematic chains.
    """

    base: Optional[Base] = None
    """
    The base of the robot, the part closes to the floor
    """

    manipulators: Set[Manipulator] = field(default_factory=set)
    """
    A collection of manipulators in the robot, such as grippers.
    """

    sensors: Set[Sensor] = field(default_factory=set)
    """
    A collection of sensors in the robot, such as cameras.
    """

    manipulator_chains: Set[KinematicChain] = field(default_factory=set)
    """
    A collection of all kinematic chains containing a manipulator, such as a gripper.
    """

    sensor_chains: Set[KinematicChain] = field(default_factory=set)
    """
    A collection of all kinematic chains containing a sensor, such as a camera.
    """

    default_collision_config: CollisionCheckingConfig = field(
        kw_only=True,
        default_factory=lambda: CollisionCheckingConfig(buffer_zone_distance=0.05),
    )

    @abstractmethod
    def setup_collision_config(self):
        """
        Loads the SRDF file for the robot, if it exists. This method is expected to be implemented in subclasses.
        """
        ...

    @property
    def controlled_connections(self) -> Set[ActiveConnection]:
        """
        A subset of the robot's connections that are controlled by a controller.
        """
        return set(self._world.controlled_connections) & set(self.connections)

    @classmethod
    @abstractmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates a robot semantic annotation from the given world.
        This method constructs the robot semantic annotation by identifying and organizing the various semantic components of the robot,
        such as manipulators, sensors, and kinematic chains. It is expected to be implemented in subclasses.

        :param world: The world from which to create the robot semantic annotation.

        :return: A robot semantic annotation.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

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
                    [None, -new_limits[connection], None, None]
                ),
                new_upper_limits=DerivativeMap(
                    [None, new_limits[connection], None, None]
                ),
            )

    def add_manipulator(self, manipulator: Manipulator):
        """
        Adds a manipulator to the robot's collection of manipulators.
        """
        self.manipulators.add(manipulator)
        self._semantic_annotations.add(manipulator)
        manipulator.assign_to_robot(self)

    def add_sensor(self, sensor: Sensor):
        """
        Adds a sensor to the robot's collection of sensors.
        """
        self.sensors.add(sensor)
        self._semantic_annotations.add(sensor)
        sensor.assign_to_robot(self)

    def add_torso(self, torso: Torso):
        """
        Adds a torso to the robot's collection of kinematic chains.
        """
        if self.torso is not None:
            raise ValueError(
                f"Robot {self.name} already has a torso: {self.torso.name}."
            )
        self.torso = torso
        self._semantic_annotations.add(torso)
        torso.assign_to_robot(self)

    def add_base(self, base: Base):
        """
        Adds the given base to the robot
        """
        if self.base is not None:
            raise ValueError(f"Robot {self.name} already has a base: {self.base.name}.")
        self.base = base
        self._semantic_annotations.add(base)
        base.assign_to_robot(self)

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
            self.manipulator_chains.add(kinematic_chain)
        if kinematic_chain.sensors:
            self.sensor_chains.add(kinematic_chain)
        self._semantic_annotations.add(kinematic_chain)
        kinematic_chain.assign_to_robot(self)

    def create_collision_matrix_for_env(self) -> List[CollisionCheck]:
        """
        Cretaes a collision matrix between the bodies of the robot and the bodies of the environment (environment is
        everything that is not the robot). Only bodes with collision will be used
        """
        all_bodies = self._world.bodies_with_enabled_collision
        env_bodies = set(all_bodies) - set(self.bodies)
        collision_matrx = []
        for body_a, body_b in product(self.bodies, env_bodies):
            collision_matrx.append(
                CollisionCheck(
                    body_a=body_a,
                    body_b=body_b,
                    distance=body_a.get_collision_config().buffer_zone_distance,
                    _world=self._world,
                )
            )
        return collision_matrx
