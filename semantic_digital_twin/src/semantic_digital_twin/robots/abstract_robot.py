from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from typing import Union, Tuple, Optional, List, Set
from uuid import UUID

from typing_extensions import TYPE_CHECKING, Type, Self, DefaultDict

from krrood.adapters.json_serializer import list_like_classes
from krrood.class_diagrams.attribute_introspector import (
    DataclassOnlyIntrospector,
    DiscoveredAttribute,
)
from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.class_diagrams.wrapped_field import WrappedField
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import MissingDefaultCameraError
from semantic_digital_twin.reasoning.predicates import LeftOf, RightOf
from semantic_digital_twin.semantic_annotations.semantic_annotations import Agent
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world_description.connections import (
    ActiveConnection,
    ActiveConnection1DOF,
    Drive,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)
from semantic_digital_twin.world_description.world_modification import (
    synchronized_attribute_modification,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World
    from semantic_digital_twin.robots.robot_parts import (
        Finger,
        Sensor,
        Camera,
        EndEffector,
        MechanicalGripper,
        ParallelGripper,
        Arm,
        MobileBase,
        Torso,
        Neck,
        HumanoidHand,
        AbstractRobotPart,
    )

logger = logging.getLogger("semantic_digital_twin")


@dataclass(eq=False)
class HasRobotParts(ABC):
    """
    Marker class for all robot parts, including the robot itself.
    This class serves as a common ancestor for all robot parts, allowing for easy identification and
    aggregation of robot parts within a robot semantic annotation.
    """

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

        if isinstance(self, HasRobotParts):
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
class HasFingers(ABC):

    fingers: list[Finger] = field(default_factory=list)

    thumb: Finger = field(default=None)

    @synchronized_attribute_modification
    def add_finger(self, finger: Finger):
        if finger == self.thumb:
            raise Exception(f"This finger is already part of the robot {self}.")
        self.fingers.append(finger)

    @synchronized_attribute_modification
    def add_thumb(self, thumb: Finger):
        if thumb in self.fingers:
            raise Exception(f"This finger is already part of the robot {self}.")
        self.thumb = thumb

    @abstractmethod
    def setup_finger_semantic_annotations(self, world: World): ...


@dataclass(eq=False)
class HasTwoFingers(HasFingers, ABC):

    @property
    def finger(self):
        if len(self.fingers) != 1 or self.thumb is None:
            raise Exception(
                f"This robot has {len(self.fingers)} fingers and {int(bool(self.thumb))} thumbs. Should have exactly one finger and one thumb"
            )
        return self.fingers[0]


@dataclass(eq=False)
class HasSensors(ABC):

    sensors: list[Sensor] = field(default_factory=list)

    @synchronized_attribute_modification
    def add_sensor(self, sensor: Sensor):
        self.sensors.append(sensor)

    @abstractmethod
    def setup_sensor_semantic_annotations(self): ...


@dataclass(eq=False)
class HasCameras(HasSensors, ABC):

    @synchronized_attribute_modification
    def add_camera(self, camera: Camera):
        self.sensors.append(camera)

    @property
    def cameras(self):
        from semantic_digital_twin.robots.robot_parts import Camera

        return [sensor for sensor in self.sensors if isinstance(sensor, Camera)]


@dataclass(eq=False)
class HasEndEffector(ABC):

    end_effector: EndEffector = field(default=None)

    @synchronized_attribute_modification
    def add_end_effector(self, end_effector: EndEffector):
        self.end_effector = end_effector

    @abstractmethod
    def setup_end_effector_semantic_annotation(self): ...


@dataclass(eq=False)
class HasMechanicalGripper(HasEndEffector, ABC):
    end_effector: MechanicalGripper = field(default=None)


@dataclass(eq=False)
class HasParallelGripper(HasMechanicalGripper, ABC):
    end_effector: ParallelGripper = field(default=None)


@dataclass(eq=False)
class HasHumanoidHand(HasMechanicalGripper, ABC):
    end_effector: HumanoidHand = field(default=None)


@dataclass(eq=False)
class HasArms(ABC):
    arms: list[Arm] = field(default_factory=list)

    @synchronized_attribute_modification
    def add_arm(self, arm: Arm):
        self.arms.append(arm)

    @abstractmethod
    def setup_arm_semantic_annotations(self): ...


@dataclass(eq=False)
class HasOneArm(HasArms, ABC):
    """
    Mixin class for robots that have exactly one arm.
    """

    @synchronized_attribute_modification
    def add_arm(self, arm: Arm):
        if len(self.arms) != 0:
            raise Exception(f"This robot already has an arm {self.arms}")
        self.arms.append(arm)

    @property
    def arm(self) -> Arm:
        return self.arms[0]


@dataclass(eq=False)
class HasLeftRightArm(HasArms, ABC):
    """
    Mixin class for robots that have two arms and can specify which is the left and which is the right arm.
    """

    @cached_property
    def left_arm(self):
        from semantic_digital_twin.reasoning.predicates import LeftOf

        return self._assign_left_right_arms(LeftOf)

    @cached_property
    def right_arm(self):
        from semantic_digital_twin.reasoning.predicates import RightOf

        return self._assign_left_right_arms(RightOf)

    def _assign_left_right_arms(self, relation: Type[Union[LeftOf, RightOf]]) -> Arm:
        """
        Assigns the left and right arms based on their position relative to the robot's root body.
        :param relation: The relation to use for determining left or right (LeftOf or RightOf).
        :return: The arm that is on the left or right side of the robot.
        """
        assert (
            len(self.arms) == 2
        ), f"Must have exactly two arms to specify left and right arm, but found {len(self.arms)}."
        pov = self.root.global_transform
        first_arm = self.arms[0]
        second_arm = self.arms[1]
        # the arms may share a root, but the first body after the root should be different
        world_P_first_body = first_arm.bodies[1].global_transform.to_position()
        world_P_second_body = second_arm.bodies[1].global_transform.to_position()

        return (
            first_arm
            if relation(
                world_P_first_body,
                world_P_second_body,
                pov,
            )()
            else second_arm
        )


@dataclass(eq=False)
class HasMobileBase(ABC):
    mobile_base: MobileBase = field(default=None)

    @synchronized_attribute_modification
    def add_mobile_base(self, mobile_base: MobileBase):
        self.mobile_base = mobile_base

    @abstractmethod
    def setup_mobile_base_semantic_annotation(self): ...


@dataclass(eq=False)
class HasTorso(ABC):
    torso: Torso = field(default=None)

    @synchronized_attribute_modification
    def add_torso(self, torso: Torso):
        self.torso = torso

    @abstractmethod
    def setup_default_torso_semantic_annotation(self): ...


@dataclass(eq=False)
class HasNeck(ABC):
    neck: Neck = field(default=None)

    @synchronized_attribute_modification
    def add_neck(self, neck: Neck):
        self.neck = neck

    @abstractmethod
    def setup_neck_semantic_annotations(self): ...


@dataclass(eq=False)
class AbstractRobot(Agent, HasRobotParts, ABC):
    """
    Specification of an abstract robot
    """

    @classmethod
    @abstractmethod
    def _get_root_body_name(cls) -> str: ...

    @abstractmethod
    def setup_robot_part_semantic_annotations(self): ...

    @classmethod
    def from_world(cls, world: World) -> Self:
        return cls.from_branch_in_world(world.root)

    @classmethod
    def from_branch_in_world(cls, branch_root: KinematicStructureEntity) -> Self:
        world = branch_root._world
        with world.modify_world():
            self = cls(
                root=world.get_body_in_branch_by_name(
                    branch_root=branch_root, name=cls._get_root_body_name()
                ),
            )
            world.add_semantic_annotation(self)
            self.setup_robot_part_semantic_annotations()
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
        for sensor in self.sensors:
            if isinstance(sensor, Camera) and sensor.default_camera:
                return sensor
        raise MissingDefaultCameraError(type(self))
