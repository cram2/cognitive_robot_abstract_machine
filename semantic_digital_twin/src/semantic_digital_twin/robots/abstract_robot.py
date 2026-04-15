from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Type, Union, TYPE_CHECKING, Optional

from typing_extensions import Self, DefaultDict

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import MissingDefaultCameraError
from semantic_digital_twin.robots.robot_parts import (
    Arm,
    Torso,
    MobileBase,
    RobotPart,
    Camera,
    Manipulator,
    Sensor,
    AggregatesRobotParts,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Agent
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world_description.connections import (
    ActiveConnection,
    ActiveConnection1DOF,
    Drive,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.world_modification import (
    synchronized_attribute_modification,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World
    from semantic_digital_twin.reasoning.predicates import LeftOf, RightOf

logger = logging.getLogger("semantic_digital_twin")


@dataclass(eq=False)
class HasRobotPart(AggregatesRobotParts, ABC):
    """
    Mixin class for robots that have robot parts.
    """

    @abstractmethod
    def _setup_robot_parts(self): ...

    @property
    def manipulators(self) -> list[RobotPart]:
        """
        A collection of all manipulators in the robot.
        """
        return [part for part in self._robot_parts if isinstance(part, Manipulator)]

    @property
    def sensors(self) -> list[Sensor]:
        """
        A collection of all sensors in the robot.
        """
        return [part for part in self._robot_parts if isinstance(part, Sensor)]


@dataclass(eq=False)
class HasArms(HasRobotPart, ABC):
    """
    Mixin class for robots that have arms.
    """

    arms: List[Arm] = field(default_factory=list)
    """
    A collection of arms in the robot.
    """

    @synchronized_attribute_modification
    def add_arm(self, arm: Arm):
        """
        Adds a kinematic chain to the PR2 robot's collection of kinematic chains.
        If the kinematic chain is an arm, it will be added to the left or right arm accordingly.

        :param arm: The kinematic chain to add to the PR2 robot.
        """
        self.arms.append(arm)

    def _setup_robot_parts(self):
        super()._setup_robot_parts()
        self._setup_arm_semantic_annotations()
        self._setup_arm_hardware_interfaces()
        self._setup_arm_joint_state()

    @abstractmethod
    def _setup_arm_semantic_annotations(self): ...

    @abstractmethod
    def _setup_arm_hardware_interfaces(self): ...

    @abstractmethod
    def _setup_arm_joint_state(self): ...


@dataclass(eq=False)
class HasOneArm(HasArms, ABC):
    """
    Mixin class for robots that have exactly one arm.
    """

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
class HasExternalSensors(HasRobotPart, ABC):
    """
    Mixin class for robots that have an external camera.
    """

    external_sensors: List[Sensor] = field(default_factory=list)
    """
    A collection of external sensors in the robot.
    """

    @synchronized_attribute_modification
    def add_external_sensor(self, sensor: Sensor):
        self.external_sensors.append(sensor)

    def _setup_robot_parts(self):
        super()._setup_robot_parts()
        self._setup_external_sensors()

    @abstractmethod
    def _setup_external_sensors(self): ...


@dataclass(eq=False)
class HasTorso(HasRobotPart, ABC):
    """
    Mixin class for robots that have a torso.
    """

    torso: Optional[Torso] = field(default=None)
    """
    The torso of the robot, represented as an arm.
    """

    @synchronized_attribute_modification
    def add_torso(self, torso: Torso):
        self.torso = torso

    def _setup_robot_parts(self):
        super()._setup_robot_parts()
        self._setup_torso_semantic_annotations()
        self._setup_torso_hardware_interfaces()
        self._setup_torso_joint_state()

    @abstractmethod
    def _setup_torso_semantic_annotations(self): ...

    @abstractmethod
    def _setup_torso_hardware_interfaces(self): ...

    @abstractmethod
    def _setup_torso_joint_state(self): ...


@dataclass(eq=False)
class HasMobileBase(HasRobotPart, ABC):

    mobile_base: Optional[MobileBase] = field(default=None)

    @synchronized_attribute_modification
    def add_mobile_base(self, mobile_base: MobileBase):
        self.mobile_base = mobile_base

    def _setup_robot_parts(self):
        super()._setup_robot_parts()
        self._setup_mobile_base_semantic_annotations()

    @abstractmethod
    def _setup_mobile_base_semantic_annotations(self): ...

    @property
    def full_body_controlled(self):
        return self.mobile_base.full_body_controlled


@dataclass(eq=False)
class AbstractRobot(Agent, HasRobotPart, ABC):
    """
    Specification of an abstract robot
    """

    def _setup_robot_parts(self):
        super()._setup_robot_parts()

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
            robot_root_body = cls._get_robot_root_body(world)
            robot = cls(
                name=PrefixedName(cls.__name__, world.name),
                root=robot_root_body,
            )
            world.add_semantic_annotation(robot)
            robot._setup_other_hardware_interfaces()
            robot._setup_robot_parts()
            robot._setup_collision_rules()
            robot._setup_velocity_limits()
        return robot

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

    @classmethod
    @abstractmethod
    def _get_robot_root_body(cls, world: World) -> Body: ...

    @abstractmethod
    def _setup_collision_rules(self): ...

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(
            lambda: 1.0,
        )
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

    @abstractmethod
    def _setup_other_hardware_interfaces(self): ...

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
