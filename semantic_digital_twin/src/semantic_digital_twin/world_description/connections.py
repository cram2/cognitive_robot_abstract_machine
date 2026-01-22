from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from uuid import UUID

import numpy as np
from krrood.adapters.json_serializer import from_json, to_json
from typing_extensions import List, TYPE_CHECKING, Union, Optional, Dict, Any, Self

from .degree_of_freedom import DegreeOfFreedom
from .world_entity import CollisionCheckingConfig, Connection, KinematicStructureEntity
from ..adapters.world_entity_kwargs_tracker import WorldEntityWithIDKwargsTracker
from ..datastructures.prefixed_name import PrefixedName
from ..datastructures.types import NpMatrix4x4
from ..spatial_types import HomogeneousTransformationMatrix, Vector3, Point3, Quaternion
from ..spatial_types.derivatives import DerivativeMap
from .connection_properties import JointDynamics

if TYPE_CHECKING:
    from ..world import World


class HasUpdateState(ABC):
    """
    Mixin class for connections that need state updated which are not trivial integrations.
    Typically needed for connections that use active and passive degrees of freedom.
    Look at OmniDrive for an example usage.
    """

    @abstractmethod
    def update_state(self, dt: float) -> None:
        """
        Allows the connection to update the state of its dofs.
        An integration update for active dofs will have happened before this method is called.
        Write directly into self._world.state, but don't touch dofs that don't belong to this connection.
        :param dt: Time passed since last update.
        """
        pass


@dataclass(eq=False)
class FixedConnection(Connection):
    """
    Has 0 degrees of freedom.
    """


@dataclass(eq=False)
class ActiveConnection(Connection):
    """
    Has one or more degrees of freedom that can be actively controlled, e.g., robot joints.
    """

    frozen_for_collision_avoidance: bool = field(default=False)
    """
    Should be treated as fixed for collision avoidance.
    Common example are gripper joints, you generally don't want to avoid collisions by closing the fingers, 
    but by moving the whole hand away.
    """

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["frozen_for_collision_avoidance"] = self.frozen_for_collision_avoidance
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)
        parent = tracker.get_world_entity_with_id(id=from_json(data["parent_id"]))
        child = tracker.get_world_entity_with_id(id=from_json(data["child_id"]))
        return cls(
            name=PrefixedName.from_json(data["name"]),
            parent=parent,
            child=child,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_json(
                data["parent_T_connection_expression"], **kwargs
            ),
            frozen_for_collision_avoidance=data["frozen_for_collision_avoidance"],
            **kwargs,
        )

    @property
    def has_hardware_interface(self) -> bool:
        """
        Whether this connection is linked to a controller and can therefore respond to control commands.

        E.g. the caster wheels of a PR2 are active, because they have a DOF, but they are not directly controlled.
        Instead a the omni drive connection is directly controlled and a low level controller translates these commands
        to commands for the caster wheels.

        A door hinge is also active but cannot be controlled.
        """
        return any(dof.has_hardware_interface for dof in self.dofs)

    @has_hardware_interface.setter
    def has_hardware_interface(self, value: bool) -> None:
        for dof in self.dofs:
            dof.has_hardware_interface = value

    @property
    def is_controlled(self):
        return self.has_hardware_interface and not self.frozen_for_collision_avoidance

    def set_static_collision_config_for_direct_child_bodies(
        self, collision_config: CollisionCheckingConfig
    ):
        for child_body in self._world.get_direct_child_bodies_with_collision(self):
            if not child_body.get_collision_config().disabled:
                child_body.set_static_collision_config(collision_config)


@dataclass(eq=False)
class ActiveConnection1DOF(ActiveConnection, ABC):
    """
    Superclass for active connections with 1 degree of freedom.
    """

    axis: Vector3 = field(kw_only=True)
    """
    Connection moves along this axis, should be a unit vector.
    The axis is defined relative to the local reference frame of the parent KinematicStructureEntity.
    """

    multiplier: float = 1.0
    """
    Movement along the axis is multiplied by this value. Useful if Connections share DoFs.
    """

    offset: float = 0.0
    """
    Movement along the axis is offset by this value. Useful if Connections share DoFs.
    """

    dof_id: UUID = field(kw_only=True)
    """
    UUID of a Degree of freedom to control movement along the axis.
    """

    dynamics: JointDynamics = field(default_factory=JointDynamics)
    """
    Dynamic properties of the joint.
    """

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["axis"] = self.axis.to_np().tolist()
        result["multiplier"] = self.multiplier
        result["offset"] = self.offset
        result["id"] = to_json(self.dof_id)
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)
        parent = tracker.get_world_entity_with_id(id=from_json(data["parent_id"]))
        child = tracker.get_world_entity_with_id(id=from_json(data["child_id"]))
        return cls(
            name=PrefixedName.from_json(data["name"]),
            parent=parent,
            child=child,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_json(
                data["parent_T_connection_expression"], **kwargs
            ),
            frozen_for_collision_avoidance=data["frozen_for_collision_avoidance"],
            axis=Vector3.from_iterable(data["axis"]),
            multiplier=data["multiplier"],
            offset=data["offset"],
            dof_id=from_json(data["id"]),
        )

    @classmethod
    def create_with_dofs(
        cls,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
        axis: Vector3,
        name: Optional[PrefixedName] = None,
        multiplier: float = 1.0,
        offset: float = 0.0,
        *args,
        **kwargs,
    ) -> Self:
        """
        Creates and returns an instance of the class with associated degrees of freedom
        (DOFs) based on the specified parameters. This method facilitates initializing
        a kinematic relationship between a parent and a child entity, augmented by
        an axis representation and configurable properties such as multiplier and offset.

        :param world: The motion world in which to add the degree of freedom.
        :param parent: The parent kinematic structure entity.
        :param child: The child kinematic structure entity.
        :param axis: The axis vector defining the joint relation.
        :param name: Optional specific name for the DOF entity. If not provided, a
                     default name is generated based on the parent and child.
        :param multiplier: A scaling factor applied to the DOF's motion. Defaults to 1.0.
        :param offset: A constant offset value applied to the DOF's motion. Defaults to 0.0.
        :return: An instance of the class representing the defined relationship with
                 its DOF added to the world.
        """
        name = name or cls._generate_default_name(parent=parent, child=child)
        dof = DegreeOfFreedom(name=PrefixedName("dof", str(name)))
        world.add_degree_of_freedom(dof)
        connection = cls(
            parent=parent,
            child=child,
            axis=axis,
            multiplier=multiplier,
            offset=offset,
            dof_id=dof.id,
            *args,
            **kwargs,
        )
        return connection

    def add_to_world(self, world: World):
        super().add_to_world(world)
        if self.multiplier is None:
            self.multiplier = 1
        else:
            self.multiplier = self.multiplier
        if self.offset is None:
            self.offset = 0
        else:
            self.offset = self.offset
        self.axis = self.axis

    @property
    def dof(self) -> DegreeOfFreedom:
        """
        A reference to the Degree of Freedom associated with this connection.
        .. warning:: WITH multiplier and offset applied.
        """
        result = deepcopy(self.raw_dof)
        result.variables = self.raw_dof.variables * self.multiplier
        if self.multiplier < 0:
            # if multiplier is negative, we need to swap the limits
            result.lower_limits, result.upper_limits = (
                result.upper_limits,
                result.lower_limits,
            )
        result.lower_limits = result.lower_limits * self.multiplier
        result.upper_limits = result.upper_limits * self.multiplier

        result.variables.position += self.offset
        if result.lower_limits.position is not None:
            result.lower_limits.position = result.lower_limits.position + self.offset
        if result.upper_limits.position is not None:
            result.upper_limits.position = result.upper_limits.position + self.offset
        return result

    @property
    def raw_dof(self) -> DegreeOfFreedom:
        """
        A reference to the Degree of Freedom associated with this connection.
        .. warning:: WITHOUT multiplier and offset applied.
        """
        return self._world.get_degree_of_freedom_by_id(self.dof_id)

    @property
    def active_dofs(self) -> List[DegreeOfFreedom]:
        return [self.raw_dof]

    @property
    def position(self) -> float:
        return (
            self._world.state[self.raw_dof.id].position * self.multiplier + self.offset
        )

    @position.setter
    def position(self, value: float) -> None:
        self._world.state[self.raw_dof.id].position = (
            value - self.offset
        ) / self.multiplier
        self._world.notify_state_change()

    @property
    def velocity(self) -> float:
        return self._world.state[self.raw_dof.id].velocity * self.multiplier

    @velocity.setter
    def velocity(self, value: float) -> None:
        self._world.state[self.raw_dof.id].velocity = value / self.multiplier
        self._world.notify_state_change()

    @property
    def acceleration(self) -> float:
        return self._world.state[self.raw_dof.id].acceleration * self.multiplier

    @acceleration.setter
    def acceleration(self, value: float) -> None:
        self._world.state[self.raw_dof.id].acceleration = value / self.multiplier
        self._world.notify_state_change()

    @property
    def jerk(self) -> float:
        return self._world.state[self.raw_dof.id].jerk * self.multiplier

    @jerk.setter
    def jerk(self, value: float) -> None:
        self._world.state[self.raw_dof.id].jerk = value / self.multiplier
        self._world.notify_state_change()

    def copy_for_world(self, world: World):
        (
            other_parent,
            other_child,
            parent_T_connection_expression,
            connection_T_child_expression,
        ) = self._find_references_in_world(world)

        return self.__class__(
            name=PrefixedName(self.name.name, self.name.prefix),
            parent=other_parent,
            child=other_child,
            parent_T_connection_expression=parent_T_connection_expression,
            connection_T_child_expression=connection_T_child_expression,
            axis=self.axis,
            multiplier=self.multiplier,
            offset=self.offset,
            dof_id=self.dof_id,
        )


@dataclass(eq=False)
class PrismaticConnection(ActiveConnection1DOF):
    """
    Allows translation along an axis.
    """

    def add_to_world(self, world: World):
        super().add_to_world(world)

        translation_axis = self.axis * self.dof.variables.position
        self._kinematics = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=translation_axis[0],
            y=translation_axis[1],
            z=translation_axis[2],
            child_frame=self.child,
        )


@dataclass(eq=False)
class RevoluteConnection(ActiveConnection1DOF):
    """
    Allows rotation about an axis.
    """

    def add_to_world(self, world: World):
        super().add_to_world(world)

        self._kinematics = HomogeneousTransformationMatrix.from_xyz_axis_angle(
            axis=self.axis,
            angle=self.dof.variables.position,
            child_frame=self.child,
        )


@dataclass(eq=False)
class Connection6DoF(Connection):
    """
    Has full 6 degrees of freedom, that cannot be actively controlled.
    Useful for synchronizing with transformations from external providers.
    """

    x_id: UUID = field(kw_only=True)
    """
    Displacement of child KinematicStructureEntity with respect to parent KinematicStructureEntity along the x-axis.
    """
    y_id: UUID = field(kw_only=True)
    """
    Displacement of child KinematicStructureEntity with respect to parent KinematicStructureEntity along the y-axis.
    """
    z_id: UUID = field(kw_only=True)
    """
    Displacement of child KinematicStructureEntity with respect to parent KinematicStructureEntity along the z-axis.
    """

    qx_id: UUID = field(kw_only=True)
    qy_id: UUID = field(kw_only=True)
    qz_id: UUID = field(kw_only=True)
    qw_id: UUID = field(kw_only=True)
    """
    Rotation of child KinematicStructureEntity with respect to parent KinematicStructureEntity represented as a quaternion.
    """

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["x_id"] = to_json(self.x_id)
        result["y_id"] = to_json(self.y_id)
        result["z_id"] = to_json(self.z_id)
        result["qx_id"] = to_json(self.qx_id)
        result["qy_id"] = to_json(self.qy_id)
        result["qz_id"] = to_json(self.qz_id)
        result["qw_id"] = to_json(self.qw_id)
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)
        parent = tracker.get_world_entity_with_id(id=from_json(data["parent_id"]))
        child = tracker.get_world_entity_with_id(id=from_json(data["child_id"]))
        return cls(
            name=PrefixedName.from_json(data["name"]),
            parent=parent,
            child=child,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_json(
                data["parent_T_connection_expression"], **kwargs
            ),
            x_id=from_json(data["x_id"]),
            y_id=from_json(data["y_id"]),
            z_id=from_json(data["z_id"]),
            qx_id=from_json(data["qx_id"]),
            qy_id=from_json(data["qy_id"]),
            qz_id=from_json(data["qz_id"]),
            qw_id=from_json(data["qw_id"]),
        )

    @property
    def x(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_id(self.x_id)

    @property
    def y(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_id(self.y_id)

    @property
    def z(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_id(self.z_id)

    @property
    def qx(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_id(self.qx_id)

    @property
    def qy(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_id(self.qy_id)

    @property
    def qz(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_id(self.qz_id)

    @property
    def qw(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_id(self.qw_id)

    def add_to_world(self, world: World):
        super().add_to_world(world)
        parent_P_child = Point3(
            x=self.x.variables.position,
            y=self.y.variables.position,
            z=self.z.variables.position,
        )
        parent_R_child = Quaternion(
            x=self.qx.variables.position,
            y=self.qy.variables.position,
            z=self.qz.variables.position,
            w=self.qw.variables.position,
        ).to_rotation_matrix()
        self._kinematics = HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=parent_P_child,
            rotation_matrix=parent_R_child,
            child_frame=self.child,
        )

    @classmethod
    def create_with_dofs(
        cls,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
        name: Optional[PrefixedName] = None,
        parent_T_connection_expression: Optional[
            HomogeneousTransformationMatrix
        ] = None,
        *args,
        **kwargs,
    ) -> Self:
        """
        Creates an instance of the class with automatically generated degrees of freedom (DoFs)
        for the provided parent and child kinematic entities within the specified world.

        This method initializes and adds the required degrees of freedom to the world,
        and sets their properties accordingly. It generates a name for the connection if
        none is provided, and ensures valid initial state for relevant degrees of freedom.

        :param world: The World object where the degrees of freedom are added and modified.
        :param parent: The KinematicStructureEntity serving as the parent.
        :param child: The KinematicStructureEntity serving as the child.
        :param name: An optional PrefixedName for the connection. If None, it will be
                     auto-generated based on the parent and child names.
        :param parent_T_connection_expression: Optional transformation matrix specifying
                                               the connection relationship between parent
                                               and child entities.
        :return: A new instance of the class representing the parent-child connection with
                 automatically defined degrees of freedom.
        """
        name = name or cls._generate_default_name(parent=parent, child=child)

        stringified_name = str(name)
        x = DegreeOfFreedom(name=PrefixedName("x", stringified_name))
        world.add_degree_of_freedom(x)
        y = DegreeOfFreedom(name=PrefixedName("y", stringified_name))
        world.add_degree_of_freedom(y)
        z = DegreeOfFreedom(name=PrefixedName("z", stringified_name))
        world.add_degree_of_freedom(z)
        qx = DegreeOfFreedom(name=PrefixedName("qx", stringified_name))
        world.add_degree_of_freedom(qx)
        qy = DegreeOfFreedom(name=PrefixedName("qy", stringified_name))
        world.add_degree_of_freedom(qy)
        qz = DegreeOfFreedom(name=PrefixedName("qz", stringified_name))
        world.add_degree_of_freedom(qz)
        qw = DegreeOfFreedom(name=PrefixedName("qw", stringified_name))
        world.add_degree_of_freedom(qw)
        world.state[qw.id].position = 1.0

        return cls(
            parent=parent,
            child=child,
            parent_T_connection_expression=parent_T_connection_expression,
            name=name,
            x_id=x.id,
            y_id=y.id,
            z_id=z.id,
            qx_id=qx.id,
            qy_id=qy.id,
            qz_id=qz.id,
            qw_id=qw.id,
        )

    @property
    def passive_dofs(self) -> List[DegreeOfFreedom]:
        return [self.x, self.y, self.z, self.qx, self.qy, self.qz, self.qw]

    @property
    def origin(self) -> HomogeneousTransformationMatrix:
        return super().origin

    @origin.setter
    def origin(
        self, transformation: Union[NpMatrix4x4, HomogeneousTransformationMatrix]
    ) -> None:
        if not isinstance(transformation, HomogeneousTransformationMatrix):
            transformation = HomogeneousTransformationMatrix(data=transformation)
        position = transformation.to_position().to_np()
        orientation = transformation.to_rotation_matrix().to_quaternion().to_np()
        self._world.state[self.x.id].position = position[0]
        self._world.state[self.y.id].position = position[1]
        self._world.state[self.z.id].position = position[2]
        self._world.state[self.qx.id].position = orientation[0]
        self._world.state[self.qy.id].position = orientation[1]
        self._world.state[self.qz.id].position = orientation[2]
        self._world.state[self.qw.id].position = orientation[3]
        self._world.notify_state_change()

    def copy_for_world(self, world: World) -> Connection6DoF:
        """
        Copies this 6DoF connection for another world. Returns a new connection with references to the given world.
        :param world: The world to copy this connection for.
        :return: A copy of this connection for the given world.
        """
        (
            other_parent,
            other_child,
            parent_T_connection_expression,
            connection_T_child_expression,
        ) = self._find_references_in_world(world)

        return Connection6DoF(
            name=deepcopy(self.name),
            parent=other_parent,
            child=other_child,
            parent_T_connection_expression=parent_T_connection_expression,
            connection_T_child_expression=connection_T_child_expression,
            x_id=deepcopy(self.x_id),
            y_id=deepcopy(self.y_id),
            z_id=deepcopy(self.z_id),
            qx_id=deepcopy(self.qx_id),
            qy_id=deepcopy(self.qy_id),
            qz_id=deepcopy(self.qz_id),
            qw_id=deepcopy(self.qw_id),
        )


@dataclass(eq=False)
class OmniDrive(ActiveConnection, HasUpdateState):
    """
    A connection describing an omnidirectional drive.
    It can rotate about its z-axis and drive on the x-y plane simultaneously.
    - x/y: Passive dofs describing the measured odometry with respect to parent frame.
        We assume that the robot can't fly, and we can't measure its z-axis position, so z=0.
        The odometry sensors typically provide velocity measurements with respect to the child frame,
        therefore the velocity values of x/y must stay 0.
    - x_vel/y_vel: The measured and commanded velocity is represented with respect to the child frame with these
        active dofs. It must be ensured that their position values stay 0.
    - roll/pitch: Some robots, like the PR2, have sensors to measure pitch and roll using an IMU,
        we therefore have passive dofs for them.
    - yaw: Since the robot can only rotate about its z-axis, we don't need different dofs for position and velocity of yaw.
        They are combined into one active dof.
    """

    # passive dofs
    x_id: UUID = field(kw_only=True)
    y_id: UUID = field(kw_only=True)
    roll_id: UUID = field(kw_only=True)
    pitch_id: UUID = field(kw_only=True)

    # active dofs
    yaw_id: UUID = field(kw_only=True)
    x_velocity_id: UUID = field(kw_only=True)
    y_velocity_id: UUID = field(kw_only=True)

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["x_id"] = to_json(self.x_id)
        result["y_id"] = to_json(self.y_id)
        result["roll_id"] = to_json(self.roll_id)
        result["pitch_id"] = to_json(self.pitch_id)
        result["yaw_id"] = to_json(self.yaw_id)
        result["x_velocity_id"] = to_json(self.x_velocity_id)
        result["y_velocity_id"] = to_json(self.y_velocity_id)
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)
        parent = tracker.get_world_entity_with_id(id=from_json(data["parent_id"]))
        child = tracker.get_world_entity_with_id(id=from_json(data["child_id"]))
        return cls(
            name=PrefixedName.from_json(data["name"], **kwargs),
            parent=parent,
            child=child,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_json(
                data["parent_T_connection_expression"], **kwargs
            ),
            x_id=from_json(data["x_id"]),
            y_id=from_json(data["y_id"]),
            roll_id=from_json(data["roll_id"]),
            pitch_id=from_json(data["pitch_id"]),
            yaw_id=from_json(data["yaw_id"]),
            x_velocity_id=from_json(data["x_velocity_id"]),
            y_velocity_id=from_json(data["y_velocity_id"]),
        )

    @property
    def x(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_id(self.x_id)

    @property
    def y(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_id(self.y_id)

    @property
    def roll(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_id(self.roll_id)

    @property
    def pitch(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_id(self.pitch_id)

    @property
    def yaw(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_id(self.yaw_id)

    @property
    def x_velocity(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_id(self.x_velocity_id)

    @property
    def y_velocity(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_id(self.y_velocity_id)

    def add_to_world(self, world: World):
        super().add_to_world(world)
        odom_T_bf = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=self.x.variables.position,
            y=self.y.variables.position,
            yaw=self.yaw.variables.position,
        )
        bf_T_bf_vel = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=self.x_velocity.variables.position, y=self.y_velocity.variables.position
        )
        bf_vel_T_bf = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0,
            y=0,
            z=0,
            roll=self.roll.variables.position,
            pitch=self.pitch.variables.position,
            yaw=0,
        )
        self._kinematics = odom_T_bf @ bf_T_bf_vel @ bf_vel_T_bf
        self._kinematics.child_frame = self.child

    @classmethod
    def create_with_dofs(
        cls,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
        name: Optional[PrefixedName] = None,
        parent_T_connection_expression: Optional[
            HomogeneousTransformationMatrix
        ] = None,
        translation_velocity_limits: float = 0.6,
        rotation_velocity_limits: float = 0.5,
        *args,
        **kwargs,
    ) -> Self:
        """
        Creates an instance of the class with automatically generated degrees of freedom
        (DOFs) for translation on the x and y axes, rotation along roll, pitch, and yaw
        axes, and velocity limits for translation and rotation.

        This method modifies the provided world to add all required degrees of freedom
        and their limits, based on the provided settings. Names for the degrees of
        freedom are auto-generated using the stringified version of the provided name
        or its default setting.

        :param world: The world where the configuration is being applied, and degrees of freedom are added.
        :param parent: The parent kinematic structure entity.
        :param child: The child kinematic structure entity.
        :param name: Name of the connection. If None, it will be auto-generated.
        :param parent_T_connection_expression: Transformation matrix representing the
            relative position/orientation of the child to the parent. Default is Identity.
        :param translation_velocity_limits: The velocity limit applied to the
            translation degrees of freedom (default is 0.6).
        :param rotation_velocity_limits: The velocity limit applied to the rotation
            degrees of freedom (default is 0.5).
        :return: An instance of the class with the auto-generated DOFs incorporated.
        """
        name = name or cls._generate_default_name(parent=parent, child=child)
        stringified_name = str(name)
        lower_translation_limits = DerivativeMap()
        lower_translation_limits.velocity = -translation_velocity_limits
        upper_translation_limits = DerivativeMap()
        upper_translation_limits.velocity = translation_velocity_limits
        lower_rotation_limits = DerivativeMap()
        lower_rotation_limits.velocity = -rotation_velocity_limits
        upper_rotation_limits = DerivativeMap()
        upper_rotation_limits.velocity = rotation_velocity_limits

        x = DegreeOfFreedom(name=PrefixedName("x", stringified_name))
        world.add_degree_of_freedom(x)
        y = DegreeOfFreedom(name=PrefixedName("y", stringified_name))
        world.add_degree_of_freedom(y)
        roll = DegreeOfFreedom(name=PrefixedName("roll", stringified_name))
        world.add_degree_of_freedom(roll)
        pitch = DegreeOfFreedom(name=PrefixedName("pitch", stringified_name))
        world.add_degree_of_freedom(pitch)
        yaw = DegreeOfFreedom(
            name=PrefixedName("yaw", stringified_name),
            lower_limits=lower_rotation_limits,
            upper_limits=upper_rotation_limits,
        )
        world.add_degree_of_freedom(yaw)

        x_vel = DegreeOfFreedom(
            name=PrefixedName("x_vel", stringified_name),
            lower_limits=lower_translation_limits,
            upper_limits=upper_translation_limits,
        )
        world.add_degree_of_freedom(x_vel)
        y_vel = DegreeOfFreedom(
            name=PrefixedName("y_vel", stringified_name),
            lower_limits=lower_translation_limits,
            upper_limits=upper_translation_limits,
        )
        world.add_degree_of_freedom(y_vel)

        return cls(
            parent=parent,
            child=child,
            parent_T_connection_expression=parent_T_connection_expression,
            name=name,
            x_id=x.id,
            y_id=y.id,
            roll_id=roll.id,
            pitch_id=pitch.id,
            yaw_id=yaw.id,
            x_velocity_id=x_vel.id,
            y_velocity_id=y_vel.id,
            *args,
            **kwargs,
        )

    @property
    def active_dofs(self) -> List[DegreeOfFreedom]:
        return [self.x_velocity, self.y_velocity, self.yaw]

    @property
    def passive_dofs(self) -> List[DegreeOfFreedom]:
        return [self.x, self.y, self.roll, self.pitch]

    @property
    def dofs(self) -> List[DegreeOfFreedom]:
        return self.active_dofs + self.passive_dofs

    def update_state(self, dt: float) -> None:
        state = self._world.state
        state[self.x_velocity.id].position = 0
        state[self.y_velocity.id].position = 0

        x_vel = state[self.x_velocity.id].velocity
        y_vel = state[self.y_velocity.id].velocity
        delta = state[self.yaw.id].position
        x_velocity = np.cos(delta) * x_vel - np.sin(delta) * y_vel
        state[self.x.id].position += x_velocity * dt
        y_velocity = np.sin(delta) * x_vel + np.cos(delta) * y_vel
        state[self.y.id].position += y_velocity * dt

    @property
    def origin(self) -> HomogeneousTransformationMatrix:
        return super().origin

    @origin.setter
    def origin(
        self, transformation: Union[NpMatrix4x4, HomogeneousTransformationMatrix]
    ) -> None:
        """
        Overwrites the origin of the connection.
        .. warning:: Ignores z position, pitch, and yaw values.
        :param parent_T_child:
        """
        if isinstance(transformation, np.ndarray):
            transformation = HomogeneousTransformationMatrix(data=transformation)
        position = transformation.to_position()
        roll, pitch, yaw = transformation.to_rotation_matrix().to_rpy()
        self._world.state[self.x.id].position = position.x
        self._world.state[self.y.id].position = position.y
        self._world.state[self.yaw.id].position = yaw
        self._world.notify_state_change()

    def get_free_variable_names(self) -> List[UUID]:
        return [self.x.id, self.y.id, self.yaw.id]

    @property
    def has_hardware_interface(self) -> bool:
        return self.x_velocity.has_hardware_interface

    @has_hardware_interface.setter
    def has_hardware_interface(self, value: bool) -> None:
        self.x_velocity.has_hardware_interface = value
        self.y_velocity.has_hardware_interface = value
        self.yaw.has_hardware_interface = value

    def copy_for_world(self, world: World) -> OmniDrive:
        """
        Copies this OmniDriveConnection for the provided world. This finds the references for the parent and child in
        the new world and returns a new connection with references to the new parent and child.
        :param world: The world where the connection is copied.
        :return: The connection with references to the new parent and child.
        """
        (
            other_parent,
            other_child,
            parent_T_connection_expression,
            connection_T_child_expression,
        ) = self._find_references_in_world(world)

        return OmniDrive(
            name=deepcopy(self.name),
            parent=other_parent,
            child=other_child,
            parent_T_connection_expression=parent_T_connection_expression,
            connection_T_child_expression=connection_T_child_expression,
            x_id=deepcopy(self.x_id),
            y_id=deepcopy(self.y_id),
            roll_id=deepcopy(self.roll_id),
            pitch_id=deepcopy(self.pitch_id),
            yaw_id=deepcopy(self.yaw_id),
            x_velocity_id=deepcopy(self.x_velocity_id),
            y_velocity_id=deepcopy(self.y_velocity_id),
        )
