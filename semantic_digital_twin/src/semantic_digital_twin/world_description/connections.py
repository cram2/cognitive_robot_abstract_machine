"""
The connections between kinematic structure entities: fixed ones, ones that move their
child along a screw axis, ones with six degrees of freedom, and wheeled drives.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field, replace
from uuid import UUID

import krrood.symbolic_math.symbolic_math as sm
import numpy as np
from typing_extensions import TYPE_CHECKING, Self

from krrood.patterns.classproperty import classproperty

from semantic_digital_twin.world_description.connection_properties import (
    JointDynamics,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.world_entity import (
    Connection,
    KinematicStructureEntity,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.types import NpMatrix4x4
from semantic_digital_twin.exceptions import (
    ScrewAxisWithoutTranslationError,
    UnnormalizedScrewAxisError,
)
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix3Dimensional,
    Point3Dimensional,
    Quaternion,
    RotationMatrix,
    ScrewAxis,
    Vector3Dimensional,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


class HasUpdateState(ABC):
    """
    Mixin class for connections that need state updated which are not trivial
    integrations.

    Typically needed for connections that use active and passive degrees of freedom.
    Look at OmniDrive for an example usage.
    """

    @abstractmethod
    def update_state(self, dt: float) -> None:
        """
        Allows the connection to update the state of its dofs.

        An integration update for active dofs will have happened before this method is
        called. Write directly into self._world.state, but don't touch dofs that don't
        belong to this connection.
        :param dt: Time passed since last update.
        """
        pass


@dataclass(eq=False)
class FixedConnection(Connection):
    """
    Has 0 degrees of freedom.
    """

    @classmethod
    def create_with_dofs(
        cls,
        *,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
        name: PrefixedName | None = None,
        parent_T_connection_expression: HomogeneousTransformationMatrix3Dimensional
        | None = None,
        connection_T_child_expression: HomogeneousTransformationMatrix3Dimensional
        | None = None,
    ) -> Self:
        return cls(
            parent=parent,
            child=child,
            name=name,
            parent_T_connection_expression=parent_T_connection_expression,
            connection_T_child_expression=connection_T_child_expression,
        )


@dataclass(eq=False)
class ActiveConnection(Connection, ABC):
    """
    Has one or more degrees of freedom that can be actively controlled, e.g., robot
    joints.
    """

    @property
    def has_hardware_interface(self) -> bool:
        """
        Whether this connection is linked to a controller and can therefore respond to
        control commands.

        E.g. the caster wheels of a PR2 are active, because they have a DOF, but they
        are not directly controlled. Instead a the omni drive connection is directly
        controlled and a low level controller translates these commands to commands for
        the caster wheels.

        A door hinge is also active but cannot be controlled.
        """
        return any(dof.has_hardware_interface for dof in self.active_dofs)

    @property
    def is_controlled(self):
        return self.has_hardware_interface


@dataclass(eq=False)
class ActiveConnection1DOF(ActiveConnection, ABC):
    """
    Superclass for active connections with 1 degree of freedom.
    """

    multiplier: float = 1.0
    """
    The position is multiplied by this value, e.g. for shared DoFs.

    .. warning:: Set it only before the connection is added to a world: a later
        assignment is not recorded, published or rolled back. Replace the connection.
    """

    offset: float = 0.0
    """
    The position is offset by this value, e.g. for shared DoFs.

    .. warning:: Set it only before the connection is added to a world: a later
        assignment is not recorded, published or rolled back. Replace the connection.
    """

    raw_dof: DegreeOfFreedom = field(kw_only=True)
    """
    The degree of freedom whose raw, unscaled state this connection drives.

    Use the :attr:`dof` property to obtain it with ``multiplier`` and ``offset``
    applied.
    """

    dynamics: JointDynamics = field(default_factory=JointDynamics)
    """
    Dynamic properties of the joint.
    """

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
            result.limits.lower, result.limits.upper = (
                result.limits.upper,
                result.limits.lower,
            )
        result.limits.lower = result.limits.lower * self.multiplier
        result.limits.upper = result.limits.upper * self.multiplier

        result.variables.position += self.offset
        if result.limits.lower.position is not None:
            result.limits.lower.position = result.limits.lower.position + self.offset
        if result.limits.upper.position is not None:
            result.limits.upper.position = result.limits.upper.position + self.offset
        return result

    @property
    def active_dofs(self) -> list[DegreeOfFreedom]:
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


@dataclass(eq=False, kw_only=True)
class ScrewAxisConnection(ActiveConnection1DOF):
    """
    Moves the child along a screw axis, in proportion to the position, so revolute,
    prismatic and helical connections differ only in their :attr:`screw_axis`, which may
    lie anywhere in the connection frame, as `Screw axis connections
    <https://cram2.github.io/cognitive_robot_abstract_machine/semantic_digital_twin/examples/screw_axis_connections.html>`__ explains.
    """

    screw_axis: ScrewAxis
    """
    How the child's pose changes per unit of position, in the connection frame,
    `normalized <https://cram2.github.io/cognitive_robot_abstract_machine/semantic_digital_twin/examples/screw_axis_connections.html#why-the-screw-axis-is-normalized>`__
    as in :cite:t:`lynch2017modern` (definition 3.24).

    .. warning:: Set it only before the connection is added to a world: a later
        assignment is not recorded, published or rolled back. Replace the connection.
    """

    def __post_init__(self) -> None:
        """
        :raises UnnormalizedScrewAxisError: if :attr:`screw_axis` is not normalized,
            see :attr:`~krrood_spatial_math.spatial_types.ScrewAxis.is_normalized`.
        """
        super().__post_init__()
        if not self.screw_axis.is_normalized:
            raise UnnormalizedScrewAxisError(screw_axis=self.screw_axis)

    @classmethod
    def create_revolute_screw_axis(cls, *, axis: Vector3Dimensional) -> ScrewAxis:
        """
        :param axis: The unit vector the child rotates about, in the connection frame.
        :return: The screw axis of a revolute connection, such as a door's hinge.
        """
        return ScrewAxis(rotation_per_unit_position=axis)

    @classmethod
    def create_prismatic_screw_axis(cls, *, axis: Vector3Dimensional) -> ScrewAxis:
        """
        :param axis: The unit vector the child translates along, in the connection
            frame.
        :return: The screw axis of a prismatic connection, such as a drawer's.
        """
        return ScrewAxis(translation_per_unit_position=axis)

    @classmethod
    def create_helical_screw_axis(
        cls, *, axis: Vector3Dimensional, pitch: float
    ) -> ScrewAxis:
        """
        :param axis: The unit vector the child rotates about and translates along, in
            the connection frame.
        :param pitch: How far the child advances along ``axis`` per revolution, in
            meters, as the pitch of a single-start thread
            :cite:p:`wikipedia_screw_thread` (section "Lead, pitch, and starts");
            negative for a left-handed thread. The pitch of :cite:t:`lynch2017modern` is
            per radian: ``pitch / (2 * pi)``.
        :return: The screw axis of a helical connection, such as a bottle cap's.
        """
        return ScrewAxis(
            rotation_per_unit_position=axis,
            translation_per_unit_position=axis * (pitch / (2 * np.pi)),
        )

    @classmethod
    def create_with_dofs(
        cls,
        *,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
        name: PrefixedName | None = None,
        parent_T_connection_expression: HomogeneousTransformationMatrix3Dimensional
        | None = None,
        connection_T_child_expression: HomogeneousTransformationMatrix3Dimensional
        | None = None,
        multiplier: float = 1.0,
        offset: float = 0.0,
        dof_limits: DegreeOfFreedomLimits | None = None,
        dynamics: JointDynamics | None = None,
        screw_axis: ScrewAxis,
    ) -> Self:
        """
        Creates and returns a screw axis connection with its single degree of freedom
        added to ``world``.

        :param world: The world in which to add the degree of freedom.
        :param parent: The parent kinematic structure entity.
        :param child: The child kinematic structure entity.
        :param name: Optional specific name for the connection. If not provided, a
            default name is generated based on the parent and child.
        :param parent_T_connection_expression: Constant pose of the connection relative
            to its parent.
        :param connection_T_child_expression: Constant pose of the child relative to the
            connection.
        :param multiplier: A scaling factor applied to the degree of freedom's motion.
        :param offset: A constant offset applied to the degree of freedom's motion.
        :param dof_limits: Optional limits for the generated degree of freedom.
        :param dynamics: The dynamic properties of the joint. Frictionless and undamped
            if None.
        :param screw_axis: How the child's pose changes per unit of position, see
            :attr:`screw_axis`.
        """
        name = name or cls._generate_default_name(parent=parent, child=child)
        dof = DegreeOfFreedom(name=PrefixedName("dof", str(name)), limits=dof_limits)
        world.add_degree_of_freedom(dof)
        return cls(
            name=name,
            parent=parent,
            child=child,
            parent_T_connection_expression=parent_T_connection_expression,
            connection_T_child_expression=connection_T_child_expression,
            multiplier=multiplier,
            offset=offset,
            raw_dof=dof,
            dynamics=dynamics or JointDynamics(),
            screw_axis=screw_axis,
        )

    def add_to_world(self, *, world: World) -> None:
        """
        Extends :meth:`WorldEntity.add_to_world` by building the motion along
        :attr:`screw_axis` from the position, as `The transformation matrix
        <https://cram2.github.io/cognitive_robot_abstract_machine/semantic_digital_twin/examples/screw_axis_connections.html#the-transformation-matrix>`__
        explains.
        """
        super().add_to_world(world=world)

        position = self.dof.variables.position
        rotation = self.screw_axis.rotation_per_unit_position
        translation = self.screw_axis.translation_per_unit_position
        sine = sm.sin(position)
        one_minus_cosine = 1 - sm.cos(position)
        turned_axes = [
            axis
            + rotation.cross(other=axis) * sine
            + rotation.cross(other=rotation.cross(other=axis)) * one_minus_cosine
            for axis in (
                Vector3Dimensional.X(),
                Vector3Dimensional.Y(),
                Vector3Dimensional.Z(),
            )
        ]
        rotation_cross_translation = rotation.cross(other=translation)
        connection_V_child = (
            translation * position
            + rotation_cross_translation * one_minus_cosine
            + rotation.cross(other=rotation_cross_translation) * (position - sine)
        )
        self._kinematics = (
            HomogeneousTransformationMatrix3Dimensional.from_point_rotation_matrix(
                point=connection_V_child.point_3_dimensional,
                rotation_matrix=RotationMatrix(
                    data=sm.hstack([*turned_axes, Point3Dimensional(0, 0, 0)])
                ),
                child_frame=self.child,
            )
        )

    def position_for_translation(self, *, distance: float) -> float:
        """
        :param distance: How far the child is to advance along the direction of the
            screw axis, in meters; negative to move it back.
        :return: The position at which the child has advanced ``distance`` from its pose
            at position zero, which is also the change of position that advances it by
            ``distance``.
        :raises ScrewAxisWithoutTranslationError: if the child does not advance.
        """
        if not self._advances_child:
            raise ScrewAxisWithoutTranslationError(screw_axis=self.screw_axis)
        translation_per_position = self.screw_axis.translation_per_unit_position.dot(
            self.screw_axis.unit_direction
        )
        return distance / float(translation_per_position)

    @property
    def connection_T_frame_on_axis(self) -> HomogeneousTransformationMatrix3Dimensional:
        """
        The pose, relative to the connection frame, of the frame of the same orientation
        whose origin is the point of the screw axis closest to the connection frame's
        origin; the connection frame itself for an axis through that origin, or for a
        child that does not rotate, whose axis has no line.
        """
        return HomogeneousTransformationMatrix3Dimensional.from_point_rotation_matrix(
            point=self.screw_axis.axis_point_closest_to_origin
        )

    def move_connection_frame_onto_axis(self) -> Self:
        """
        Replaces this connection, in its world, by one whose connection frame lies on the
        screw axis, as parsed joints have it, and which moves the child exactly as this
        one does: :attr:`parent_T_connection_expression` moves by
        :attr:`connection_T_frame_on_axis`, :attr:`connection_T_child_expression` by its
        inverse, and the screw axis keeps its direction and its pitch through the moved
        frame's origin. The replacement keeps the name and the degree of freedom. A
        connection whose screw axis already passes through its frame's origin is left as
        it is, without a model change.

        :return: The replacement, or this connection if it is left as it is.
        """
        if self.screw_axis.passes_through_origin:
            return self
        world = self._world
        connection_T_frame_on_axis = self.connection_T_frame_on_axis
        rotation = self.screw_axis.rotation_per_unit_position
        moved = replace(
            self,
            name=PrefixedName(self.name.name, prefix=self.name.prefix),
            parent_T_connection_expression=self.parent_T_connection_expression
            @ connection_T_frame_on_axis,
            connection_T_child_expression=connection_T_frame_on_axis.inverse()
            @ self.connection_T_child_expression,
            screw_axis=ScrewAxis(
                rotation_per_unit_position=rotation,
                translation_per_unit_position=rotation
                * self.screw_axis.translation_per_radian,
                reference_frame=self.screw_axis.reference_frame,
            ),
            simulator_additional_properties=deepcopy(
                self.simulator_additional_properties
            ),
        )
        with world.modify_world():
            world.remove_connection(self)
            world.add_connection(moved)
        return moved

    @classproperty
    def smallest_rotation_per_unit_position(cls) -> float:
        """
        Radians per unit of position up to which the child counts as not rotating.

        A normalized screw axis rotates the child by one radian per unit of position or
        not at all, so this only absorbs rounding errors, up to the same tolerance as
        the normalization.
        """
        return ScrewAxis.normalization_tolerance

    @classproperty
    def smallest_advance_per_radian(cls) -> float:
        """
        Meters per radian below which a rotating child counts as not advancing along the
        axis: a micrometer, about 6 micrometers per revolution, far below the pitch of any
        thread and far above rounding errors, such as those of a screw axis estimated
        from observed poses.
        """
        return 1e-6

    @property
    def behaves_like_a_revolute_connection(self) -> bool:
        """
        Whether the child only rotates about the axis, advancing along it by less than
        :attr:`smallest_advance_per_radian`.
        """
        return self._rotates_child and not self._advances_child

    @property
    def behaves_like_a_prismatic_connection(self) -> bool:
        """
        Whether the child only translates along the axis, rotating by no more than
        :attr:`smallest_rotation_per_unit_position`.
        """
        return not self._rotates_child

    @property
    def behaves_like_a_helical_connection(self) -> bool:
        """
        Whether the child rotates about the axis and advances along it at once, by at
        least :attr:`smallest_advance_per_radian`, as a thread does.
        """
        return self._rotates_child and self._advances_child

    @property
    def _rotates_child(self) -> bool:
        """
        Whether the position rotates the child by more than
        :attr:`smallest_rotation_per_unit_position`.
        """
        return self.screw_axis.turns_body(
            tolerance=self.smallest_rotation_per_unit_position
        )

    @property
    def _advances_child(self) -> bool:
        """
        Whether the child advances along the axis: by at least
        :attr:`smallest_advance_per_radian` per radian if it rotates, and always if it
        does not, since its normalized screw axis then only translates it.

        Unlike whether the point at the frame's origin moves, this holds wherever the
        axis lies.
        """
        if not self._rotates_child:
            return True
        return bool(
            abs(float(self.screw_axis.translation_per_radian))
            >= self.smallest_advance_per_radian
        )


@dataclass(eq=False)
class Connection6DoF(Connection):
    """
    Has full 6 degrees of freedom, that cannot be actively controlled.

    Useful for synchronizing with transformations from external providers.
    """

    x: DegreeOfFreedom = field(kw_only=True)
    """
    Displacement of child KinematicStructureEntity with respect to parent
    KinematicStructureEntity along the x-axis.
    """

    y: DegreeOfFreedom = field(kw_only=True)
    """
    Displacement of child KinematicStructureEntity with respect to parent
    KinematicStructureEntity along the y-axis.
    """

    z: DegreeOfFreedom = field(kw_only=True)
    """
    Displacement of child KinematicStructureEntity with respect to parent
    KinematicStructureEntity along the z-axis.
    """

    qx: DegreeOfFreedom = field(kw_only=True)
    qy: DegreeOfFreedom = field(kw_only=True)
    qz: DegreeOfFreedom = field(kw_only=True)
    qw: DegreeOfFreedom = field(kw_only=True)
    """
    Rotation of child KinematicStructureEntity with respect to parent
    KinematicStructureEntity represented as a quaternion.
    """

    def add_to_world(self, *, world: World):
        """
        Extends :meth:`WorldEntity.add_to_world` by building the pose from the
        translation and quaternion degrees of freedom.
        """
        super().add_to_world(world=world)
        parent_P_child = Point3Dimensional(
            x=self.x.variables.position,
            y=self.y.variables.position,
            z=self.z.variables.position,
        )
        parent_R_child = Quaternion(
            x=self.qx.variables.position,
            y=self.qy.variables.position,
            z=self.qz.variables.position,
            w=self.qw.variables.position,
        ).to_coordinate_change()
        self._kinematics = (
            HomogeneousTransformationMatrix3Dimensional.from_point_rotation_matrix(
                point=parent_P_child,
                rotation_matrix=parent_R_child,
                child_frame=self.child,
            )
        )

    @classmethod
    def create_with_dofs(
        cls,
        *,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
        name: PrefixedName | None = None,
        parent_T_connection_expression: HomogeneousTransformationMatrix3Dimensional
        | None = None,
        connection_T_child_expression: HomogeneousTransformationMatrix3Dimensional
        | None = None,
    ) -> Self:
        """
        Creates an instance of the class with automatically generated degrees of freedom
        (DoFs) for the provided parent and child kinematic entities within the specified
        world.

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
                                               and the connection.
        :param connection_T_child_expression: Optional transformation matrix specifying
                                              the connection relationship between the connection
                                              and its child entities
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
            connection_T_child_expression=connection_T_child_expression,
            name=name,
            x=x,
            y=y,
            z=z,
            qx=qx,
            qy=qy,
            qz=qz,
            qw=qw,
        )

    @property
    def passive_dofs(self) -> list[DegreeOfFreedom]:
        return [self.x, self.y, self.z, self.qx, self.qy, self.qz, self.qw]

    @property
    def origin(self) -> HomogeneousTransformationMatrix3Dimensional:
        return super().origin

    @origin.setter
    def origin(
        self, transformation: HomogeneousTransformationMatrix3Dimensional
    ) -> None:
        """
        Set this connection's parent-to-child origin, un-composing the fixed expressions
        around ``_kinematics`` to find the DOF state that produces it. The writes are
        held together under ``World._world_lock``, so no observer sees a half applied
        pose.

        :param transformation: The desired parent-to-child origin. Must carry a reference
            frame, else :class:`~krrood_spatial_math.exceptions.MissingReferenceFrameError`
            is raised; need not be expressed in the parent frame.
        """
        local_kinematics = self._calculate_local_kinematics(
            self._world.transform(transformation, self.parent)
        )
        position = local_kinematics.position
        orientation = local_kinematics.rotation_matrix.quaternion
        with self._world._world_lock:
            self._world.state[self.x.id].position = position[0]
            self._world.state[self.y.id].position = position[1]
            self._world.state[self.z.id].position = position[2]
            self._world.state[self.qx.id].position = orientation[0]
            self._world.state[self.qy.id].position = orientation[1]
            self._world.state[self.qz.id].position = orientation[2]
            self._world.state[self.qw.id].position = orientation[3]
            self._world.notify_state_change()


@dataclass(eq=False)
class WheeledDrive(ActiveConnection, HasUpdateState, ABC):
    """
    Superclass for connections that describe a drive, e.g., an omnidirectional drive or
    a differential drive.
    """

    x: DegreeOfFreedom = field(kw_only=True)
    """
    Passive DoFs describing the measured odometry in x with respect to parent frame.
    """

    y: DegreeOfFreedom = field(kw_only=True)
    """
    Passive DoFs describing the measured odometry in y with respect to parent frame.
    """

    roll: DegreeOfFreedom = field(kw_only=True)
    """
    Passive DoF describing the measured odometry in roll using the IMU sensor.
    """

    pitch: DegreeOfFreedom = field(kw_only=True)
    """
    Passive DoF describing the measured odometry in pitch using the IMU sensor.
    """

    yaw: DegreeOfFreedom = field(kw_only=True)
    """
    Active DoF describing rotation around the robot's z-axis.
    """

    @property
    def keeps_child_on_parent_plane(self) -> bool:
        """
        Always, unlike :attr:`~semantic_digital_twin.world_description.world_entity.Conn
        ection.keeps_child_on_parent_plane`:

        a drive moves its child on its parent's plane, and the roll and pitch it
        measures tilt the child without moving it off that plane.
        """
        return True

    @property
    def origin(self) -> HomogeneousTransformationMatrix3Dimensional:
        return super().origin

    @origin.setter
    def origin(
        self, transformation: NpMatrix4x4 | HomogeneousTransformationMatrix3Dimensional
    ) -> None:
        """
        Overwrites the origin of the connection, un-composing the fixed expressions
        around ``_kinematics`` to find the drive's state that produces it. The writes
        are held together under ``World._world_lock``.

        .. warning:: Ignores z position, pitch, and roll values, because a drive cannot
            reach them.

        :param transformation: The desired parent-to-child origin. One given in a frame is
            expressed in the parent frame first; one given in none is taken to be in it.
        """
        if (
            isinstance(transformation, HomogeneousTransformationMatrix3Dimensional)
            and transformation.reference_frame is not None
        ):
            transformation = self._world.transform(transformation, self.parent)
        local_kinematics = self._calculate_local_kinematics(transformation)
        position = local_kinematics.position
        roll, pitch, yaw = local_kinematics.rotation_matrix.rpy
        with self._world._world_lock:
            self._world.state[self.x.id].position = position.x
            self._world.state[self.y.id].position = position.y
            self._world.state[self.yaw.id].position = yaw
            self._world.notify_state_change()


@dataclass(eq=False)
class OmniDrive(WheeledDrive):
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

    x_velocity: DegreeOfFreedom = field(kw_only=True)
    """
    Active DoF describing the measured and commanded velocity in x.

    Represented with respect to the child frame.
    """

    y_velocity: DegreeOfFreedom = field(kw_only=True)
    """
    Active DoF describing the measured and commanded velocity in y.

    Represented with respect to the child frame.
    """

    def add_to_world(self, *, world: World):
        """
        Extends :meth:`WorldEntity.add_to_world` by building the planar pose and the
        velocity frames from the degrees of freedom.
        """
        super().add_to_world(world=world)
        odom_T_bf = HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
            x=self.x.variables.position,
            y=self.y.variables.position,
            yaw=self.yaw.variables.position,
        )
        bf_T_bf_vel = HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
            x=self.x_velocity.variables.position, y=self.y_velocity.variables.position
        )
        bf_vel_T_bf = HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
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
        *,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
        name: PrefixedName | None = None,
        parent_T_connection_expression: HomogeneousTransformationMatrix3Dimensional
        | None = None,
        connection_T_child_expression: HomogeneousTransformationMatrix3Dimensional
        | None = None,
        translation_velocity_limits: float = 0.6,
        rotation_velocity_limits: float = 0.5,
    ) -> Self:
        """
        Creates an instance of the class with automatically generated degrees of freedom
        (DOFs) for translation on the x and y axes, rotation along roll, pitch, and yaw
        axes, and velocity limits for translation and rotation.

        This method modifies the provided world to add all required degrees of freedom
        and their limits, based on the provided settings. Names for the degrees of
        freedom are auto-generated using the stringified version of the provided name or
        its default setting.

        :param world: The world where the configuration is being applied, and degrees of
            freedom are added.
        :param parent: The parent kinematic structure entity.
        :param child: The child kinematic structure entity.
        :param name: Name of the connection. If None, it will be auto-generated.
        :param parent_T_connection_expression: Transformation matrix representing the
            relative position/orientation of the child to the parent. Default is
            Identity.
        :param connection_T_child_expression: Transformation matrix representing the
            relative position/orientation of the child to the connection. Default is
            Identity.
        :param translation_velocity_limits: The velocity limit applied to the
            translation degrees of freedom (default is 0.6).
        :param rotation_velocity_limits: The velocity limit applied to the rotation
            degrees of freedom (default is 0.5).
        :return: An instance of the class with the auto-generated DOFs incorporated.
        """
        name = name or cls._generate_default_name(parent=parent, child=child)
        stringified_name = str(name)

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
            limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                maximum_speed=rotation_velocity_limits
            ),
        )
        world.add_degree_of_freedom(yaw)

        x_vel = DegreeOfFreedom(
            name=PrefixedName("x_vel", stringified_name),
            limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                maximum_speed=translation_velocity_limits
            ),
        )
        world.add_degree_of_freedom(x_vel)
        y_vel = DegreeOfFreedom(
            name=PrefixedName("y_vel", stringified_name),
            limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                maximum_speed=translation_velocity_limits
            ),
        )
        world.add_degree_of_freedom(y_vel)

        return cls(
            parent=parent,
            child=child,
            parent_T_connection_expression=parent_T_connection_expression,
            connection_T_child_expression=connection_T_child_expression,
            name=name,
            x=x,
            y=y,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            x_velocity=x_vel,
            y_velocity=y_vel,
        )

    @property
    def active_dofs(self) -> list[DegreeOfFreedom]:
        return [self.x_velocity, self.y_velocity, self.yaw]

    @property
    def passive_dofs(self) -> list[DegreeOfFreedom]:
        return [self.x, self.y, self.roll, self.pitch]

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

    def get_free_variable_names(self) -> list[UUID]:
        return [self.x.id, self.y.id, self.yaw.id]


@dataclass(eq=False)
class DifferentialDrive(WheeledDrive):
    """
    A connection describing a differential drive.

    It can rotate around its z-axis and drive in x-direction. It allows movement in the
    x-y plane.
    """

    x_velocity: DegreeOfFreedom = field(kw_only=True)
    """
    Active DoF describing the measured and commanded velocity in x.

    Represented with respect to the child frame.
    """

    def add_to_world(self, *, world: World):
        """
        Extends :meth:`WorldEntity.add_to_world` by building the planar pose and the
        velocity frames from the degrees of freedom.
        """
        super().add_to_world(world=world)
        odom_T_bf = HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
            x=self.x.variables.position,
            y=self.y.variables.position,
            yaw=self.yaw.variables.position,
        )
        bf_T_bf_vel = HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
            x=self.x_velocity.variables.position,
        )
        bf_vel_T_bf = HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
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
        *,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
        name: PrefixedName | None = None,
        parent_T_connection_expression: HomogeneousTransformationMatrix3Dimensional
        | None = None,
        connection_T_child_expression: HomogeneousTransformationMatrix3Dimensional
        | None = None,
        translation_velocity_limits: float = 0.6,
        rotation_velocity_limits: float = 0.5,
    ) -> Self:
        """
        Creates an instance of the class with automatically generated DoFs for
        translation on the x-axis, rotation along roll, pitch, and yaw axes, and
        velocity limits for translation and rotation.

        :param world: The world where the configuration is being applied, and degrees of
            freedom are added.
        :param parent: The parent kinematic structure entity.
        :param child: The child kinematic structure entity.
        :param name: Name of the connection. If None, it will be auto-generated.
        :param parent_T_connection_expression: Transformation matrix representing the
            relative position/orientation of the child to the parent. Default is
            Identity.
        :param connection_T_child_expression: Transformation matrix representing the
            relative position/orientation of the child to the connection. Default is
            Identity.
        :param translation_velocity_limits: The velocity limit applied to the
            translation degrees of freedom (default is 0.6).
        :param rotation_velocity_limits: The velocity limit applied to the rotation
            degrees of freedom (default is 0.5).
        :return: An instance of the class with the auto-generated DOFs incorporated.
        """
        name = name or cls._generate_default_name(parent=parent, child=child)
        stringified_name = str(name)

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
            limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                maximum_speed=rotation_velocity_limits
            ),
        )
        world.add_degree_of_freedom(yaw)

        x_vel = DegreeOfFreedom(
            name=PrefixedName("x_vel", stringified_name),
            limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                maximum_speed=translation_velocity_limits
            ),
        )
        world.add_degree_of_freedom(x_vel)

        return cls(
            parent=parent,
            child=child,
            parent_T_connection_expression=parent_T_connection_expression,
            connection_T_child_expression=connection_T_child_expression,
            name=name,
            x=x,
            y=y,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            x_velocity=x_vel,
        )

    @property
    def active_dofs(self) -> list[DegreeOfFreedom]:
        return [self.x_velocity, self.yaw]

    @property
    def passive_dofs(self) -> list[DegreeOfFreedom]:
        return [self.x, self.y, self.roll, self.pitch]

    def update_state(self, dt: float) -> None:
        state = self._world.state
        state[self.x_velocity.id].position = 0

        x_vel = state[self.x_velocity.id].velocity
        delta = state[self.yaw.id].position
        x_velocity = np.cos(delta) * x_vel
        state[self.x.id].position += x_velocity * dt
        y_velocity = np.sin(delta) * x_vel
        state[self.y.id].position += y_velocity * dt

    def get_free_variable_names(self) -> list[UUID]:
        return [self.x.id, self.y.id, self.yaw.id]
