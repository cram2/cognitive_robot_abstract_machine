"""
Specifications of the connections that join two kinematic structure entities.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field, fields
from typing import Any, Generic, Type, cast

from typing_extensions import TypeVar

from krrood.class_diagrams.attribute_introspector import DataclassOnlyIntrospector
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.utils import get_generic_type_parameters
from semantic_digital_twin.exceptions import MissingConnectionParentError
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.specifications.base import NamedSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
    ScrewConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.world_entity import (
    Connection,
    KinematicStructureEntity,
)

# %% specification type parameters
TConnection = TypeVar("TConnection", bound=Connection)


# %% connection specifications


@dataclass
class ConnectionSpecification(
    NamedSpecification, Generic[TConnection], SubClassSafeGeneric, ABC
):
    """
    World- and kinematic-structure-entity-independent description of a connection.

    A connection joins two pre-existing entities, so it is *not* a
    :class:`SpawnSpecification` (which materializes an entity and its own parent connection). It is
    materialized via :meth:`connect`, which takes the ``child`` to attach.

    Each connection family is a concrete subclass that binds the connection type as its generic
    parameter (e.g. ``ConnectionSpecification[FixedConnection]``) and carries exactly the parameters
    that family uses. Materializing a specification forwards those parameters to the connection type's
    :meth:`~semantic_digital_twin.world_description.world_entity.Connection.create_with_dofs`.
    """

    name: str | None = field(default=None, kw_only=True)
    """
    Optional connection name as a plain string.

    If None, ``create_with_dofs`` auto-generates one from parent and child. Wrapped into
    a :class:`PrefixedName` only at materialization time.
    """

    connection_T_child: HomogeneousTransformationMatrix | None = field(
        default=None, kw_only=True
    )
    """
    Constant pose of the child relative to the connection frame, such as a door's centre
    relative to the hinge it swings on.

    The connection moves about its own frame, so a child placed away from it swings or
    slides around that frame rather than around its own origin. Identity if None.
    """

    @property
    def connection_type(self) -> Type[TConnection]:
        """
        The connection type this specification materializes, from its bound generic
        parameter.
        """
        [connection_type] = get_generic_type_parameters(self, ConnectionSpecification)
        return connection_type

    def _create_with_dofs_kwargs(self) -> dict[str, Any]:
        """
        Forward the parameters the connection family declares to ``create_with_dofs``.

        The fields every connection specification shares are applied by :meth:`connect`
        itself, so they are not forwarded.

        :return: The connection parameters, keyed by ``create_with_dofs`` parameter
            name.
        """
        shared_field_names = {
            shared_field.name for shared_field in fields(ConnectionSpecification)
        }
        discovered_attributes = DataclassOnlyIntrospector().discover(type(self))
        instance_values = vars(self)
        result = {}
        for attribute in discovered_attributes:
            public_name = cast(str, attribute.public_name)
            if public_name not in shared_field_names:
                result[public_name] = instance_values[public_name]

        return result

    def parent_T_connection_for_child_at(
        self, parent_T_child: HomogeneousTransformationMatrix
    ) -> HomogeneousTransformationMatrix:
        """
        Compute where the connection frame has to sit so that the child is at
        ``parent_T_child`` while the connection is at its zero position.

        :param parent_T_child: The pose the child should have relative to the parent.
        :return: The placement of the connection frame in the parent frame.
        """
        if self.connection_T_child is None:
            return parent_T_child
        return HomogeneousTransformationMatrix(
            (parent_T_child @ self.connection_T_child.inverse()).evaluate()
        )

    def reconnect(self, world: World, child: KinematicStructureEntity) -> TConnection:
        """
        Replace the connection ``child`` hangs from with one built from this
        specification, under the same parent.

        The child, and with it its whole branch, keeps its current pose: the new
        connection's frame is placed so that the child stays where it is while the
        connection is at its initial position, which its limits or offset may move away
        from zero. The degrees of freedom of the replaced connection are released when
        the outermost world modification block exits.

        :param world: The world the child lives in.
        :param child: The kinematic structure entity whose parent connection is replaced.
        :return: The new connection.
        """
        replaced_connection = child.parent_connection
        parent = replaced_connection.parent
        parent_T_child = world.compute_forward_kinematics(
            parent, child, enable_unsafe_inside_world_block=True
        )
        with world.modify_world():
            world.remove_connection(replaced_connection)
            connection = self.connect(
                world,
                child=child,
                parent=parent,
                parent_T_connection=self.parent_T_connection_for_child_at(
                    parent_T_child
                ),
            )
            placed_connection = connection.copy_with_new_parent(
                parent,
                HomogeneousTransformationMatrix(
                    (
                        parent_T_child
                        @ connection.origin_expression.inverse()
                        @ connection.parent_T_connection_expression
                    ).evaluate()
                ),
            )
            placed_connection.name = connection.name
            world.remove_connection(connection)
            world.add_connection(placed_connection)
        return placed_connection

    def connect(
        self,
        world: World,
        child: KinematicStructureEntity,
        parent: KinematicStructureEntity | None = None,
        parent_T_connection: HomogeneousTransformationMatrix | None = None,
        name: str | None = None,
    ) -> TConnection:
        """
        Materialize the connection between ``parent`` and ``child`` and add it to the
        world.

        A connection joins two pre-existing entities, so the child it connects is
        mandatory. If ``parent`` is omitted, ``world.root`` is used.

        :param world: The world the connection is added to.
        :param child: The kinematic structure entity that becomes the connection's
            child.
        :param parent: The kinematic structure entity that becomes the connection's
            parent. If None, ``world.root`` is used.
        :param parent_T_connection: Placement of the connection in the parent frame.
            Identity if None.
        :param name: Overrides the specification's own name. If None, the spec's name is
            used.
        :return: The materialized connection.
        :raises MissingConnectionParentError: If no parent is given and the world has no
            root.
        """
        parent = parent or world.root
        if parent is None:
            raise MissingConnectionParentError(connection_name=self.name)

        parent_T_connection = (
            parent_T_connection.copy_with_new_reference_frames(
                new_reference_frame=parent, new_child_frame=child
            )
            if parent_T_connection is not None
            else HomogeneousTransformationMatrix(
                reference_frame=parent, child_frame=child
            )
        )

        with world.modify_world():
            connection = self.connection_type.create_with_dofs(
                world=world,
                parent=parent,
                child=child,
                name=self._resolved_name(name),
                parent_T_connection_expression=parent_T_connection,
                connection_T_child_expression=(
                    self.connection_T_child.copy_with_new_reference_frames(
                        new_reference_frame=None, new_child_frame=child
                    )
                    if self.connection_T_child is not None
                    else None
                ),
                **self._create_with_dofs_kwargs(),
            )
            world.add_connection(connection)
        return connection


@dataclass
class FixedConnectionSpecification(ConnectionSpecification[FixedConnection]):
    """
    Declares a rigid
    :class:`~semantic_digital_twin.world_description.connections.FixedConnection`.

    Use this when two entities should keep a constant relative pose and never move with
    respect to each other.
    """


@dataclass
class Connection6DoFSpecification(ConnectionSpecification[Connection6DoF]):
    """
    Declares a free-floating
    :class:`~semantic_digital_twin.world_description.connections.Connection6DoF`.

    Use this when an entity may move and rotate freely relative to its parent, such as
    an object resting in the world that is not rigidly attached to anything.
    """


@dataclass
class ActiveConnection1DOFSpecification(ConnectionSpecification[TConnection], ABC):
    """
    Specification for a single-DoF active connection.

    Concrete leaf subclasses bind the connection type as their generic parameter (e.g.
    prismatic or revolute).
    """

    axis: Vector3 = field(kw_only=True)
    """
    Movement axis of the connection.

    Mandatory: a single-DoF connection without an axis has no meaning, so it cannot be
    constructed.
    """

    multiplier: float = 1.0
    """
    Scaling factor applied to the degree of freedom's motion.
    """

    offset: float = 0.0
    """
    Constant offset applied to the degree of freedom's motion.
    """

    dof_limits: DegreeOfFreedomLimits | None = None
    """
    Limits for the generated degree of freedom.
    """


@dataclass
class PrismaticConnectionSpecification(
    ActiveConnection1DOFSpecification[PrismaticConnection]
):
    """
    Declares a
    :class:`~semantic_digital_twin.world_description.connections.PrismaticConnection`.

    Use this for a single translational degree of freedom along the connection axis,
    such as a drawer sliding in or out.
    """


@dataclass
class RevoluteConnectionSpecification(
    ActiveConnection1DOFSpecification[RevoluteConnection]
):
    """
    Declares a
    :class:`~semantic_digital_twin.world_description.connections.RevoluteConnection`.

    Use this for a single rotational degree of freedom about the connection axis, such
    as a door swinging on its hinge.
    """


@dataclass
class ScrewConnectionSpecification(ActiveConnection1DOFSpecification[ScrewConnection]):
    """
    Declares a
    :class:`~semantic_digital_twin.world_description.connections.ScrewConnection`.

    Use this where rotation about the connection axis and translation along it are
    coupled into a single degree of freedom, such as the thread between a bottle and its
    cap.
    """

    screw_pitch: float = field(kw_only=True)
    """
    The distance between adjacent threads along the connection axis in meters.

    Mandatory: a thread without a pitch couples no translation to its rotation, so it
    cannot be constructed.
    """
