import copy
import inspect
import os
from uuid import uuid4
from pathlib import Path

import numpy as np
import pytest

from krrood.utils import recursive_subclasses
from semantic_digital_twin.specifications.base import SpawnSpecification
from semantic_digital_twin.specifications.connections import (
    ActiveConnection1DOFSpecification,
    ConnectionSpecification,
    FixedConnectionSpecification,
    Connection6DoFSpecification,
    PrismaticConnectionSpecification,
    RevoluteConnectionSpecification,
    ScrewConnectionSpecification,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
    RegionSpecification,
)
from semantic_digital_twin.specifications.semantic_annotations import (
    SemanticAnnotationWithRootSpecification,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from krrood.ormatic.data_access_objects.helper import to_dao
import semantic_digital_twin.orm.ormatic_interface  # registers the DAOs to_dao looks up

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    InvalidPlaneDimensions,
    MergedRobotAnnotationNotFound,
    MissingConnectionAxisError,
    ParsingError,
    PartWholeCardinalityError,
    PartWholeFieldInAnnotationKwargs,
    UnknownPartWholeRelationshipField,
    UselessConceptError,
)
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobotPart
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Milk,
    Handle,
    Door,
    Floor,
    Wall,
    Aperture,
    Drawer,
    Table,
    DoorWithType,
    EntryWay,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from semantic_digital_twin.semantic_annotations.mixins import (
    HasApertures,
    HasRootBody,
    HasRootRegion,
)
from semantic_digital_twin.semantic_annotations.part_whole import (
    IsPartWholeRelationship,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
    ScrewConnection,
    Connection6DoF,
    OmniDrive,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Color, Scale, Box
from semantic_digital_twin.world_description.inertial_properties import Inertial
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body, Region

RESOURCES = Path(__file__).resolve().parents[2] / "semantic_digital_twin" / "resources"


# %% spawning entities, connections and annotations


def _fresh_world() -> World:
    """
    A world holding nothing but its root body.
    """
    return World.create_with_root_body("root")


@pytest.fixture
def empty_world() -> World:
    return _fresh_world()


def test_body_specification_spawns_fixed(empty_world):
    body = BodySpecification.box("box", Scale(1, 1, 1)).spawn(empty_world)
    assert isinstance(body, Body)
    assert body in empty_world.bodies
    assert isinstance(body.parent_connection, FixedConnection)
    assert body.parent_connection.parent is empty_world.root


def test_region_specification_spawns(empty_world):
    region = RegionSpecification.box("region", Scale(1, 1, 1)).spawn(empty_world)
    assert isinstance(region, Region)
    assert isinstance(region.parent_connection, FixedConnection)


def test_connection_specification_defaults_to_none():
    spec = BodySpecification.box("box", Scale(1, 1, 1))
    assert spec.connection_specification is None


def test_region_specification_uses_connection_specification(empty_world):
    # The connection now lives on the entity spec, so regions get non-fixed connections too.
    spec = RegionSpecification.box(
        "sliding_region",
        Scale(1, 1, 1),
        connection_specification=PrismaticConnectionSpecification(axis=Vector3.Z()),
    )
    region = spec.spawn(empty_world)
    assert isinstance(region.parent_connection, PrismaticConnection)


def test_body_and_connection_pose_and_name_override(empty_world):
    spec = BodySpecification.box("box", Scale(1, 1, 1))
    spec.parent_T_self = HomogeneousTransformationMatrix.from_xyz_rpy(x=1, y=2, z=3)
    body = spec.spawn(empty_world, name="renamed")
    assert body.name == PrefixedName("renamed")
    root_T_body = empty_world.compute_forward_kinematics(empty_world.root, body)
    np.testing.assert_allclose(root_T_body.position.to_np()[:3], [1, 2, 3])


def test_body_and_connection_spawn_arg_overrides_stored_pose(empty_world):
    spec = BodySpecification.box("box", Scale(1, 1, 1))
    spec.parent_T_self = HomogeneousTransformationMatrix.from_xyz_rpy(x=1)
    body = spec.spawn(
        empty_world,
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=5),
    )
    root_T_body = empty_world.compute_forward_kinematics(empty_world.root, body)
    np.testing.assert_allclose(root_T_body.position.to_np()[0], 5)


def test_body_and_connection_active(empty_world):
    spec = BodySpecification.box(
        "drawer",
        Scale(1, 1, 1),
        connection_specification=PrismaticConnectionSpecification(axis=Vector3.Z()),
    )
    body = spec.spawn(empty_world)
    assert isinstance(body.parent_connection, PrismaticConnection)


def test_child_specification_recursion(empty_world):
    parent_spec = BodySpecification.box("parent", Scale(1, 1, 1))
    parent_spec.child_specifications.append(
        BodySpecification.box("child", Scale(1, 1, 1))
    )
    parent_body = parent_spec.spawn(empty_world)
    child = empty_world.get_body_by_name("child")
    assert child.parent_connection.parent is parent_body


def test_fixed_annotation_spawns(empty_world):
    spec = Milk.get_annotation_specification(
        "milk",
        BodySpecification.box("milk", Scale(0.1, 0.1, 0.2)),
    )
    annotation = spec.spawn(empty_world)
    assert isinstance(annotation, Milk)
    assert annotation in empty_world.semantic_annotations
    assert isinstance(annotation.root.parent_connection, FixedConnection)


def test_active_annotation_spawns(empty_world):
    spec = Drawer.get_annotation_specification(
        "drawer",
        BodySpecification.box("drawer", Scale(0.4, 0.5, 0.6)),
        parent_connection_specification=PrismaticConnectionSpecification(
            axis=Vector3.X()
        ),
    )
    annotation = spec.spawn(empty_world)
    assert isinstance(annotation, Drawer)
    assert isinstance(annotation.movable_joint, PrismaticConnection)


def test_annotation_root_connection_specification_overrides_type(empty_world):
    # Milk fixes its root connection to fixed by type, but a connection set on the root
    # specification wins.
    spec = Milk.get_annotation_specification(
        "milk",
        BodySpecification.box(
            "milk",
            Scale(0.1, 0.1, 0.2),
            connection_specification=Connection6DoFSpecification(),
        ),
    )
    annotation = spec.spawn(empty_world)
    assert isinstance(annotation.root.parent_connection, Connection6DoF)


def test_active_connection_specification_requires_axis():
    # An axis-less single-DoF connection is meaningless, so it cannot be built at all.
    # This replaces the former spawn-time MissingConnectionAxisError for specifications.
    with pytest.raises(TypeError):
        PrismaticConnectionSpecification()


def test_nested_annotation_on_non_part_whole_field_raises():
    # Milk has no part-whole field, so a nested annotation spec cannot be mounted onto it.
    # Part bindings are validated at construction.
    with pytest.raises(UnknownPartWholeRelationshipField):
        Milk.get_annotation_specification(
            "milk",
            BodySpecification.box("milk", Scale(0.1, 0.1, 0.2)),
            part_specifications={
                "handle": Handle.get_annotation_specification(
                    "handle",
                    Handle.get_default_root_kinematic_structure_entity_specification(
                        scale=Scale(0.1, 0.05, 0.05)
                    ),
                )
            },
        )


# %% world specifications


def test_world_specification_robotless():
    world = WorldSpecification(
        world_parser=None,
        objects=[BodySpecification.box("obj", Scale(1, 1, 1))],
    ).to_domain_object()
    assert not world.is_empty()
    assert world.get_body_by_name("obj") is not None


def test_to_domain_object_is_repeatable():
    spec = WorldSpecification(
        world_parser=None,
        objects=[BodySpecification.box("obj", Scale(1, 1, 1))],
    )
    first = spec.to_domain_object()
    second = spec.to_domain_object()

    assert first is not second
    assert len(first.bodies) == 2
    assert len(second.bodies) == 2


def test_world_specification_from_urdf_environment():
    world = WorldSpecification.from_urdf(
        os.path.join(RESOURCES, "urdf", "table.urdf")
    ).to_domain_object()
    assert not world.is_empty()
    assert world.root is not None


def test_materialized_environments_share_no_entity_ids():
    specification = WorldSpecification.from_urdf(
        os.path.join(RESOURCES, "urdf", "table.urdf")
    )
    first = specification.to_domain_object()
    second = specification.to_domain_object()

    first_ids = {body.id for body in first.bodies}
    second_ids = {body.id for body in second.bodies}
    assert len(first_ids) == len(first.bodies)
    assert first_ids.isdisjoint(second_ids)


def test_world_specification_from_mjcf_environment():
    pytest.importorskip("mujoco")
    world = WorldSpecification.from_mjcf(
        os.path.join(RESOURCES, "mjcf", "table.xml")
    ).to_domain_object()
    assert not world.is_empty()
    assert world.root is not None


def test_world_specification_from_gazebo_environment():
    world = WorldSpecification.from_gazebo(
        os.path.join(RESOURCES, "gazebo", "mini_warehouse", "worlds", "mini.world"),
        objects=[BodySpecification.box("obj", Scale(1, 1, 1))],
    ).to_domain_object()

    assert not world.is_empty()
    assert world.get_body_by_name("obj") is not None


# %% shape constructors


@pytest.mark.parametrize(
    "make_spec",
    [
        lambda: BodySpecification.box("shape", Scale(1, 1, 1)),
        lambda: BodySpecification.sphere("shape", 0.5),
        lambda: BodySpecification.cylinder("shape", 0.4, 1.0),
        lambda: BodySpecification.mesh("shape", str(RESOURCES / "stl" / "milk.stl")),
    ],
)
def test_shape_constructors_spawn(empty_world, make_spec):
    body = make_spec().spawn(empty_world)
    assert isinstance(body, Body)
    assert len(body.collision.shapes) >= 1
    assert isinstance(body.parent_connection, FixedConnection)


def test_from_event_constructor_spawns(empty_world):
    event = Scale(1, 1, 1).to_simple_event().as_composite_set()
    body = BodySpecification.from_event("event_body", event).spawn(empty_world)
    assert isinstance(body, Body)
    assert len(body.collision.shapes) >= 1


@pytest.mark.parametrize(
    "make_spec",
    [
        lambda pose: BodySpecification.box("shape", Scale(1, 1, 1), parent_T_self=pose),
        lambda pose: BodySpecification.sphere("shape", 0.5, parent_T_self=pose),
        lambda pose: BodySpecification.cylinder("shape", 0.4, 1.0, parent_T_self=pose),
        lambda pose: BodySpecification.mesh(
            "shape", str(RESOURCES / "stl" / "milk.stl"), parent_T_self=pose
        ),
        lambda pose: BodySpecification.from_event(
            "shape",
            Scale(1, 1, 1).to_simple_event().as_composite_set(),
            parent_T_self=pose,
        ),
        lambda pose: BodySpecification.from_3d_points(
            "shape",
            [Point3(0, 0, 0), Point3(1, 0, 0), Point3(0, 1, 0), Point3(1, 1, 1)],
            parent_T_self=pose,
        ),
        lambda pose: RegionSpecification.box(
            "shape", Scale(1, 1, 1), parent_T_self=pose
        ),
    ],
)
def test_shape_constructors_apply_parent_T_self(empty_world, make_spec):
    pose = HomogeneousTransformationMatrix.from_xyz_rpy(x=1, y=2, z=3)
    entity = make_spec(pose).spawn(empty_world)
    root_T_entity = empty_world.compute_forward_kinematics(empty_world.root, entity)
    np.testing.assert_allclose(root_T_entity.position.to_np()[:3], [1, 2, 3])


def test_body_specification_from_3d_points_matches_direct_construction():
    """
    ``from_3d_points`` mirrors :meth:`Body.from_3d_points`.
    """
    points = [Point3(0, 0, 0), Point3(1, 0, 0), Point3(0, 1, 0), Point3(1, 1, 1)]
    name = "polytope"

    materialized = BodySpecification.from_3d_points(name, points).to_domain_object(name)
    directly_built = Body.from_3d_points(name=PrefixedName(name), points_3d=points)

    assert (
        len(materialized.collision.shapes) == len(directly_built.collision.shapes) == 1
    )


# %% visual shapes fall back to the collision shapes only when unset


def test_unset_visual_shapes_are_shared_with_the_collision_shapes():
    """
    A ``None`` visual collection makes the body reuse `shapes` for both purposes.
    """
    specification = BodySpecification.box("box", Scale(1, 1, 1))
    assert specification.visual_shapes is None

    body = specification.to_domain_object()

    assert len(body.visual.shapes) == len(body.collision.shapes) == 1


def test_empty_visual_shapes_leave_the_body_without_visual_geometry():
    """
    An empty visual collection means no visual geometry, unlike an unset one.
    """
    specification = BodySpecification.box("box", Scale(1, 1, 1))
    specification.visual_shapes = ShapeCollection(shapes=[])

    body = specification.to_domain_object()

    assert len(body.collision.shapes) == 1
    assert len(body.visual.shapes) == 0


def test_has_root_body_default_specification_without_scale_is_geometryless(empty_world):
    """
    A scale-less body factory yields a bare body, mirrored by an empty body spec.
    """
    spec = HasRootBody.get_default_root_kinematic_structure_entity_specification(
        "bare_body"
    )
    assert isinstance(spec, BodySpecification)

    body = spec.spawn(empty_world)
    assert isinstance(body, Body)
    assert len(body.collision.shapes) == 0


def test_has_root_region_default_specification_without_scale_is_geometryless(
    empty_world,
):
    """
    The base region factory creates a bare region, mirrored by an empty region spec.
    """
    spec = HasRootRegion.get_default_root_kinematic_structure_entity_specification(
        "bare_region"
    )
    assert isinstance(spec, RegionSpecification)

    region = spec.spawn(empty_world)
    assert isinstance(region, Region)
    assert len(region.area.shapes) == 0


def test_constructor_child_specification_param(empty_world):
    parent = BodySpecification.box(
        "parent",
        Scale(1, 1, 1),
        child_specifications=[BodySpecification.box("child", Scale(1, 1, 1))],
    )
    parent_body = parent.spawn(empty_world)
    child = empty_world.get_body_by_name("child")
    assert child.parent_connection.parent is parent_body


def test_body_specification_distinct_visual_and_inertial(empty_world):
    spec = BodySpecification(
        name="box",
        shapes=Box(scale=Scale(1, 1, 1)).as_shape_collection(),
        visual_shapes=ShapeCollection([Box(scale=Scale(2, 2, 2))]),
        inertial=Inertial(mass=2.0),
    )
    body = spec.spawn(empty_world)
    assert body.visual is not body.collision
    assert len(body.visual.shapes) == 1
    assert body.inertial.mass == 2.0


def test_to_domain_object_is_reusable():
    spec = BodySpecification.box("box", Scale(1, 1, 1))
    first = spec.to_domain_object("first")
    second = spec.to_domain_object("second")
    assert first is not second
    assert first.name == PrefixedName("first")
    assert second.name == PrefixedName("second")
    # Shapes are copied, not shared with the spec or between materializations.
    assert first.collision is not second.collision
    assert first.collision is not spec.shapes


def test_spawn_does_not_alias_or_mutate_stored_pose(empty_world):
    """
    A specification is reusable: spawning it must neither bind nor mutate its stored
    pose, and each materialized connection must own a distinct pose bound to its own
    child.
    """
    spec = BodySpecification.box("box", Scale(1, 1, 1))
    first = spec.spawn(empty_world, name="first")

    assert spec.parent_T_self.reference_frame is None
    assert spec.parent_T_self.child_frame is None

    second = spec.spawn(empty_world, name="second")
    first_expression = first.parent_connection.parent_T_connection_expression
    second_expression = second.parent_connection.parent_T_connection_expression

    assert first_expression is not second_expression
    assert first_expression.child_frame is first
    assert second_expression.child_frame is second


def test_to_domain_object_generic_resolution():
    assert isinstance(
        BodySpecification.box("b", Scale(1, 1, 1)).to_domain_object(), Body
    )
    assert isinstance(
        RegionSpecification.box("r", Scale(1, 1, 1)).to_domain_object(), Region
    )


def test_body_and_connection_6dof(empty_world):
    spec = BodySpecification.box(
        "free",
        Scale(1, 1, 1),
        connection_specification=Connection6DoFSpecification(),
    )
    body = spec.spawn(empty_world)
    assert isinstance(body.parent_connection, Connection6DoF)


def test_spawn_positional_name(empty_world):
    body = BodySpecification.box("b", Scale(1, 1, 1)).spawn(empty_world, "renamed")
    assert body.name == PrefixedName("renamed")


# %% robot specifications


def _odom_bodies(world: World) -> list[Body]:
    """
    Every odom body of a world, regardless of the prefix disambiguating it.
    """
    return [body for body in world.bodies if body.name.name == "odom"]


def test_world_specification_with_robot():
    try:
        world = WorldSpecification(
            world_parser=None,
            robots=[
                RobotSpecification(
                    semantic_annotation_type=PR2,
                    world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0),
                    odom_T_robot_start=HomogeneousTransformationMatrix.from_xyz_rpy(
                        y=2.0
                    ),
                )
            ],
        ).to_domain_object()
    except ParsingError as error:
        pytest.skip(f"PR2 URDF not available: {error}")

    odom_body = world.get_body_by_name("odom")
    assert odom_body is not None
    assert isinstance(odom_body.parent_connection, FixedConnection)
    assert odom_body.parent_connection.parent is world.root

    drive = world.get_body_by_name("base_footprint").parent_connection
    assert isinstance(drive, OmniDrive)

    root_T_odom = world.compute_forward_kinematics(world.root, odom_body)
    np.testing.assert_allclose(root_T_odom.position.to_np()[0], 1.0)


def test_world_specification_from_urdf_with_robot():
    try:
        world = WorldSpecification.from_urdf(
            os.path.join(RESOURCES, "urdf", "table.urdf"),
            robots=[RobotSpecification(semantic_annotation_type=PR2)],
        ).to_domain_object()
    except ParsingError as error:
        pytest.skip(f"PR2 URDF not available: {error}")

    odom_body = world.get_body_by_name("odom")
    assert odom_body is not None
    assert odom_body.parent_connection.parent is world.root
    drive = world.get_body_by_name("base_footprint").parent_connection
    assert isinstance(drive, OmniDrive)


def test_robot_specification_returns_spawned_annotation(empty_world):
    try:
        robot = RobotSpecification(semantic_annotation_type=PR2).spawn(empty_world)
    except ParsingError as error:
        pytest.skip(f"PR2 URDF not available: {error}")

    assert isinstance(robot, PR2)
    assert robot in empty_world.get_semantic_annotations_by_type(PR2)


def test_world_specification_with_several_robots():
    try:
        world = WorldSpecification(
            world_parser=None,
            robots=[
                RobotSpecification(
                    semantic_annotation_type=PR2,
                    world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0),
                ),
                RobotSpecification(
                    semantic_annotation_type=PR2,
                    world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(x=-1.0),
                ),
            ],
        ).to_domain_object()
    except ParsingError as error:
        pytest.skip(f"PR2 URDF not available: {error}")

    assert len(world.get_semantic_annotations_by_type(PR2)) == 2

    odom_bodies = _odom_bodies(world)
    assert len(odom_bodies) == 2
    assert odom_bodies[0].name != odom_bodies[1].name
    for odom_body in odom_bodies:
        assert isinstance(odom_body.parent_connection, FixedConnection)
        assert odom_body.parent_connection.parent is world.root

    odom_positions = sorted(
        world.compute_forward_kinematics(world.root, odom_body).position.to_np()[0]
        for odom_body in odom_bodies
    )
    np.testing.assert_allclose(odom_positions, [-1.0, 1.0])


# %% world specifications with starting objects


def test_world_specification_annotation_starting_object():
    world = WorldSpecification(
        world_parser=None,
        objects=[
            Milk.get_annotation_specification(
                "milk",
                BodySpecification.box("milk", Scale(0.1, 0.1, 0.2)),
            )
        ],
    ).to_domain_object()
    milks = world.get_semantic_annotations_by_type(Milk)
    assert len(milks) == 1


# %% connection specification parameters
# ConnectionSpecification captures a connection type and the keyword
# arguments forwarded to its create_with_dofs.


def test_fixed_connection_spec_binds_type_without_params():
    spec = FixedConnectionSpecification()
    assert spec.connection_type is FixedConnection
    assert spec._create_with_dofs_kwargs() == {}


def test_connection_6dof_spec_binds_type_without_params():
    spec = Connection6DoFSpecification()
    assert spec.connection_type is Connection6DoF
    assert spec._create_with_dofs_kwargs() == {}


def test_active_1dof_spec_captures_parameters():
    limits = DegreeOfFreedomLimits.from_position_range_and_speed(maximum_speed=1.0)
    axis = Vector3.Z()
    spec = PrismaticConnectionSpecification(
        axis=axis, multiplier=2.0, offset=0.5, dof_limits=limits
    )
    assert spec.connection_type is PrismaticConnection
    assert spec.axis is axis
    assert spec.multiplier == 2.0
    assert spec.offset == 0.5
    assert spec.dof_limits is limits


def test_active_1dof_spec_defaults():
    axis = Vector3.Z()
    spec = RevoluteConnectionSpecification(axis=axis)
    assert spec.connection_type is RevoluteConnection
    assert spec._create_with_dofs_kwargs() == {
        "axis": axis,
        "multiplier": 1.0,
        "offset": 0.0,
        "dof_limits": None,
    }


def test_fixed_spec_rejects_active_parameters():
    # Each connection family carries exactly its own parameters, so an inapplicable one
    # is a construction error instead of being silently dropped.
    with pytest.raises(TypeError):
        FixedConnectionSpecification(axis=Vector3.Z())


def _connection_specification_types() -> list[type[ConnectionSpecification]]:
    # Specifications that still carry an unbound type parameter have not committed to a
    # connection type yet, so they have no signature to be checked against.
    return [
        specification_type
        for specification_type in recursive_subclasses(ConnectionSpecification)
        if not specification_type.__parameters__
    ]


def _build_connection_specification(
    specification_type: type[ConnectionSpecification],
) -> ConnectionSpecification:
    """
    Build a specification, supplying the parameters its own family requires.
    """
    if issubclass(specification_type, ScrewConnectionSpecification):
        return specification_type(axis=Vector3.Z(), screw_pitch=0.005)
    if issubclass(specification_type, ActiveConnection1DOFSpecification):
        return specification_type(axis=Vector3.Z())
    return specification_type()


@pytest.mark.parametrize("specification_type", _connection_specification_types())
def test_spec_kwargs_match_create_with_dofs_signature(specification_type):
    # The forwarded kwargs must be keyword arguments that create_with_dofs accepts,
    # otherwise the specification cannot materialize its connection.
    spec = _build_connection_specification(specification_type)
    accepted_parameters = inspect.signature(
        spec.connection_type.create_with_dofs
    ).parameters
    assert set(spec._create_with_dofs_kwargs()).issubset(accepted_parameters)


def test_connection_specification_is_not_a_spawn_specification():
    # A connection joins two existing entities; it must not be substitutable for a spawn spec,
    # so it cannot be placed in a child_specification / objects list.
    assert not issubclass(ConnectionSpecification, SpawnSpecification)


def test_connection_spec_connect_fixed(empty_world):
    child = BodySpecification.box("child", Scale(1, 1, 1)).to_domain_object()
    connection = FixedConnectionSpecification().connect(
        empty_world, parent=empty_world.root, child=child
    )
    assert isinstance(connection, FixedConnection)
    assert connection.parent is empty_world.root
    assert connection.child is child
    assert child in empty_world.bodies


def test_connection_spec_connect_defaults_parent_to_root(empty_world):
    child = BodySpecification.box("child", Scale(1, 1, 1)).to_domain_object()
    connection = Connection6DoFSpecification().connect(empty_world, child=child)
    assert isinstance(connection, Connection6DoF)
    assert connection.parent is empty_world.root


def test_connection_spec_connect_active_forwards_kwargs(empty_world):
    limits = DegreeOfFreedomLimits.from_position_range_and_speed(maximum_speed=1.5)
    child = BodySpecification.box("slider", Scale(1, 1, 1)).to_domain_object()
    connection = PrismaticConnectionSpecification(
        axis=Vector3.Z(), dof_limits=limits
    ).connect(empty_world, child=child)
    assert isinstance(connection, PrismaticConnection)
    assert connection.dof.limits.upper.velocity == 1.5
    assert connection.dof.limits.lower.velocity == -1.5


def test_screw_connection_spec_connect_forwards_pitch(empty_world):
    child = BodySpecification.box("cap", Scale(1, 1, 1)).to_domain_object()
    connection = ScrewConnectionSpecification(
        axis=Vector3.Z(), screw_pitch=0.005
    ).connect(empty_world, child=child)
    assert isinstance(connection, ScrewConnection)
    assert connection.screw_pitch == 0.005


def test_connection_spec_connect_applies_pose(empty_world):
    child = BodySpecification.box("child", Scale(1, 1, 1)).to_domain_object()
    FixedConnectionSpecification().connect(
        empty_world,
        parent_T_connection=HomogeneousTransformationMatrix.from_xyz_rpy(x=1, y=2, z=3),
        child=child,
    )
    root_T_child = empty_world.compute_forward_kinematics(empty_world.root, child)
    np.testing.assert_allclose(root_T_child.position.to_np()[:3], [1, 2, 3])


def test_connection_spec_connect_without_pose_places_the_child_at_the_parent(
    empty_world,
):
    """
    An omitted ``parent_T_connection`` places the connection at the parent's origin.
    """
    child = BodySpecification.box("child", Scale(1, 1, 1)).to_domain_object()
    FixedConnectionSpecification().connect(
        empty_world, parent_T_connection=None, child=child
    )
    root_T_child = empty_world.compute_forward_kinematics(empty_world.root, child)
    np.testing.assert_allclose(root_T_child.position.to_np()[:3], [0, 0, 0])


def test_connection_spec_connect_requires_child(empty_world):
    # The child a connection joins is mandatory in the signature, so omitting it is
    # rejected at call time rather than by a runtime guard inside connect.
    with pytest.raises(TypeError):
        FixedConnectionSpecification().connect(empty_world)


def test_connection_spec_connect_without_name_matches_direct_creation(empty_world):
    # A nameless spec must auto-generate the same connection name as creating the
    # connection directly between an identically-named parent and child.
    spec_child = BodySpecification.box("child", Scale(1, 1, 1)).to_domain_object()
    spec_connection = FixedConnectionSpecification().connect(
        empty_world, parent=empty_world.root, child=spec_child
    )

    direct_world = _fresh_world()
    direct_child = BodySpecification.box("child", Scale(1, 1, 1)).to_domain_object()
    with direct_world.modify_world():
        direct_connection = FixedConnection.create_with_dofs(
            world=direct_world, parent=direct_world.root, child=direct_child
        )
        direct_world.add_connection(direct_connection)

    assert spec_connection.name == direct_connection.name


# %% the child's constant offset from its connection frame
# connection_T_child lets a child sit away from the frame its connection moves about,
# like a door whose origin is its centre hanging on a hinge at its edge.


def _hinge_T_door() -> HomogeneousTransformationMatrix:
    """
    The pose of a door's centre relative to the hinge it swings on.
    """
    return HomogeneousTransformationMatrix.from_xyz_rpy(y=0.5)


def _door_on_hinge_specification() -> BodySpecification:
    return BodySpecification.box(
        "door",
        Scale(0.03, 1, 2),
        connection_specification=RevoluteConnectionSpecification(
            axis=Vector3.Z(), connection_T_child=_hinge_T_door()
        ),
    )


def test_connect_places_the_child_at_its_offset_from_the_connection(empty_world):
    child = BodySpecification.box("child", Scale(1, 1, 1)).to_domain_object()
    parent_T_connection = HomogeneousTransformationMatrix.from_xyz_rpy(x=1)
    connection = FixedConnectionSpecification(
        connection_T_child=_hinge_T_door()
    ).connect(empty_world, child=child, parent_T_connection=parent_T_connection)

    np.testing.assert_allclose(
        connection.connection_T_child_expression.to_np(), _hinge_T_door().to_np()
    )
    np.testing.assert_allclose(
        empty_world.compute_forward_kinematics(empty_world.root, child).to_np(),
        (parent_T_connection @ _hinge_T_door()).to_np(),
    )


def test_spawn_places_the_child_at_parent_T_self_when_its_joint_is_at_zero(
    empty_world,
):
    parent_T_door = HomogeneousTransformationMatrix.from_xyz_rpy(x=2, yaw=0.3)
    door = _door_on_hinge_specification().spawn(
        empty_world, parent_T_self=parent_T_door
    )

    np.testing.assert_allclose(
        empty_world.compute_forward_kinematics(empty_world.root, door).to_np(),
        parent_T_door.to_np(),
    )


def test_child_swings_about_its_connection_frame_not_its_own_origin(empty_world):
    door = _door_on_hinge_specification().spawn(empty_world)
    angle = np.pi / 2

    door.parent_connection.position = angle
    empty_world.notify_state_change()

    expected_root_T_door = (
        _hinge_T_door().inverse()
        @ HomogeneousTransformationMatrix.from_xyz_rpy(yaw=angle)
        @ _hinge_T_door()
    )
    np.testing.assert_allclose(
        empty_world.compute_forward_kinematics(empty_world.root, door).to_np(),
        expected_root_T_door.to_np(),
        atol=1e-12,
    )


def test_connection_T_child_is_not_bound_to_the_children_it_connected(empty_world):
    specification = FixedConnectionSpecification(connection_T_child=_hinge_T_door())
    first_child = BodySpecification.box("first", Scale(1, 1, 1)).to_domain_object()
    second_child = BodySpecification.box("second", Scale(1, 1, 1)).to_domain_object()

    first = specification.connect(empty_world, child=first_child)
    second = specification.connect(empty_world, child=second_child)

    assert first.connection_T_child_expression.child_frame is first_child
    assert second.connection_T_child_expression.child_frame is second_child
    assert specification.connection_T_child.child_frame is None


def test_reconnect_keeps_the_child_and_its_branch_where_they_are(empty_world):
    door_specification = BodySpecification.box("door", Scale(0.03, 1, 2))
    door_specification.child_specifications.append(
        BodySpecification.box("handle", Scale(0.05, 0.1, 0.02))
    )
    door = door_specification.spawn(
        empty_world,
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=1, yaw=0.3),
    )
    handle = empty_world.get_body_by_name("handle")
    root_T_door = empty_world.compute_forward_kinematics(empty_world.root, door)

    connection = RevoluteConnectionSpecification(
        axis=Vector3.Z(), connection_T_child=_hinge_T_door()
    ).reconnect(empty_world, door)

    assert isinstance(door.parent_connection, RevoluteConnection)
    assert door.parent_connection is connection
    assert connection.parent is empty_world.root
    assert handle.parent_kinematic_structure_entity is door
    np.testing.assert_allclose(
        empty_world.compute_forward_kinematics(empty_world.root, door).to_np(),
        root_T_door.to_np(),
        atol=1e-12,
    )


def test_reconnect_swings_the_child_about_the_new_connection_frame(empty_world):
    door = BodySpecification.box("door", Scale(0.03, 1, 2)).spawn(empty_world)
    connection = RevoluteConnectionSpecification(
        axis=Vector3.Z(), connection_T_child=_hinge_T_door()
    ).reconnect(empty_world, door)
    angle = np.pi / 2

    connection.position = angle
    empty_world.notify_state_change()

    expected_root_T_door = (
        _hinge_T_door().inverse()
        @ HomogeneousTransformationMatrix.from_xyz_rpy(yaw=angle)
        @ _hinge_T_door()
    )
    np.testing.assert_allclose(
        empty_world.compute_forward_kinematics(empty_world.root, door).to_np(),
        expected_root_T_door.to_np(),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "joint_specification",
    [
        RevoluteConnectionSpecification(
            axis=Vector3.Z(), offset=0.3, connection_T_child=_hinge_T_door()
        ),
        RevoluteConnectionSpecification(
            axis=Vector3.Z(),
            dof_limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                lower_position=0.2, upper_position=1.0
            ),
            connection_T_child=_hinge_T_door(),
        ),
    ],
    ids=["offset", "limits_excluding_zero"],
)
def test_reconnect_keeps_the_child_where_it_is_when_the_joint_starts_away_from_zero(
    empty_world, joint_specification
):
    door = BodySpecification.box("door", Scale(0.03, 1, 2)).spawn(
        empty_world,
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=1, yaw=0.3),
    )
    root_T_door = empty_world.compute_forward_kinematics(empty_world.root, door)

    connection = joint_specification.reconnect(empty_world, door)

    assert connection.position != 0
    np.testing.assert_allclose(
        empty_world.compute_forward_kinematics(empty_world.root, door).to_np(),
        root_T_door.to_np(),
        atol=1e-12,
    )


def test_reconnect_keeps_the_name_of_the_specification(empty_world):
    door = BodySpecification.box("door", Scale(0.03, 1, 2)).spawn(empty_world)
    specification = RevoluteConnectionSpecification(
        name="front_door_hinge", axis=Vector3.Z(), connection_T_child=_hinge_T_door()
    )

    connection = specification.reconnect(empty_world, door)

    assert empty_world.get_connection_by_name(specification.name) is connection


def test_reconnect_releases_the_degrees_of_freedom_of_the_replaced_connection(
    empty_world,
):
    drawer = BodySpecification.box(
        "drawer",
        Scale(0.5, 0.5, 0.2),
        connection_specification=PrismaticConnectionSpecification(axis=Vector3.X()),
    ).spawn(empty_world)
    [replaced_degree_of_freedom] = drawer.parent_connection.dofs

    connection = RevoluteConnectionSpecification(axis=Vector3.Z()).reconnect(
        empty_world, drawer
    )

    assert set(empty_world.degrees_of_freedom) == set(connection.dofs)
    assert replaced_degree_of_freedom not in empty_world.degrees_of_freedom


def test_connection_T_child_survives_orm_round_trip(empty_world):
    specification = Door.get_annotation_specification(
        "door",
        Door.get_default_root_kinematic_structure_entity_specification(
            scale=Scale(0.03, 1, 2)
        ),
        parent_connection_specification=RevoluteConnectionSpecification(
            axis=Vector3.Z(), connection_T_child=_hinge_T_door()
        ),
    )

    door = to_dao(specification).from_dao().spawn(empty_world)

    np.testing.assert_allclose(
        door.root.parent_connection.connection_T_child_expression.to_np(),
        _hinge_T_door().to_np(),
    )


# %% default geometry specifications match the factories
# get_default_root_kinematic_structure_entity_specification reproduces
# the geometry that create_with_new_body_in_world(scale=...)
# (and Aperture's region factory) generate, for every class that
# implements its own geometry-generating factory override.


def _assert_same_geometry(
    spec_collection: ShapeCollection, factory_collection: ShapeCollection
):
    np.testing.assert_allclose(
        spec_collection.combined_mesh.bounds,
        factory_collection.combined_mesh.bounds,
    )
    assert len(spec_collection) == len(factory_collection)


def test_default_spec_matches_base_body(empty_world):
    scale = Scale(0.2, 0.3, 0.4)
    with empty_world.modify_world():
        factory = Milk.create_with_new_body_in_world(
            name="milk", world=empty_world, scale=scale
        )
    spec_body = Milk.get_default_root_kinematic_structure_entity_specification(
        "milk", scale
    ).to_domain_object()
    _assert_same_geometry(spec_body.collision, factory.root.collision)
    assert spec_body.collision is spec_body.visual


def test_default_spec_matches_case_body(empty_world):
    scale = Scale(0.3, 0.4, 0.5)
    with empty_world.modify_world():
        factory = Drawer.create_with_new_body_in_world(
            name="drawer", world=empty_world, scale=scale
        )
    spec_body = Drawer.get_default_root_kinematic_structure_entity_specification(
        "drawer", scale
    ).to_domain_object()
    _assert_same_geometry(spec_body.collision, factory.root.collision)
    assert spec_body.collision is spec_body.visual


def test_default_spec_matches_case_body_with_wall_thickness(empty_world):
    scale = Scale(0.4, 0.5, 0.6)
    with empty_world.modify_world():
        factory = Drawer.get_annotation_specification(
            "drawer",
            Drawer.get_default_root_kinematic_structure_entity_specification(
                scale=scale, wall_thickness=0.05
            ),
        ).spawn(empty_world)
    spec_body = Drawer.get_default_root_kinematic_structure_entity_specification(
        "drawer", scale, wall_thickness=0.05
    ).to_domain_object()
    _assert_same_geometry(spec_body.collision, factory.root.collision)


def test_default_spec_matches_handle(empty_world):
    scale = Scale(0.1, 0.05, 0.05)
    with empty_world.modify_world():
        factory = Handle.get_annotation_specification(
            "handle",
            Handle.get_default_root_kinematic_structure_entity_specification(
                scale=scale, thickness=0.01
            ),
        ).spawn(empty_world)
    spec_body = Handle.get_default_root_kinematic_structure_entity_specification(
        "handle", scale, thickness=0.01
    ).to_domain_object()
    _assert_same_geometry(spec_body.collision, factory.root.collision)


def test_default_spec_matches_handle_without_explicit_scale(empty_world):
    # Both sides must fall back to the same default scale, otherwise the factory and the
    # specification silently produce differently shaped handles.
    with empty_world.modify_world():
        factory = Handle.create_with_new_body_in_world(name="handle", world=empty_world)
    spec_body = Handle.get_default_root_kinematic_structure_entity_specification(
        "handle"
    ).to_domain_object()
    _assert_same_geometry(spec_body.collision, factory.root.collision)


def test_default_spec_matches_door(empty_world):
    scale = Scale(0.03, 1, 2)
    with empty_world.modify_world():
        factory = Door.create_with_new_body_in_world(
            name="door", world=empty_world, scale=scale
        )
    spec_body = Door.get_default_root_kinematic_structure_entity_specification(
        "door", scale
    ).to_domain_object()
    _assert_same_geometry(spec_body.collision, factory.root.collision)


def test_default_spec_door_validates_plane():
    with pytest.raises(InvalidPlaneDimensions):
        Door.get_default_root_kinematic_structure_entity_specification(
            "door", Scale(2, 1, 1)
        )


def test_default_spec_matches_floor(empty_world):
    scale = Scale(2, 2, 0.1)
    with empty_world.modify_world():
        factory = Floor.create_with_new_body_in_world(
            name="floor", world=empty_world, scale=scale
        )
    spec_body = Floor.get_default_root_kinematic_structure_entity_specification(
        "floor", scale
    ).to_domain_object()
    _assert_same_geometry(spec_body.collision, factory.root.collision)


def test_default_spec_matches_wall(empty_world):
    scale = Scale(0.1, 4, 2)
    with empty_world.modify_world():
        factory = Wall.create_with_new_body_in_world(
            name="wall", world=empty_world, scale=scale
        )
    spec_body = Wall.get_default_root_kinematic_structure_entity_specification(
        "wall", scale
    ).to_domain_object()
    _assert_same_geometry(spec_body.collision, factory.root.collision)


def test_default_spec_matches_aperture_region(empty_world):
    scale = Scale(0.1, 1, 2)
    with empty_world.modify_world():
        factory = Aperture.create_with_new_region_in_world(
            name="aperture", world=empty_world, scale=scale
        )
    spec_region = Aperture.get_default_root_kinematic_structure_entity_specification(
        "aperture", scale
    ).to_domain_object()
    _assert_same_geometry(spec_region.area, factory.root.area)


def test_default_spec_robot_part_raises():
    # AbstractRobotPart's geometry must come from URDF parsing, not from scale,
    # so it raises just like its create_with_new_body_in_world override.
    with pytest.raises(UselessConceptError):
        AbstractRobotPart.get_default_root_kinematic_structure_entity_specification(
            "part", Scale(1, 1, 1)
        )


# %% annotation specifications
# get_annotation_specification wraps a geometry spec into a
# SemanticAnnotationWithRootSpecification that spawns an annotation
# equivalent to create_with_new_body_in_world.


def test_annotation_spec_base_body(empty_world):
    scale = Scale(0.2, 0.3, 0.4)
    spec = Milk.get_annotation_specification(
        "milk",
        Milk.get_default_root_kinematic_structure_entity_specification(scale=scale),
    )
    assert isinstance(spec, SemanticAnnotationWithRootSpecification)
    assert spec.semantic_annotation_type is Milk
    assert isinstance(spec.root_specification, BodySpecification)

    annotation = spec.spawn(empty_world)
    assert isinstance(annotation, Milk)
    assert annotation in empty_world.semantic_annotations
    assert isinstance(annotation.root.parent_connection, FixedConnection)

    factory_world = _fresh_world()
    with factory_world.modify_world():
        factory = Milk.create_with_new_body_in_world(
            name="milk_factory", world=factory_world, scale=scale
        )
    _assert_same_geometry(annotation.root.collision, factory.root.collision)


def test_annotation_spec_active_door(empty_world):
    spec = Door.get_annotation_specification(
        "door",
        Door.get_default_root_kinematic_structure_entity_specification(
            scale=Scale(0.03, 1, 2)
        ),
        parent_connection_specification=RevoluteConnectionSpecification(
            axis=Vector3.Z()
        ),
    )
    annotation = spec.spawn(empty_world)
    assert isinstance(annotation, Door)
    assert isinstance(annotation.movable_joint, RevoluteConnection)


def test_annotation_spec_aperture_region(empty_world):
    scale = Scale(0.1, 1, 2)
    spec = Aperture.get_annotation_specification(
        "aperture",
        Aperture.get_default_root_kinematic_structure_entity_specification(scale=scale),
    )
    assert isinstance(spec.root_specification, RegionSpecification)

    annotation = spec.spawn(empty_world)
    assert isinstance(annotation, Aperture)
    factory_world = _fresh_world()
    with factory_world.modify_world():
        factory = Aperture.create_with_new_region_in_world(
            name="aperture_factory", world=factory_world, scale=scale
        )
    _assert_same_geometry(annotation.root.area, factory.root.area)


def test_annotation_spec_robot_part_raises():
    with pytest.raises(UselessConceptError):
        AbstractRobotPart.get_annotation_specification(
            "part",
            AbstractRobotPart.get_default_root_kinematic_structure_entity_specification(
                scale=Scale(1, 1, 1)
            ),
        )


# %% nested annotation parts
# Nested annotations: part_specifications entries are spawned and
# mounted via the part-whole `add`, keyed by the target field name.


def _spawn_with_parts(world, whole_type, whole_scale, parts):
    """
    Spawn ``whole_type`` from its default annotation spec, with ``parts`` as nested
    annotations.
    """
    return whole_type.get_annotation_specification(
        "whole",
        whole_type.get_default_root_kinematic_structure_entity_specification(
            scale=whole_scale
        ),
        part_specifications=parts,
    ).spawn(world)


def test_nested_handle_attaches_as_child(empty_world):
    handle_part = Handle.get_annotation_specification(
        "handle",
        Handle.get_default_root_kinematic_structure_entity_specification(
            scale=Scale(0.1, 0.05, 0.05)
        ),
    )
    drawer = _spawn_with_parts(
        empty_world, Drawer, Scale(0.4, 0.5, 0.6), {"handle": handle_part}
    )
    assert isinstance(drawer.handle, Handle)
    assert drawer.handle.root.parent_connection.parent is drawer.root
    assert isinstance(drawer.handle.root.parent_connection, FixedConnection)


def test_door_joint_survives_mounting_the_door_as_a_part(empty_world):
    """
    A door swings on its hinge, and mounting the door onto a cabinet must keep that
    hinge, including where the door sits relative to it: a door attached rigidly to the
    cabinet can no longer be opened.
    """
    door_part = Door.get_annotation_specification(
        "door",
        Door.get_default_root_kinematic_structure_entity_specification(
            scale=Scale(0.03, 1, 2)
        ),
        parent_connection_specification=RevoluteConnectionSpecification(
            axis=Vector3.Z(), connection_T_child=_hinge_T_door()
        ),
    )

    cabinet = _spawn_with_parts(
        empty_world, Cabinet, Scale(0.5, 1, 2), {"doors": door_part}
    )

    [door] = cabinet.doors
    assert isinstance(door.movable_joint, RevoluteConnection)
    assert door.movable_joint.parent is cabinet.root
    np.testing.assert_allclose(
        door.movable_joint.connection_T_child_expression.to_np(),
        _hinge_T_door().to_np(),
    )


def test_drawer_joint_survives_mounting_the_drawer_as_a_part(empty_world):
    """
    A drawer slides on its rails, and mounting the drawer into a cabinet must keep
    them: a drawer attached rigidly to the cabinet can no longer be pulled out.
    """
    drawer_part = Drawer.get_annotation_specification(
        "drawer",
        Drawer.get_default_root_kinematic_structure_entity_specification(
            scale=Scale(0.4, 0.5, 0.6)
        ),
        parent_connection_specification=PrismaticConnectionSpecification(
            axis=Vector3.X()
        ),
    )

    cabinet = _spawn_with_parts(
        empty_world, Cabinet, Scale(0.5, 1, 2), {"drawers": drawer_part}
    )

    [drawer] = cabinet.drawers
    assert isinstance(drawer.movable_joint, PrismaticConnection)
    assert drawer.movable_joint.parent is cabinet.root


def test_nested_aperture_cuts_geometry(empty_world):
    plain_wall = Wall.get_annotation_specification(
        "plain_wall",
        Wall.get_default_root_kinematic_structure_entity_specification(
            scale=Scale(0.1, 2, 2)
        ),
    ).spawn(empty_world)
    plain_shape_count = len(plain_wall.root.collision.shapes)

    aperture_part = Aperture.get_annotation_specification(
        "hole",
        Aperture.get_default_root_kinematic_structure_entity_specification(
            scale=Scale(0.1, 0.5, 0.5)
        ),
    )
    wall = _spawn_with_parts(
        empty_world, Wall, Scale(0.1, 2, 2), {"apertures": aperture_part}
    )
    assert len(wall.apertures) == 1
    assert isinstance(wall.apertures[0], Aperture)
    # cutting the aperture out of the wall changes its collision geometry
    assert len(wall.root.collision.shapes) != plain_shape_count


def test_nested_list_valued_parts_on_to_many_field(empty_world):
    aperture_a = Aperture.get_annotation_specification(
        "hole_a",
        Aperture.get_default_root_kinematic_structure_entity_specification(
            scale=Scale(0.1, 0.5, 0.5)
        ),
    )
    aperture_a.root_specification.parent_T_self = (
        HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.8)
    )
    aperture_b = Aperture.get_annotation_specification(
        "hole_b",
        Aperture.get_default_root_kinematic_structure_entity_specification(
            scale=Scale(0.1, 0.5, 0.5)
        ),
    )
    aperture_b.root_specification.parent_T_self = (
        HomogeneousTransformationMatrix.from_xyz_rpy(y=0.8)
    )
    wall = Wall.get_annotation_specification(
        "wall",
        Wall.get_default_root_kinematic_structure_entity_specification(
            scale=Scale(0.1, 3, 3)
        ),
        part_specifications={"apertures": [aperture_a, aperture_b]},
    ).spawn(empty_world)
    assert len(wall.apertures) == 2
    assert all(isinstance(aperture, Aperture) for aperture in wall.apertures)


def test_list_value_on_singular_part_field_raises():
    # part_specifications are validated at construction.
    with pytest.raises(PartWholeCardinalityError):
        Drawer.get_annotation_specification(
            "drawer",
            Drawer.get_default_root_kinematic_structure_entity_specification(
                scale=Scale(0.4, 0.5, 0.6)
            ),
            part_specifications={
                "handle": [
                    Handle.get_annotation_specification(
                        "h1",
                        Handle.get_default_root_kinematic_structure_entity_specification(
                            scale=Scale(0.1, 0.05, 0.05)
                        ),
                    ),
                    Handle.get_annotation_specification(
                        "h2",
                        Handle.get_default_root_kinematic_structure_entity_specification(
                            scale=Scale(0.1, 0.05, 0.05)
                        ),
                    ),
                ]
            },
        )


def test_nested_part_placement_is_relative_to_whole(empty_world):
    handle_part = Handle.get_annotation_specification(
        "handle",
        Handle.get_default_root_kinematic_structure_entity_specification(
            scale=Scale(0.1, 0.05, 0.05)
        ),
    )
    handle_part.root_specification.parent_T_self = (
        HomogeneousTransformationMatrix.from_xyz_rpy(y=0.5)
    )
    drawer = _spawn_with_parts(
        empty_world, Drawer, Scale(0.4, 0.5, 0.6), {"handle": handle_part}
    )
    drawer_T_handle = empty_world.compute_forward_kinematics(
        drawer.root, drawer.handle.root
    )
    np.testing.assert_allclose(
        drawer_T_handle.position.to_np()[:3], [0, 0.5, 0], atol=1e-9
    )


def test_annotation_connection_limits_threaded(empty_world):
    limits = DegreeOfFreedomLimits.from_position_range_and_speed(maximum_speed=1.5)
    spec = Drawer.get_annotation_specification(
        "drawer",
        Drawer.get_default_root_kinematic_structure_entity_specification(
            scale=Scale(0.4, 0.5, 0.6)
        ),
        parent_connection_specification=PrismaticConnectionSpecification(
            axis=Vector3.Z(), dof_limits=limits
        ),
    )
    drawer = spec.spawn(empty_world)
    dof_limits = drawer.movable_joint.dof.limits
    assert dof_limits.upper.velocity == 1.5
    assert dof_limits.lower.velocity == -1.5


def test_inert_annotation_kwargs_reach_constructor(empty_world):
    # supporting_surface is a plain (non-part-whole) constructor field, so it is allowed in
    # annotation_kwargs and reaches the constructor unchanged.
    surface = RegionSpecification.box("surface", Scale(1, 1, 0.01)).spawn(empty_world)
    table = Table.get_annotation_specification(
        "table",
        Table.get_default_root_kinematic_structure_entity_specification(
            scale=Scale(1, 1, 0.5)
        ),
        annotation_kwargs={"supporting_surface": surface},
    ).spawn(empty_world)
    assert table.supporting_surface is surface


def test_part_whole_field_in_annotation_kwargs_raises():
    # A part-whole relationship field (Drawer.handle) must not be passed via annotation_kwargs;
    # it belongs in part_specifications. This is rejected at spec construction.
    with pytest.raises(PartWholeFieldInAnnotationKwargs):
        Drawer.get_annotation_specification(
            "drawer",
            Drawer.get_default_root_kinematic_structure_entity_specification(
                scale=Scale(0.4, 0.5, 0.6)
            ),
            annotation_kwargs={
                "handle": Handle.get_annotation_specification(
                    "handle",
                    Handle.get_default_root_kinematic_structure_entity_specification(
                        scale=Scale(0.1, 0.05, 0.05)
                    ),
                )
            },
        )


def test_non_part_whole_field_in_part_specifications_raises():
    # supporting_surface is not a part-whole relationship, so it cannot hold a nested part spec.
    # part_specifications are validated at construction.
    with pytest.raises(UnknownPartWholeRelationshipField):
        Table.get_annotation_specification(
            "table",
            Table.get_default_root_kinematic_structure_entity_specification(
                scale=Scale(1, 1, 0.5)
            ),
            part_specifications={
                "supporting_surface": RegionSpecification.box(
                    "surface", Scale(1, 1, 0.01)
                )
            },
        )


def test_storage_objects_in_part_specifications_raises():
    # IsStorageSpace.objects is not a part-whole relationship, so spec-based occupants are unsupported.
    with pytest.raises(UnknownPartWholeRelationshipField):
        Table.get_annotation_specification(
            "table",
            Table.get_default_root_kinematic_structure_entity_specification(
                scale=Scale(1, 1, 0.5)
            ),
            part_specifications={
                "objects": [
                    Milk.get_annotation_specification(
                        "milk",
                        Milk.get_default_root_kinematic_structure_entity_specification(
                            scale=Scale(0.1, 0.1, 0.2)
                        ),
                    )
                ]
            },
        )


def test_complex_spawned_world_is_deepcopyable(empty_world):
    Drawer.get_annotation_specification(
        "drawer",
        Drawer.get_default_root_kinematic_structure_entity_specification(
            scale=Scale(0.4, 0.5, 0.6)
        ),
        parent_connection_specification=RevoluteConnectionSpecification(
            axis=Vector3.Z(), connection_T_child=_hinge_T_door()
        ),
        part_specifications={
            "handle": Handle.get_annotation_specification(
                "handle",
                Handle.get_default_root_kinematic_structure_entity_specification(
                    scale=Scale(0.1, 0.05, 0.05)
                ),
            ),
        },
    ).spawn(empty_world)
    Wall.get_annotation_specification(
        "wall",
        Wall.get_default_root_kinematic_structure_entity_specification(
            scale=Scale(0.1, 2, 2)
        ),
        part_specifications={
            "apertures": Aperture.get_annotation_specification(
                "hole",
                Aperture.get_default_root_kinematic_structure_entity_specification(
                    scale=Scale(0.1, 0.5, 0.5)
                ),
            )
        },
    ).spawn(empty_world)
    Milk.get_annotation_specification(
        "milk",
        Milk.get_default_root_kinematic_structure_entity_specification(
            scale=Scale(0.1, 0.1, 0.2)
        ),
    ).spawn(empty_world)

    world_copy = copy.deepcopy(empty_world)

    assert world_copy is not empty_world
    assert len(world_copy.kinematic_structure_entities) == len(
        empty_world.kinematic_structure_entities
    )
    assert len(world_copy.connections) == len(empty_world.connections)
    assert len(list(world_copy.semantic_annotations)) == len(
        list(empty_world.semantic_annotations)
    )


def test_nested_composite_matches_manual_construction(empty_world):
    scale = Scale(0.4, 0.5, 0.6)
    handle_scale = Scale(0.1, 0.05, 0.05)

    drawer = Drawer.get_annotation_specification(
        "drawer",
        Drawer.get_default_root_kinematic_structure_entity_specification(scale=scale),
        parent_connection_specification=PrismaticConnectionSpecification(
            axis=Vector3.X()
        ),
        part_specifications={
            "handle": Handle.get_annotation_specification(
                "handle",
                Handle.get_default_root_kinematic_structure_entity_specification(
                    scale=handle_scale
                ),
            ),
        },
    ).spawn(empty_world)

    assert isinstance(drawer.handle, Handle)
    assert isinstance(drawer.movable_joint, PrismaticConnection)
    assert drawer.handle.root.parent_connection.parent is drawer.root

    manual_world = _fresh_world()
    with manual_world.modify_world():
        manual_drawer = Drawer.create_with_new_body_in_world(
            name="drawer_manual",
            world=manual_world,
            scale=scale,
            parent_connection_specification=PrismaticConnectionSpecification(
                axis=Vector3.X()
            ),
        )
        manual_handle = Handle.create_with_new_body_in_world(
            name="handle_manual",
            world=manual_world,
            scale=handle_scale,
        )
        manual_drawer.add(manual_handle)

    _assert_same_geometry(drawer.root.collision, manual_drawer.root.collision)
    _assert_same_geometry(drawer.handle.root.collision, manual_handle.root.collision)


# %% factory defaults are not shared mutable state
# Factory overrides must not carry shared mutable spatial defaults:
# a Vector3 default is constructed once at definition time and would
# be aliased by every caller that mutates it.


def _factory_overrides():
    """
    Yield every ``create_with_new_*_in_world`` factory declared on an annotation class,
    as ``(owner, method_name, signature)``.
    """
    roots = [HasRootBody, HasRootRegion]
    annotation_types = {
        subclass for root in roots for subclass in [root, *recursive_subclasses(root)]
    }
    for annotation_type in annotation_types:
        for method_name, method in vars(annotation_type).items():
            if not method_name.startswith("create_with_new_"):
                continue
            yield annotation_type, method_name, inspect.signature(
                method.__func__ if isinstance(method, classmethod) else method
            )


def test_factories_have_no_shared_mutable_spatial_defaults():
    offenders = [
        f"{owner.__name__}.{method_name}({parameter.name})"
        for owner, method_name, signature in _factory_overrides()
        for parameter in signature.parameters.values()
        if isinstance(parameter.default, (Vector3, Point3, Scale))
    ]
    assert not offenders, f"shared mutable defaults: {offenders}"


# %% default geometry builders take a connection
# The per-class default geometry builders take a connection just like
# BodySpecification's own builders do.


@pytest.mark.parametrize(
    "annotation_type, builder_name",
    [
        (Milk, "get_default_root_kinematic_structure_entity_specification"),
        (Drawer, "get_default_root_kinematic_structure_entity_specification"),
        (Handle, "get_default_root_kinematic_structure_entity_specification"),
        (Aperture, "get_default_root_kinematic_structure_entity_specification"),
    ],
)
def test_default_geometry_builder_takes_a_connection(annotation_type, builder_name):
    # Without this, custom geometry would force the connection to be assigned afterwards.
    specification = getattr(annotation_type, builder_name)(
        "entity",
        Scale(0.1, 0.1, 0.1),
        connection_specification=Connection6DoFSpecification(),
    )
    assert isinstance(
        specification.connection_specification, Connection6DoFSpecification
    )


def test_custom_geometry_and_connection_build_in_one_expression(empty_world):
    annotation = Handle.get_annotation_specification(
        "handle",
        Handle.get_default_root_kinematic_structure_entity_specification(
            scale=Scale(0.1, 0.05, 0.05),
            connection_specification=Connection6DoFSpecification(),
            thickness=0.01,
        ),
    ).spawn(empty_world)
    assert isinstance(annotation.root.parent_connection, Connection6DoF)


# %% door entry way as a nested part
# A door's entry way is declared on the door's default specification and mounted
# through the part-whole `add`. It is coextensive with the door leaf, so mounting it
# must not cut the door; mounting the same instance onto a wall must still cut.


def test_entry_way_matches_custom_door_geometry(empty_world):
    # The entry way is coextensive with the door leaf, so a door built from custom
    # geometry must get an entry way sized from that geometry rather than from the
    # type's default scale.
    with empty_world.modify_world():
        door = Door.get_annotation_specification(
            "door",
            Door.get_default_root_kinematic_structure_entity_specification(
                scale=Scale(0.05, 2.0, 2.5)
            ),
        ).spawn(empty_world)
    door_scale = door.root.collision.scale
    entry_way_scale = door.entry_way.root.area.scale
    assert (entry_way_scale.x, entry_way_scale.y, entry_way_scale.z) == pytest.approx(
        (door_scale.x, door_scale.y, door_scale.z)
    )


def test_door_keeps_geometry_when_entry_way_is_mounted(empty_world):
    scale = Scale(0.03, 1, 2)
    with empty_world.modify_world():
        door = Door.create_with_new_body_in_world(
            name="door", world=empty_world, scale=scale
        )
    assert len(door.root.collision.shapes) > 0
    expected = Door.get_default_root_kinematic_structure_entity_specification(
        "door", scale
    ).to_domain_object()
    _assert_same_geometry(expected.collision, door.root.collision)


def test_entry_way_still_cuts_a_wall(empty_world):
    with empty_world.modify_world():
        wall = Wall.create_with_new_body_in_world(
            name="wall", world=empty_world, scale=Scale(0.1, 4, 3)
        )
        door = Door.create_with_new_body_in_world(
            name="door", world=empty_world, scale=Scale(0.03, 1, 2)
        )
    shape_count_before = len(wall.root.collision.shapes)

    with empty_world.modify_world():
        wall.add(door.entry_way)

    assert door.entry_way in wall.apertures
    assert len(wall.root.collision.shapes) != shape_count_before
    assert door.entry_way.root.parent_connection.parent is wall.root


def test_door_spawns_its_entry_way_as_a_part(empty_world):
    with empty_world.modify_world():
        door = Door.create_with_new_body_in_world(name="door", world=empty_world)
    assert isinstance(door.entry_way, EntryWay)
    assert door.entry_way.root.parent_connection.parent is door.root
    assert isinstance(door.entry_way.root.parent_connection, FixedConnection)
    assert door.entry_way in empty_world.get_semantic_annotations_by_type(EntryWay)


def test_entry_way_is_coextensive_with_the_door_leaf(empty_world):
    with empty_world.modify_world():
        door = Door.create_with_new_body_in_world(
            name="door", world=empty_world, scale=Scale(0.03, 1, 2)
        )
    np.testing.assert_allclose(
        door.entry_way.root.area.combined_mesh.bounds,
        door.root.collision.combined_mesh.bounds,
    )


def test_entry_way_geometry_is_translucent_white(empty_world):
    with empty_world.modify_world():
        door = Door.create_with_new_body_in_world(name="door", world=empty_world)
    assert door.entry_way.root.area[0].color == Color(R=1.0, G=1.0, B=1.0, A=0.2)


def test_entry_way_is_named_after_the_door(empty_world):
    with empty_world.modify_world():
        door = Door.create_with_new_body_in_world(name="door", world=empty_world)
    assert door.entry_way.name == PrefixedName("door_entry_way")


def test_door_default_specification_declares_the_entry_way_part():
    specification = Door.get_annotation_specification(
        "door", Door.get_default_root_kinematic_structure_entity_specification()
    )
    [entry_way_binding] = [
        binding
        for binding in specification.part_bindings
        if binding.field_name == "entry_way"
    ]
    [entry_way_specification] = entry_way_binding.specifications
    assert entry_way_specification.semantic_annotation_type is EntryWay


def test_caller_supplied_entry_way_specification_replaces_the_default(empty_world):
    custom = EntryWay.get_annotation_specification(
        "custom",
        EntryWay.get_default_root_kinematic_structure_entity_specification(
            scale=Scale(0.03, 0.5, 1)
        ),
    )
    door = Door.get_annotation_specification(
        "door",
        Door.get_default_root_kinematic_structure_entity_specification(),
        part_specifications={"entry_way": custom},
    ).spawn(empty_world)
    assert door.entry_way.name == PrefixedName("custom")
    assert len(empty_world.get_semantic_annotations_by_type(EntryWay)) == 1


def test_caller_supplied_parts_survive_entry_way_injection(empty_world):
    handle_part = Handle.get_annotation_specification(
        "handle",
        Handle.get_default_root_kinematic_structure_entity_specification(
            scale=Scale(0.1, 0.05, 0.05)
        ),
    )
    door = Door.get_annotation_specification(
        "door",
        Door.get_default_root_kinematic_structure_entity_specification(),
        part_specifications={"handle": handle_part},
    ).spawn(empty_world)
    assert isinstance(door.handle, Handle)
    assert isinstance(door.entry_way, EntryWay)


def test_entry_way_in_annotation_kwargs_raises():
    with pytest.raises(PartWholeFieldInAnnotationKwargs):
        Door.get_annotation_specification(
            "door",
            Door.get_default_root_kinematic_structure_entity_specification(),
            annotation_kwargs={"entry_way": None},
        )


def test_door_with_type_spawns_an_entry_way(empty_world):
    with empty_world.modify_world():
        door = DoorWithType.create_with_new_body_in_world(
            name="typed_door", world=empty_world
        )
    assert isinstance(door.entry_way, EntryWay)


def test_only_aperture_fields_remove_geometry_from_the_whole():
    # The cut is declared by the relationship, not by the part's type: the same EntryWay
    # cuts a wall it is mounted into but not the door it belongs to.
    apertures = IsPartWholeRelationship.of_field(HasApertures, "apertures")
    entry_way = IsPartWholeRelationship.of_field(Door, "entry_way")
    assert apertures.removes_part_geometry_from_whole
    assert not entry_way.removes_part_geometry_from_whole


def test_aperture_default_annotation_specification_without_scale_uses_default_scale(
    empty_world,
):
    # Aperture overrides get_default_root_kinematic_structure_entity_specification to fall back to Scale()
    # where the base yields a geometry-less spec; dispatch must reach that override.
    annotation = Aperture.get_annotation_specification(
        "aperture", Aperture.get_default_root_kinematic_structure_entity_specification()
    ).spawn(empty_world)
    expected = Aperture.get_default_root_kinematic_structure_entity_specification(
        "aperture", Scale()
    ).to_domain_object()
    _assert_same_geometry(expected.area, annotation.root.area)


# %% specification persistence
# A specification that survives a database round trip must still describe the same
# annotation, otherwise persisting one silently loses its geometry and its parts.


def _drawer_specification_with_handle() -> SemanticAnnotationWithRootSpecification:
    return Drawer.get_annotation_specification(
        "drawer",
        Drawer.get_default_root_kinematic_structure_entity_specification(
            scale=Scale(0.2, 0.3, 0.2)
        ),
        part_specifications={
            "handle": Handle.get_annotation_specification(
                "handle",
                Handle.get_default_root_kinematic_structure_entity_specification(
                    scale=Scale(0.1, 0.05, 0.05)
                ),
            )
        },
    )


def test_part_bindings_survive_orm_round_trip():
    # The parts a specification declares are part of what is persisted: the annotation
    # type and the part-whole fields the parts were bound to come back unchanged.
    reconstructed = to_dao(_drawer_specification_with_handle()).from_dao()

    assert reconstructed.semantic_annotation_type is Drawer
    assert [binding.field_name for binding in reconstructed.part_bindings] == ["handle"]


def test_annotation_specification_survives_orm_round_trip(empty_world):
    reconstructed = to_dao(_drawer_specification_with_handle()).from_dao()

    drawer = reconstructed.spawn(empty_world)
    assert isinstance(drawer, Drawer)
    assert isinstance(drawer.handle, Handle)
    assert len(drawer.root.collision.shapes) > 0
