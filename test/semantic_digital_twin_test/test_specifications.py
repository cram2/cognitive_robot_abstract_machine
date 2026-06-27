import copy
import inspect
from pathlib import Path

import numpy as np
import pytest

from semantic_digital_twin.api.specifications import (
    BodySpecification,
    RegionSpecification,
    ConnectedBodySpecification,
    ConnectionSpecification,
    FixedConnectionSpecification,
    Connection6DoFSpecification,
    PrismaticConnectionSpecification,
    RevoluteConnectionSpecification,
    SemanticAnnotationWithRootSpecification,
    WorldSpecification,
    WorldEntitySpawnSpecification,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    InvalidPlaneDimensions,
    MissingConnectionAxisError,
    MissingConnectionChildError,
    ParsingError,
    UselessConceptError,
)
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobotPart
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Milk,
    Slider,
    Handle,
    Hinge,
    Door,
    Floor,
    Wall,
    Aperture,
    Drawer,
    Table,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootBody,
    HasRootRegion,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
    Connection6DoF,
    OmniDrive,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world_description.geometry import Scale, Box
from semantic_digital_twin.world_description.inertial_properties import Inertial
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body, Region

RESOURCES = (
    Path(__file__).resolve().parents[2] / "semantic_digital_twin" / "resources" / "stl"
)


@pytest.fixture
def empty_world() -> World:
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("root", "world")))
    return world


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


def test_body_and_connection_pose_and_name_override(empty_world):
    body_spec = BodySpecification.box("box", Scale(1, 1, 1))
    body_spec.parent_T_self = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1, y=2, z=3
    )
    spec = ConnectedBodySpecification(body_specification=body_spec)
    body = spec.spawn(empty_world, name="renamed")
    assert body.name == PrefixedName("renamed")
    root_T_body = empty_world.compute_forward_kinematics(empty_world.root, body)
    np.testing.assert_allclose(root_T_body.to_position().to_np()[:3], [1, 2, 3])


def test_body_and_connection_spawn_arg_overrides_stored_pose(empty_world):
    body_spec = BodySpecification.box("box", Scale(1, 1, 1))
    body_spec.parent_T_self = HomogeneousTransformationMatrix.from_xyz_rpy(x=1)
    spec = ConnectedBodySpecification(body_specification=body_spec)
    body = spec.spawn(
        empty_world,
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=5),
    )
    root_T_body = empty_world.compute_forward_kinematics(empty_world.root, body)
    np.testing.assert_allclose(root_T_body.to_position().to_np()[0], 5)


def test_body_and_connection_active(empty_world):
    spec = ConnectedBodySpecification(
        body_specification=BodySpecification.box("drawer", Scale(1, 1, 1)),
        connection_specification=PrismaticConnectionSpecification(axis=Vector3.Z()),
    )
    body = spec.spawn(empty_world)
    assert isinstance(body.parent_connection, PrismaticConnection)


def test_child_specification_recursion(empty_world):
    parent_spec = BodySpecification.box("parent", Scale(1, 1, 1))
    parent_spec.child_specification.append(
        BodySpecification.box("child", Scale(1, 1, 1))
    )
    parent_body = parent_spec.spawn(empty_world)
    child = empty_world.get_body_by_name("child")
    assert child.parent_connection.parent is parent_body


def test_fixed_annotation_spawns(empty_world):
    spec = SemanticAnnotationWithRootSpecification(
        name="milk",
        semantic_annotation_type=Milk,
        root_specification=BodySpecification.box("milk", Scale(0.1, 0.1, 0.2)),
    )
    annotation = spec.spawn(empty_world)
    assert isinstance(annotation, Milk)
    assert annotation in empty_world.semantic_annotations
    assert isinstance(annotation.root.parent_connection, FixedConnection)


def test_active_annotation_spawns(empty_world):
    spec = SemanticAnnotationWithRootSpecification(
        name="slider",
        semantic_annotation_type=Slider,
        root_specification=BodySpecification.box("slider", Scale(0.1, 0.1, 0.1)),
        axis=Vector3.Z(),
    )
    annotation = spec.spawn(empty_world)
    assert isinstance(annotation, Slider)
    assert isinstance(annotation.root.parent_connection, PrismaticConnection)


def test_active_connection_requires_parameters_body(empty_world):
    # An active connection without an axis is rejected at spawn time by create_with_dofs.
    spec = ConnectedBodySpecification(
        body_specification=BodySpecification.box("b", Scale(1, 1, 1)),
        connection_specification=PrismaticConnectionSpecification(),
    )
    with pytest.raises(MissingConnectionAxisError):
        spec.spawn(empty_world)


def test_active_annotation_requires_parameters(empty_world):
    # Slider's parent connection is active, so spawning without an axis must raise.
    spec = SemanticAnnotationWithRootSpecification(
        name="slider",
        semantic_annotation_type=Slider,
        root_specification=None,
    )
    with pytest.raises(MissingConnectionAxisError):
        spec.spawn(empty_world)


def test_nested_annotation_on_non_part_whole_field_raises(empty_world):
    # Milk has no part-whole field, so a nested annotation spec cannot be mounted onto it.
    spec = SemanticAnnotationWithRootSpecification(
        name="milk",
        semantic_annotation_type=Milk,
        root_specification=BodySpecification.box("milk", Scale(0.1, 0.1, 0.2)),
        annotation_kwargs={
            "handle": Handle.get_default_annotation_specification(
                "handle", Scale(0.1, 0.05, 0.05)
            )
        },
    )
    with pytest.raises(NotImplementedError):
        spec.spawn(empty_world)


def test_world_specification_robotless():
    world = WorldSpecification(
        starting_objects=[BodySpecification.box("obj", Scale(1, 1, 1))]
    ).to_world()
    assert not world.is_empty()
    assert world.get_body_by_name("obj") is not None


@pytest.mark.parametrize(
    "make_spec",
    [
        lambda: BodySpecification.box("shape", Scale(1, 1, 1)),
        lambda: BodySpecification.sphere("shape", 0.5),
        lambda: BodySpecification.cylinder("shape", 0.4, 1.0),
        lambda: BodySpecification.mesh("shape", str(RESOURCES / "milk.stl")),
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


def test_body_specification_from_3d_points_matches_direct_construction():
    """``from_3d_points`` is the declarative counterpart of :meth:`Body.from_3d_points`."""
    points = [Point3(0, 0, 0), Point3(1, 0, 0), Point3(0, 1, 0), Point3(1, 1, 1)]
    name = PrefixedName("polytope")

    materialized = BodySpecification.from_3d_points(name, points).to_domain_object(name)
    directly_built = Body.from_3d_points(name=name, points_3d=points)

    assert len(materialized.collision.shapes) == len(directly_built.collision.shapes) == 1


def test_has_root_body_default_specification_without_scale_is_geometryless(empty_world):
    """A scale-less body factory yields a bare body, mirrored by an empty body spec."""
    spec = HasRootBody.get_default_body_specification(PrefixedName("bare_body"))
    assert isinstance(spec, BodySpecification)

    body = spec.spawn(empty_world)
    assert isinstance(body, Body)
    assert len(body.collision.shapes) == 0


def test_has_root_region_default_specification_without_scale_is_geometryless(empty_world):
    """The base region factory creates a bare region, mirrored by an empty region spec."""
    spec = HasRootRegion.get_default_region_specification(PrefixedName("bare_region"))
    assert isinstance(spec, RegionSpecification)

    region = spec.spawn(empty_world)
    assert isinstance(region, Region)
    assert len(region.area.shapes) == 0


def test_constructor_child_specification_param(empty_world):
    parent = BodySpecification.box(
        "parent",
        Scale(1, 1, 1),
        child_specification=[BodySpecification.box("child", Scale(1, 1, 1))],
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
    pose, and each materialized connection must own a distinct pose bound to its own child.
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
    spec = ConnectedBodySpecification(
        body_specification=BodySpecification.box("free", Scale(1, 1, 1)),
        connection_specification=Connection6DoFSpecification(),
    )
    body = spec.spawn(empty_world)
    assert isinstance(body.parent_connection, Connection6DoF)


def test_connected_body_specification_name_defaults_to_body_name():
    spec = ConnectedBodySpecification(
        body_specification=BodySpecification.box("box", Scale(1, 1, 1))
    )
    assert spec.name == PrefixedName("box")


def test_spawn_positional_name(empty_world):
    body = BodySpecification.box("b", Scale(1, 1, 1)).spawn(empty_world, "renamed")
    assert body.name == PrefixedName("renamed")


def test_world_specification_with_robot():
    try:
        world = WorldSpecification(
            robot_semantic_annotation=PR2,
            drive_connection_type=OmniDrive,
            world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0),
            odom_T_robot_start=HomogeneousTransformationMatrix.from_xyz_rpy(y=2.0),
        ).to_world()
    except ParsingError as error:
        pytest.skip(f"PR2 URDF not available: {error}")

    map_body = world.get_body_by_name("map")
    odom_body = world.get_body_by_name("odom_combined")
    assert map_body is not None
    assert odom_body is not None
    assert isinstance(odom_body.parent_connection, Connection6DoF)

    drive = world.get_body_by_name("base_footprint").parent_connection
    assert isinstance(drive, OmniDrive)

    map_T_odom = world.compute_forward_kinematics(map_body, odom_body)
    np.testing.assert_allclose(map_T_odom.to_position().to_np()[0], 1.0)


def test_world_specification_annotation_starting_object():
    world = WorldSpecification(
        starting_objects=[
            SemanticAnnotationWithRootSpecification(
                name="milk",
                semantic_annotation_type=Milk,
                root_specification=BodySpecification.box("milk", Scale(0.1, 0.1, 0.2)),
            )
        ]
    ).to_world()
    milks = world.get_semantic_annotations_by_type(Milk)
    assert len(milks) == 1


#####################################################################
# ConnectionSpecification captures a connection type and the keyword
# arguments forwarded to its create_with_dofs.
#####################################################################


def test_fixed_connection_spec_binds_type_without_params():
    spec = FixedConnectionSpecification()
    assert spec.connection_type is FixedConnection
    assert spec._create_with_dofs_kwargs() == {}


def test_connection_6dof_spec_binds_type_without_params():
    spec = Connection6DoFSpecification()
    assert spec.connection_type is Connection6DoF
    assert spec._create_with_dofs_kwargs() == {}


def test_active_1dof_spec_captures_parameters():
    limits = DegreeOfFreedomLimits(
        lower=DerivativeMap(velocity=-1.0), upper=DerivativeMap(velocity=1.0)
    )
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
    spec = RevoluteConnectionSpecification()
    assert spec.connection_type is RevoluteConnection
    assert spec._create_with_dofs_kwargs() == {
        "axis": None,
        "multiplier": 1.0,
        "offset": 0.0,
        "dof_limits": None,
    }


def test_parameterized_active_consumes_parameters():
    axis = Vector3.Z()
    spec = PrismaticConnectionSpecification.from_kwargs(axis=axis, multiplier=2.0)
    assert isinstance(spec, PrismaticConnectionSpecification)
    assert spec.axis is axis
    assert spec.multiplier == 2.0


def test_parameterized_fixed_ignores_active_parameters():
    # A fixed spec must accept and ignore active parameters, so a caller holding a bare
    # specification type can parameterize any family uniformly.
    spec = FixedConnectionSpecification.from_kwargs(axis=Vector3.Z(), multiplier=2.0)
    assert isinstance(spec, FixedConnectionSpecification)
    assert spec._create_with_dofs_kwargs() == {}


def test_active_1dof_spec_kwargs_match_create_with_dofs_signature():
    # The forwarded kwargs must be keyword arguments that create_with_dofs accepts,
    # otherwise the specification cannot materialize its connection.
    spec = PrismaticConnectionSpecification(
        axis=Vector3.Z(), multiplier=2.0, offset=0.5
    )
    accepted_parameters = inspect.signature(
        PrismaticConnection.create_with_dofs
    ).parameters
    assert set(spec._create_with_dofs_kwargs()).issubset(accepted_parameters)


def test_connection_spec_spawn_fixed(empty_world):
    child = BodySpecification.box("child", Scale(1, 1, 1)).to_domain_object()
    connection = FixedConnectionSpecification().spawn(
        empty_world, parent=empty_world.root, child=child
    )
    assert isinstance(connection, FixedConnection)
    assert connection.parent is empty_world.root
    assert connection.child is child
    assert child in empty_world.bodies


def test_connection_spec_spawn_defaults_parent_to_root(empty_world):
    child = BodySpecification.box("child", Scale(1, 1, 1)).to_domain_object()
    connection = Connection6DoFSpecification().spawn(empty_world, child=child)
    assert isinstance(connection, Connection6DoF)
    assert connection.parent is empty_world.root


def test_connection_spec_spawn_active_forwards_kwargs(empty_world):
    limits = DegreeOfFreedomLimits(
        lower=DerivativeMap(velocity=-1.5), upper=DerivativeMap(velocity=1.5)
    )
    child = BodySpecification.box("slider", Scale(1, 1, 1)).to_domain_object()
    connection = PrismaticConnectionSpecification(
        axis=Vector3.Z(), dof_limits=limits
    ).spawn(empty_world, child=child)
    assert isinstance(connection, PrismaticConnection)
    assert connection.dof.limits.upper.velocity == 1.5
    assert connection.dof.limits.lower.velocity == -1.5


def test_connection_spec_spawn_applies_pose(empty_world):
    child = BodySpecification.box("child", Scale(1, 1, 1)).to_domain_object()
    FixedConnectionSpecification().spawn(
        empty_world,
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=1, y=2, z=3),
        child=child,
    )
    root_T_child = empty_world.compute_forward_kinematics(empty_world.root, child)
    np.testing.assert_allclose(root_T_child.to_position().to_np()[:3], [1, 2, 3])


def test_connection_spec_spawn_without_child_raises(empty_world):
    with pytest.raises(MissingConnectionChildError):
        FixedConnectionSpecification().spawn(empty_world)


def test_connection_spec_spawn_without_name_matches_direct_creation(empty_world):
    # A nameless spec must auto-generate the same connection name as creating the
    # connection directly between an identically-named parent and child.
    spec_child = BodySpecification.box("child", Scale(1, 1, 1)).to_domain_object()
    spec_connection = FixedConnectionSpecification().spawn(
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


@pytest.mark.parametrize(
    "annotation_type, expected_specification_type",
    [
        (Milk, FixedConnectionSpecification),
        (Slider, PrismaticConnectionSpecification),
        (Hinge, RevoluteConnectionSpecification),
    ],
)
def test_annotation_declares_parent_connection_specification_type(
    annotation_type, expected_specification_type
):
    assert (
        annotation_type._parent_connection_specification_type
        is expected_specification_type
    )


#####################################################################
# get_default_body_specification / get_default_region_specification
# reproduce the geometry that create_with_new_body_in_world(scale=...)
# (and Aperture's region factory) generate, for every class that
# implements its own geometry-generating factory override.
#####################################################################


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
            name=PrefixedName("milk"), world=empty_world, scale=scale
        )
    spec_body = Milk.get_default_body_specification("milk", scale).to_domain_object()
    _assert_same_geometry(spec_body.collision, factory.root.collision)
    assert spec_body.collision is spec_body.visual


def test_default_spec_matches_case_body(empty_world):
    scale = Scale(0.3, 0.4, 0.5)
    with empty_world.modify_world():
        factory = Drawer.create_with_new_body_in_world(
            name=PrefixedName("drawer"), world=empty_world, scale=scale
        )
    spec_body = Drawer.get_default_body_specification(
        "drawer", scale
    ).to_domain_object()
    _assert_same_geometry(spec_body.collision, factory.root.collision)
    assert spec_body.collision is spec_body.visual


def test_default_spec_matches_case_body_with_wall_thickness(empty_world):
    scale = Scale(0.4, 0.5, 0.6)
    with empty_world.modify_world():
        factory = Drawer.create_with_new_body_in_world(
            name=PrefixedName("drawer"),
            world=empty_world,
            scale=scale,
            wall_thickness=0.05,
        )
    spec_body = Drawer.get_default_body_specification(
        "drawer", scale, wall_thickness=0.05
    ).to_domain_object()
    _assert_same_geometry(spec_body.collision, factory.root.collision)


def test_default_spec_matches_handle(empty_world):
    scale = Scale(0.1, 0.05, 0.05)
    with empty_world.modify_world():
        factory = Handle.create_with_new_body_in_world(
            name=PrefixedName("handle"), world=empty_world, scale=scale, thickness=0.01
        )
    spec_body = Handle.get_default_body_specification(
        "handle", scale, thickness=0.01
    ).to_domain_object()
    _assert_same_geometry(spec_body.collision, factory.root.collision)


def test_default_spec_matches_door(empty_world):
    scale = Scale(0.03, 1, 2)
    with empty_world.modify_world():
        factory = Door.create_with_new_body_in_world(
            name=PrefixedName("door"), world=empty_world, scale=scale
        )
    spec_body = Door.get_default_body_specification("door", scale).to_domain_object()
    _assert_same_geometry(spec_body.collision, factory.root.collision)


def test_default_spec_door_validates_plane():
    with pytest.raises(InvalidPlaneDimensions):
        Door.get_default_body_specification("door", Scale(2, 1, 1))


def test_default_spec_matches_floor(empty_world):
    scale = Scale(2, 2, 0.1)
    with empty_world.modify_world():
        factory = Floor.create_with_new_body_in_world(
            name=PrefixedName("floor"), world=empty_world, scale=scale
        )
    spec_body = Floor.get_default_body_specification("floor", scale).to_domain_object()
    _assert_same_geometry(spec_body.collision, factory.root.collision)


def test_default_spec_matches_wall(empty_world):
    scale = Scale(0.1, 4, 2)
    with empty_world.modify_world():
        factory = Wall.create_with_new_body_in_world(
            name=PrefixedName("wall"), world=empty_world, scale=scale
        )
    spec_body = Wall.get_default_body_specification("wall", scale).to_domain_object()
    _assert_same_geometry(spec_body.collision, factory.root.collision)


def test_default_spec_matches_aperture_region(empty_world):
    scale = Scale(0.1, 1, 2)
    with empty_world.modify_world():
        factory = Aperture.create_with_new_region_in_world(
            name=PrefixedName("aperture"), world=empty_world, scale=scale
        )
    spec_region = Aperture.get_default_region_specification(
        "aperture", scale
    ).to_domain_object()
    _assert_same_geometry(spec_region.area, factory.root.area)


def test_default_spec_robot_part_raises():
    # AbstractRobotPart's geometry must come from URDF parsing, not from scale,
    # so it raises just like its create_with_new_body_in_world override.
    with pytest.raises(UselessConceptError):
        AbstractRobotPart.get_default_body_specification("part", Scale(1, 1, 1))


#####################################################################
# get_default_annotation_specification wraps the geometry spec into a
# SemanticAnnotationWithRootSpecification that spawns an annotation
# equivalent to create_with_new_body_in_world.
#####################################################################


def _fresh_world() -> World:
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("root", "world")))
    return world


def test_annotation_spec_base_body(empty_world):
    scale = Scale(0.2, 0.3, 0.4)
    spec = Milk.get_default_annotation_specification("milk", scale)
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
            name=PrefixedName("milk_factory"), world=factory_world, scale=scale
        )
    _assert_same_geometry(annotation.root.collision, factory.root.collision)


def test_annotation_spec_active_slider(empty_world):
    scale = Scale(0.1, 0.1, 0.1)
    spec = Slider.get_default_annotation_specification(
        "slider", scale, active_axis=Vector3.Z()
    )
    annotation = spec.spawn(empty_world)
    assert isinstance(annotation, Slider)
    assert isinstance(annotation.root.parent_connection, PrismaticConnection)


def test_annotation_spec_active_requires_axis(empty_world):
    # Slider's parent connection is active, so spawning without an axis must raise.
    spec = Slider.get_default_annotation_specification("slider", Scale(0.1, 0.1, 0.1))
    with pytest.raises(MissingConnectionAxisError):
        spec.spawn(empty_world)


def test_annotation_spec_case_forwards_wall_thickness(empty_world):
    scale = Scale(0.4, 0.5, 0.6)
    spec = Drawer.get_default_annotation_specification(
        "drawer", scale, wall_thickness=0.05
    )
    annotation = spec.spawn(empty_world)
    factory_world = _fresh_world()
    with factory_world.modify_world():
        factory = Drawer.create_with_new_body_in_world(
            name=PrefixedName("drawer_factory"),
            world=factory_world,
            scale=scale,
            wall_thickness=0.05,
        )
    _assert_same_geometry(annotation.root.collision, factory.root.collision)


def test_annotation_spec_handle_forwards_thickness(empty_world):
    scale = Scale(0.1, 0.05, 0.05)
    spec = Handle.get_default_annotation_specification("handle", scale, thickness=0.01)
    annotation = spec.spawn(empty_world)
    factory_world = _fresh_world()
    with factory_world.modify_world():
        factory = Handle.create_with_new_body_in_world(
            name=PrefixedName("handle_factory"),
            world=factory_world,
            scale=scale,
            thickness=0.01,
        )
    _assert_same_geometry(annotation.root.collision, factory.root.collision)


def test_annotation_spec_aperture_region(empty_world):
    scale = Scale(0.1, 1, 2)
    spec = Aperture.get_default_annotation_specification("aperture", scale)
    assert isinstance(spec.root_specification, RegionSpecification)

    annotation = spec.spawn(empty_world)
    assert isinstance(annotation, Aperture)
    factory_world = _fresh_world()
    with factory_world.modify_world():
        factory = Aperture.create_with_new_region_in_world(
            name=PrefixedName("aperture_factory"), world=factory_world, scale=scale
        )
    _assert_same_geometry(annotation.root.area, factory.root.area)


def test_annotation_spec_robot_part_raises():
    with pytest.raises(UselessConceptError):
        AbstractRobotPart.get_default_annotation_specification("part", Scale(1, 1, 1))


#####################################################################
# Nested annotations: spec-valued annotation_kwargs are spawned and
# mounted via the part-whole `add`, keyed by the target field name.
#####################################################################


def _spawn_with_parts(world, whole_type, whole_scale, parts):
    """Spawn ``whole_type`` from its default annotation spec, with ``parts`` as nested annotations."""
    return whole_type.get_default_annotation_specification(
        "whole", whole_scale, annotation_kwargs=parts
    ).spawn(world)


def test_nested_handle_attaches_as_child(empty_world):
    handle_part = Handle.get_default_annotation_specification(
        "handle", Scale(0.1, 0.05, 0.05)
    )
    drawer = _spawn_with_parts(
        empty_world, Drawer, Scale(0.4, 0.5, 0.6), {"handle": handle_part}
    )
    assert isinstance(drawer.handle, Handle)
    assert drawer.handle.root.parent_connection.parent is drawer.root
    assert isinstance(drawer.handle.root.parent_connection, FixedConnection)


def test_nested_mechanical_joint_reparents_whole(empty_world):
    hinge_part = Hinge.get_default_annotation_specification(
        "hinge", Scale(0.05, 0.05, 0.05), active_axis=Vector3.Z()
    )
    drawer = _spawn_with_parts(
        empty_world, Drawer, Scale(0.4, 0.5, 0.6), {"mechanical_joint": hinge_part}
    )
    assert isinstance(drawer.mechanical_joint, Hinge)
    # whole_parent -(revolute)-> hinge -(fixed)-> whole
    assert drawer.root.parent_connection.parent is drawer.mechanical_joint.root
    assert drawer.mechanical_joint.root.parent_connection.parent is empty_world.root
    assert isinstance(
        drawer.mechanical_joint.root.parent_connection, RevoluteConnection
    )


def test_nested_aperture_cuts_geometry(empty_world):
    plain_wall = Wall.get_default_annotation_specification(
        "plain_wall", Scale(0.1, 2, 2)
    ).spawn(empty_world)
    plain_shape_count = len(plain_wall.root.collision.shapes)

    aperture_part = Aperture.get_default_annotation_specification(
        "hole", Scale(0.1, 0.5, 0.5)
    )
    wall = _spawn_with_parts(
        empty_world, Wall, Scale(0.1, 2, 2), {"apertures": aperture_part}
    )
    assert len(wall.apertures) == 1
    assert isinstance(wall.apertures[0], Aperture)
    # cutting the aperture out of the wall changes its collision geometry
    assert len(wall.root.collision.shapes) != plain_shape_count


def test_nested_list_valued_parts_on_to_many_field(empty_world):
    aperture_a = Aperture.get_default_annotation_specification(
        "hole_a", Scale(0.1, 0.5, 0.5)
    )
    aperture_a.root_specification.parent_T_self = (
        HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.8)
    )
    aperture_b = Aperture.get_default_annotation_specification(
        "hole_b", Scale(0.1, 0.5, 0.5)
    )
    aperture_b.root_specification.parent_T_self = (
        HomogeneousTransformationMatrix.from_xyz_rpy(y=0.8)
    )
    wall = Wall.get_default_annotation_specification(
        "wall",
        Scale(0.1, 3, 3),
        annotation_kwargs={"apertures": [aperture_a, aperture_b]},
    ).spawn(empty_world)
    assert len(wall.apertures) == 2
    assert all(isinstance(aperture, Aperture) for aperture in wall.apertures)


def test_list_value_on_singular_part_field_raises(empty_world):
    spec = Drawer.get_default_annotation_specification(
        "drawer",
        Scale(0.4, 0.5, 0.6),
        annotation_kwargs={
            "handle": [
                Handle.get_default_annotation_specification(
                    "h1", Scale(0.1, 0.05, 0.05)
                ),
                Handle.get_default_annotation_specification(
                    "h2", Scale(0.1, 0.05, 0.05)
                ),
            ]
        },
    )
    with pytest.raises(NotImplementedError):
        spec.spawn(empty_world)


def test_nested_part_placement_is_relative_to_whole(empty_world):
    handle_part = Handle.get_default_annotation_specification(
        "handle", Scale(0.1, 0.05, 0.05)
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
        drawer_T_handle.to_position().to_np()[:3], [0, 0.5, 0], atol=1e-9
    )


def test_annotation_connection_limits_threaded(empty_world):
    limits = DegreeOfFreedomLimits(
        lower=DerivativeMap(velocity=-1.5), upper=DerivativeMap(velocity=1.5)
    )
    spec = Slider.get_default_annotation_specification(
        "slider",
        Scale(0.1, 0.1, 0.1),
        active_axis=Vector3.Z(),
        connection_limits=limits,
    )
    slider = spec.spawn(empty_world)
    dof_limits = slider.root.parent_connection.dof.limits
    assert dof_limits.upper.velocity == 1.5
    assert dof_limits.lower.velocity == -1.5


def test_inert_annotation_kwargs_reach_constructor(empty_world):
    existing_handle = Handle.get_default_annotation_specification(
        "existing", Scale(0.1, 0.05, 0.05)
    ).spawn(empty_world)
    drawer = Drawer.get_default_annotation_specification(
        "drawer", Scale(0.4, 0.5, 0.6), annotation_kwargs={"handle": existing_handle}
    ).spawn(empty_world)
    assert drawer.handle is existing_handle


def test_raw_entity_spec_in_annotation_kwargs_raises(empty_world):
    # Raw entity specs (e.g. a supporting-surface region) are not supported in annotation_kwargs.
    spec = Table.get_default_annotation_specification(
        "table",
        Scale(1, 1, 0.5),
        annotation_kwargs={
            "supporting_surface": RegionSpecification.box("surface", Scale(1, 1, 0.01))
        },
    )
    with pytest.raises(NotImplementedError):
        spec.spawn(empty_world)


def test_storage_objects_spec_in_annotation_kwargs_raises(empty_world):
    # IsStorageSpace.objects is not a part-whole relationship, so spec-based occupants are unsupported.
    spec = Table.get_default_annotation_specification(
        "table",
        Scale(1, 1, 0.5),
        annotation_kwargs={
            "objects": [
                Milk.get_default_annotation_specification("milk", Scale(0.1, 0.1, 0.2))
            ]
        },
    )
    with pytest.raises(NotImplementedError):
        spec.spawn(empty_world)


def test_complex_spawned_world_is_deepcopyable(empty_world):
    Drawer.get_default_annotation_specification(
        "drawer",
        Scale(0.4, 0.5, 0.6),
        annotation_kwargs={
            "handle": Handle.get_default_annotation_specification(
                "handle", Scale(0.1, 0.05, 0.05)
            ),
            "mechanical_joint": Hinge.get_default_annotation_specification(
                "hinge", Scale(0.05, 0.05, 0.05), active_axis=Vector3.Z()
            ),
        },
    ).spawn(empty_world)
    Wall.get_default_annotation_specification(
        "wall",
        Scale(0.1, 2, 2),
        annotation_kwargs={
            "apertures": Aperture.get_default_annotation_specification(
                "hole", Scale(0.1, 0.5, 0.5)
            )
        },
    ).spawn(empty_world)
    Milk.get_default_annotation_specification("milk", Scale(0.1, 0.1, 0.2)).spawn(
        empty_world
    )

    world_copy = copy.deepcopy(empty_world)

    assert world_copy is not empty_world
    assert len(world_copy.kinematic_structure_entities) == len(
        empty_world.kinematic_structure_entities
    )
    assert len(world_copy.connections) == len(empty_world.connections)
    assert len(list(world_copy.semantic_annotations)) == len(
        list(empty_world.semantic_annotations)
    )


def test_nested_composite_matches_imperative(empty_world):
    scale = Scale(0.4, 0.5, 0.6)
    handle_scale = Scale(0.1, 0.05, 0.05)
    hinge_scale = Scale(0.05, 0.05, 0.05)

    drawer = Drawer.get_default_annotation_specification(
        "drawer",
        scale,
        annotation_kwargs={
            "handle": Handle.get_default_annotation_specification(
                "handle", handle_scale
            ),
            "mechanical_joint": Hinge.get_default_annotation_specification(
                "hinge", hinge_scale, active_axis=Vector3.Z()
            ),
        },
    ).spawn(empty_world)

    assert isinstance(drawer.handle, Handle)
    assert isinstance(drawer.mechanical_joint, Hinge)
    assert drawer.handle.root.parent_connection.parent is drawer.root
    assert drawer.root.parent_connection.parent is drawer.mechanical_joint.root

    imperative_world = _fresh_world()
    with imperative_world.modify_world():
        imperative_drawer = Drawer.create_with_new_body_in_world(
            name=PrefixedName("drawer_imperative"), world=imperative_world, scale=scale
        )
        imperative_handle = Handle.create_with_new_body_in_world(
            name=PrefixedName("handle_imperative"),
            world=imperative_world,
            scale=handle_scale,
        )
        imperative_hinge = Hinge.create_with_new_body_in_world(
            name=PrefixedName("hinge_imperative"),
            world=imperative_world,
            scale=hinge_scale,
            active_axis=Vector3.Z(),
        )
        imperative_drawer.add(imperative_handle)
        imperative_drawer.add(imperative_hinge)

    _assert_same_geometry(drawer.root.collision, imperative_drawer.root.collision)
    _assert_same_geometry(
        drawer.handle.root.collision, imperative_handle.root.collision
    )
