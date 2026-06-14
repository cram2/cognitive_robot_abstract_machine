from pathlib import Path

import numpy as np
import pytest

from semantic_digital_twin.api.specifications import (
    BodySpecification,
    RegionSpecification,
    BodyAndConnectionSpecification,
    SemanticAnnotationWithRootSpecification,
    WorldSpecification,
    WorldEntitySpawnSpecification,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import ParsingError, UselessConceptError
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobotPart
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Milk,
    Slider,
    Handle,
    Door,
    Floor,
    Wall,
    Aperture,
    Drawer,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    PrismaticConnection,
    Connection6DoF,
    OmniDrive,
)
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
    spec = BodyAndConnectionSpecification(
        body_specification=BodySpecification.box("box", Scale(1, 1, 1)),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=1, y=2, z=3),
    )
    body = spec.spawn(empty_world, name="renamed")
    assert body.name == PrefixedName("renamed")
    root_T_body = empty_world.compute_forward_kinematics(empty_world.root, body)
    np.testing.assert_allclose(root_T_body.to_position().to_np()[:3], [1, 2, 3])


def test_body_and_connection_spawn_arg_overrides_stored_pose(empty_world):
    spec = BodyAndConnectionSpecification(
        body_specification=BodySpecification.box("box", Scale(1, 1, 1)),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=1),
    )
    body = spec.spawn(
        empty_world,
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=5),
    )
    root_T_body = empty_world.compute_forward_kinematics(empty_world.root, body)
    np.testing.assert_allclose(root_T_body.to_position().to_np()[0], 5)


def test_body_and_connection_active(empty_world):
    spec = BodyAndConnectionSpecification(
        body_specification=BodySpecification.box("drawer", Scale(1, 1, 1)),
        connection_type=PrismaticConnection,
        axis=Vector3.Z(),
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


def test_active_connection_requires_parameters_body():
    with pytest.raises(ValueError):
        BodyAndConnectionSpecification(
            body_specification=BodySpecification.box("b", Scale(1, 1, 1)),
            connection_type=PrismaticConnection,
        )


def test_active_annotation_requires_parameters():
    with pytest.raises(ValueError):
        SemanticAnnotationWithRootSpecification(
            name="slider",
            semantic_annotation_type=Slider,
            root_specification=None,
        )


def test_spec_valued_annotation_kwargs_not_supported(empty_world):
    spec = SemanticAnnotationWithRootSpecification(
        name="milk",
        semantic_annotation_type=Milk,
        root_specification=None,
        annotation_kwargs={"foo": BodySpecification.box("nested", Scale(1, 1, 1))},
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


def test_to_domain_object_generic_resolution():
    assert isinstance(
        BodySpecification.box("b", Scale(1, 1, 1)).to_domain_object(), Body
    )
    assert isinstance(
        RegionSpecification.box("r", Scale(1, 1, 1)).to_domain_object(), Region
    )


def test_body_and_connection_6dof(empty_world):
    spec = BodyAndConnectionSpecification(
        body_specification=BodySpecification.box("free", Scale(1, 1, 1)),
        connection_type=Connection6DoF,
    )
    body = spec.spawn(empty_world)
    assert isinstance(body.parent_connection, Connection6DoF)


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
    spec_body = Milk.get_default_body_specification(
        "milk", scale
    ).to_domain_object()
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
    spec_body = Door.get_default_body_specification(
        "door", scale
    ).to_domain_object()
    _assert_same_geometry(spec_body.collision, factory.root.collision)


def test_default_spec_door_validates_plane():
    with pytest.raises(Exception):
        Door.get_default_body_specification("door", Scale(2, 1, 1))


def test_default_spec_matches_floor(empty_world):
    scale = Scale(2, 2, 0.1)
    with empty_world.modify_world():
        factory = Floor.create_with_new_body_in_world(
            name=PrefixedName("floor"), world=empty_world, scale=scale
        )
    spec_body = Floor.get_default_body_specification(
        "floor", scale
    ).to_domain_object()
    _assert_same_geometry(spec_body.collision, factory.root.collision)


def test_default_spec_matches_wall(empty_world):
    scale = Scale(0.1, 4, 2)
    with empty_world.modify_world():
        factory = Wall.create_with_new_body_in_world(
            name=PrefixedName("wall"), world=empty_world, scale=scale
        )
    spec_body = Wall.get_default_body_specification(
        "wall", scale
    ).to_domain_object()
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


def test_annotation_spec_active_requires_axis():
    # Slider's parent connection is active, so the spec construction must demand an axis.
    with pytest.raises(ValueError):
        Slider.get_default_annotation_specification("slider", Scale(0.1, 0.1, 0.1))


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
