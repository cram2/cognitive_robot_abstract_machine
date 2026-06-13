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
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk, Slider
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    PrismaticConnection,
    Connection6DoF,
)
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body, Region


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
    parent_spec.child_specification.append(BodySpecification.box("child", Scale(1, 1, 1)))
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
