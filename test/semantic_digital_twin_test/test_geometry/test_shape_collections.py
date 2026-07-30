import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale, Sphere
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


def test_post_init_transformation():
    w = World()
    root = Body(name=PrefixedName("root"))
    b1 = Body(name=PrefixedName("b1"))

    with w.modify_world():
        w.add_connection(
            FixedConnection(
                parent=root,
                child=b1,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1, reference_frame=root
                ),
            )
        )

    shape = Sphere(
        radius=1,
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(x=3, reference_frame=root),
    )
    shape_collection = ShapeCollection(
        shapes=[shape],
        reference_frame=b1,
    )
    shape_collection.transform_all_shapes_to_own_frame()
    assert shape.origin.reference_frame == b1
    assert shape.origin.to_position().x == 2.0

    shape = Sphere(
        radius=1,
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(x=3, reference_frame=root),
    )

    shape_collection = ShapeCollection(reference_frame=b1)
    shape_collection.append(shape)
    shape_collection.transform_all_shapes_to_own_frame()
    assert shape.origin.reference_frame == b1
    assert shape.origin.to_position().x == 2.0


# %% bounding-box dimensions


def test_depth_width_height_match_bounding_box_scale():
    world = World()
    root = Body(name=PrefixedName("root"))
    with world.modify_world():
        world.add_body(root)

    shape_collection = ShapeCollection(
        shapes=[
            Box(
                scale=Scale(0.1, 0.2, 0.3),
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    reference_frame=root
                ),
            )
        ],
        reference_frame=root,
    )

    bounding_box_scale = (
        shape_collection.as_bounding_box_collection_at_origin(
            HomogeneousTransformationMatrix(reference_frame=root)
        )
        .bounding_box()
        .scale
    )
    assert shape_collection.depth == bounding_box_scale.x == pytest.approx(0.1)
    assert shape_collection.width == bounding_box_scale.y == pytest.approx(0.2)
    assert shape_collection.height == bounding_box_scale.z == pytest.approx(0.3)
