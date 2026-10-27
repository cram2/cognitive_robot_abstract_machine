"""
Tests for the footprint of a body's shapes: the outline they cover on the x-y plane of a
frame, seen from above.
"""

import math

import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import NoShapesToProject
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


def _body_with_box(
    scale: Scale, root_T_body: HomogeneousTransformationMatrix
) -> tuple[Body, Body]:
    """
    A world holding one box-shaped body, placed relative to the world's root.

    :param scale: The box's size.
    :param root_T_body: Where the body stands relative to the root.
    :return: The root and the body.
    """
    world = World()
    root = Body(name=PrefixedName("root"))
    body = Body(name=PrefixedName("body"))
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=root, child=body, parent_T_connection_expression=root_T_body
            )
        )
        body.collision = ShapeCollection(
            shapes=[
                Box(
                    scale=scale,
                    origin=HomogeneousTransformationMatrix(reference_frame=body),
                )
            ],
            reference_frame=body,
        )
    return root, body


# %% the outline from above


def test_a_box_covers_its_own_base_in_its_own_frame():
    root, body = _body_with_box(
        Scale(0.02, 0.04, 0.03), HomogeneousTransformationMatrix(reference_frame=None)
    )

    footprint = body.collision.convex_footprint_in_frame(body)

    assert footprint.area == pytest.approx(0.02 * 0.04)
    assert footprint.bounding_box.depth == pytest.approx(0.02)
    assert footprint.bounding_box.width == pytest.approx(0.04)


def test_a_footprint_is_taken_where_the_body_stands_in_the_given_frame():
    root, body = _body_with_box(
        Scale(0.02, 0.04, 0.03),
        HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0, y=-2.0, yaw=math.pi / 2),
    )

    footprint = body.collision.convex_footprint_in_frame(root)

    assert footprint.centroid.to_np() == pytest.approx([1.0, -2.0])
    assert footprint.bounding_box.depth == pytest.approx(0.04)
    assert footprint.bounding_box.width == pytest.approx(0.02)


def test_a_box_lying_on_its_side_covers_its_side_from_above():
    root, body = _body_with_box(
        Scale(0.01, 0.02, 0.03),
        HomogeneousTransformationMatrix.from_xyz_rpy(roll=math.pi / 2),
    )

    footprint = body.collision.convex_footprint_in_frame(root)

    assert footprint.bounding_box.depth == pytest.approx(0.01)
    assert footprint.bounding_box.width == pytest.approx(0.03)


def test_a_collection_without_shapes_has_no_footprint():
    root, body = _body_with_box(
        Scale(0.01, 0.02, 0.03), HomogeneousTransformationMatrix(reference_frame=None)
    )

    with pytest.raises(NoShapesToProject):
        ShapeCollection(reference_frame=body).convex_footprint_in_frame(body)
