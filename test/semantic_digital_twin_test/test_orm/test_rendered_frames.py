"""
Persistence of ray-traced camera frame data.
"""

from __future__ import annotations

# %% Imports
import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from krrood.ormatic.data_access_objects.helper import to_dao
from semantic_digital_twin.datastructures.camera_rendering import (
    BodyColorAssignment,
    RenderedColorFrame,
    RenderedRGBDFrame,
    SegmentationDepthFrame,
)
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import Body

# %% Frame persistence


def test_segmentation_depth_frame_database_roundtrip(
    in_memory_session_maker: sessionmaker[Session],
) -> None:
    """
    Stored segmentation and meter depths retain their values and array encodings.
    """
    frame = SegmentationDepthFrame(
        segmentation=np.array([[1, -1], [2, 1]], dtype=np.int32),
        depth=np.array([[1.5, -1.0], [2.5, 1.0]], dtype=np.float32),
    )
    stored = to_dao(frame)
    with in_memory_session_maker() as session:
        session.add(stored)
        session.commit()
    with in_memory_session_maker() as session:
        restored = session.scalar(select(type(stored))).from_dao()
    np.testing.assert_array_equal(restored.segmentation, frame.segmentation)
    np.testing.assert_array_equal(restored.depth, frame.depth)
    assert restored.segmentation.dtype == frame.segmentation.dtype
    assert restored.depth.dtype == frame.depth.dtype


# %% Body color persistence
@pytest.mark.parametrize("frame_type", [RenderedColorFrame, RenderedRGBDFrame])
def test_rendered_frame_retains_body_color_assignments(
    in_memory_session_maker: sessionmaker[Session],
    frame_type: type[RenderedColorFrame] | type[RenderedRGBDFrame],
) -> None:
    """
    Both frame types retain structured colors and all associated body references.
    """
    body_colors = [
        BodyColorAssignment(
            color=Color(hue=0.75, saturation=0.0, value=0.5, opacity=0.25),
            bodies=[
                Body(name=PrefixedName("first")),
                Body(name=PrefixedName("second")),
            ],
        ),
        BodyColorAssignment(
            color=Color.BLUE(), bodies=[Body(name=PrefixedName("third"))]
        ),
    ]
    color_image = np.zeros((2, 2, 3), dtype=np.uint8)
    frame = (
        RenderedColorFrame(color_image=color_image, body_colors=body_colors)
        if frame_type is RenderedColorFrame
        else RenderedRGBDFrame(
            color_bgr=color_image,
            depth_in_millimeters=np.zeros((2, 2), dtype=np.uint16),
            segmentation=np.full((2, 2), -1, dtype=np.int32),
            body_colors=body_colors,
            camera_model=PinholeCameraModel.from_field_of_view(
                resolution=CameraResolution(width=2, height=2),
                field_of_view=FieldOfView(),
            ),
        )
    )
    stored = to_dao(frame)
    with in_memory_session_maker() as session:
        session.add(stored)
        session.commit()
    with in_memory_session_maker() as session:
        restored = session.scalar(select(type(stored))).from_dao()

    assert len(restored.body_colors) == len(body_colors)
    for actual, expected in zip(restored.body_colors, body_colors, strict=True):
        assert actual.color.hue == expected.color.hue
        assert actual.color.saturation == expected.color.saturation
        assert actual.color.value == expected.color.value
        assert actual.color.opacity == expected.color.opacity
        assert [body.id for body in actual.bodies] == [
            body.id for body in expected.bodies
        ]
