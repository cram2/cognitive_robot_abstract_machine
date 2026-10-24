"""
Persistence of ray-traced camera frame data.
"""

from __future__ import annotations

# %% Imports
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from krrood.ormatic.data_access_objects.helper import to_dao
from semantic_digital_twin.datastructures.camera_rendering import (
    SegmentationDepthFrame,
)

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
