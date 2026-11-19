"""
The straight line segments of 3D space and the lines they lie on.
"""

from __future__ import annotations

import numpy as np

from krrood_spatial_math.spatial_types import (
    Point3Dimensional,
    Line3Dimensional,
    LineSegment3Dimensional,
)

# %% the direction a segment runs in


def test_a_segment_runs_from_its_start_to_its_end_as_a_unit_vector():
    segment = LineSegment3Dimensional(
        start=Point3Dimensional(1.0, 2.0, 3.0),
        end=Point3Dimensional(1.0, 5.0, 7.0),
    )

    np.testing.assert_allclose(segment.direction.to_np()[:3], [0.0, 0.6, 0.8])


# %% the line a segment lies on


def test_a_segment_lies_on_a_3d_line_from_its_start_to_its_end():
    """
    A 3D segment's line is a :class:`Line3Dimensional` whose position 0.5 is the
    segment's midpoint.
    """
    segment = LineSegment3Dimensional(
        start=Point3Dimensional(1.0, 2.0, 3.0),
        end=Point3Dimensional(1.0, 5.0, 7.0),
    )

    line = segment.line

    assert isinstance(line, Line3Dimensional)
    np.testing.assert_allclose(
        line.point_at_position(position_along_line=0.5).to_np()[:3], [1.0, 3.5, 5.0]
    )
