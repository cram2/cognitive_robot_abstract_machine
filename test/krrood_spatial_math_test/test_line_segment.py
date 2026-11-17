from __future__ import annotations

import numpy as np

from krrood_spatial_math.spatial_types import LineSegment3, Point3

# %% the direction a segment runs in


def test_a_segment_runs_from_its_start_to_its_end_as_a_unit_vector():
    segment = LineSegment3(start=Point3(1.0, 2.0, 3.0), end=Point3(1.0, 5.0, 7.0))

    np.testing.assert_allclose(segment.direction.to_np()[:3], [0.0, 0.6, 0.8])
