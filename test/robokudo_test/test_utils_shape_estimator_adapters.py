import numpy as np

from robokudo.utils.shape_estimator_adapters import (
    CUBOID_FIT_ADAPTER,
    CYLINDER_FIT_ADAPTER,
    SPHERE_FIT_ADAPTER,
    CuboidFitSummary,
    CylinderFitSummary,
    SphereFitSummary,
)
from robokudo.utils.shape_fitting import CuboidFit, CylinderFit, SphereFit


class TestUtilsShapeEstimatorAdapters:
    def test_sphere_summary_returns_dataclass_with_compact_text(self) -> None:
        fit = SphereFit(
            center=np.zeros(3, dtype=np.float64),
            radius=0.12345,
            inlier_indices=np.array([0, 1], dtype=np.int64),
            inlier_ratio=0.45678,
            root_mean_square_error=0.012345,
            score=0.98765,
        )

        summary = SPHERE_FIT_ADAPTER.summary(fit)

        assert isinstance(summary, SphereFitSummary)
        assert summary.radius == 0.12345
        assert (
            str(summary)
            == "Sphere(score=0.988, inlier_ratio=0.457, rmse=0.0123, radius=0.123)"
        )

    def test_cylinder_summary_returns_dataclass_with_compact_text(self) -> None:
        fit = CylinderFit(
            axis_center=np.zeros(3, dtype=np.float64),
            axis_direction=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            radius=0.12345,
            height=1.23456,
            inlier_indices=np.array([0, 1], dtype=np.int64),
            inlier_ratio=0.45678,
            root_mean_square_error=0.012345,
            score=0.98765,
        )

        summary = CYLINDER_FIT_ADAPTER.summary(fit)

        assert isinstance(summary, CylinderFitSummary)
        assert summary.height == 1.23456
        assert (
            str(summary)
            == "Cylinder(score=0.988, inlier_ratio=0.457, rmse=0.0123, radius=0.123, height=1.235)"
        )

    def test_cuboid_summary_returns_dataclass_with_compact_text(self) -> None:
        fit = CuboidFit(
            center=np.zeros(3, dtype=np.float64),
            rotation_matrix=np.eye(3, dtype=np.float64),
            extents=np.array([0.12345, 1.23456, 2.34567], dtype=np.float64),
            inlier_indices=np.array([0, 1], dtype=np.int64),
            inlier_ratio=0.45678,
            root_mean_square_error=0.012345,
            score=0.98765,
        )

        summary = CUBOID_FIT_ADAPTER.summary(fit)

        assert isinstance(summary, CuboidFitSummary)
        assert summary.extents == [0.123, 1.235, 2.346]
        assert (
            str(summary)
            == "Cuboid(score=0.988, inlier_ratio=0.457, rmse=0.0123, extents=[0.123, 1.235, 2.346])"
        )
