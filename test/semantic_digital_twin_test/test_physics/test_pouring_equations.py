"""
Geometry-validation tests for the analytic pouring-domain fill equations.
"""

import pytest

from semantic_digital_twin.exceptions import NonPositiveContainerGeometryError
from semantic_digital_twin.physics.equations.pouring_equations import (
    ArticulatedPouringEquation,
    InflowEquation,
)

# %% container geometry guard


class TestContainerGeometryGuard:
    """
    Non-positive container dimensions must raise instead of producing inf/NaN drains.
    """

    @pytest.mark.parametrize(
        "container_height, container_width",
        [(0.0, 0.08), (0.1, 0.0), (-0.1, 0.08), (0.1, -0.08)],
    )
    def test_pouring_equation_rejects_non_positive_geometry(
        self, container_height: float, container_width: float
    ):
        with pytest.raises(NonPositiveContainerGeometryError):
            ArticulatedPouringEquation(
                container_height=container_height, container_width=container_width
            )

    @pytest.mark.parametrize(
        "container_height, container_width",
        [(0.0, 0.08), (0.1, 0.0)],
    )
    def test_inflow_equation_rejects_non_positive_geometry(
        self, container_height: float, container_width: float
    ):
        with pytest.raises(NonPositiveContainerGeometryError):
            InflowEquation(
                container_height=container_height, container_width=container_width
            )

    def test_positive_geometry_is_accepted(self):
        equation = ArticulatedPouringEquation(
            container_height=0.1, container_width=0.08
        )
        assert equation.container_height == 0.1
        assert equation.container_width == 0.08
