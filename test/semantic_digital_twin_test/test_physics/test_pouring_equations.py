"""
Tests of the analytic fill equations of the pouring domain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import krrood.symbolic_math.symbolic_math as sm
import pytest

from semantic_digital_twin.physics.equations.pouring_equations import (
    PouringEquation,
)
from semantic_digital_twin.physics.exceptions import (
    NonPositiveContainerGeometryError,
)

# %% the cup the equations are evaluated for


@dataclass(kw_only=True)
class PouringCup:
    """
    A rectangular cup and the operating point the equation tests evaluate it at.
    """

    container_height: float = 0.2
    """
    Height of the cup, in metres.
    """

    container_width: float = 0.08
    """
    Width of the cup, in metres.
    """

    repose_angle: float = 0.5
    """
    Angle a heap of the contents holds before it slumps, in radians.
    """

    tilt: float = 0.5
    """
    Tilt at which the head is evaluated, in radians.
    """

    fill: float = 0.8
    """
    Fill level at which the head is evaluated.
    """

    def equation(self, *, repose_angle: float = 0.0) -> PouringEquation:
        """
        :return: The drain of this cup.
        """
        return PouringEquation(
            container_height=self.container_height,
            container_width=self.container_width,
            repose_angle=repose_angle,
        )

    def expected_head(
        self, *, lip_offset: float, tilt: float, fill: float, repose_angle: float = 0.0
    ) -> float:
        """
        :return: Closed-form head of the tilted surface above a lip at ``lip_offset``.
        """
        dry_height = self.container_height * (1 - fill)
        lip_distance = math.hypot(dry_height, lip_offset)
        onset = math.atan2(dry_height, lip_offset) + repose_angle
        return max(0.0, lip_distance * math.sin(tilt - onset))


@pytest.fixture
def cup() -> PouringCup:
    """
    The cup the equation tests pour from.
    """
    return PouringCup()


def _head(*, equation: PouringEquation, tilt: float, fill: float) -> float:
    """
    :return: The equation's numeric head at the given tilt and fill.
    """
    return equation.head_above_lip(
        tilt=sm.Scalar(tilt), fill=sm.Scalar(fill)
    ).evaluate()[0]


def _drain(*, equation: PouringEquation, tilt: float, fill: float) -> float:
    """
    :return: The equation's numeric fill rate at the given tilt and fill.
    """
    return equation.fill_velocity(
        tilt=sm.Scalar(tilt), fill=sm.Scalar(fill)
    ).evaluate()[0]


def _exit_speed(*, equation: PouringEquation, tilt: float, fill: float) -> float:
    """
    :return: The equation's numeric exit speed at the given tilt and fill.
    """
    return equation.exit_speed(tilt=sm.Scalar(tilt), fill=sm.Scalar(fill)).evaluate()[0]


# %% container geometry guard


@pytest.mark.parametrize(
    "container_height, container_width", [(0.0, 0.08), (0.1, -0.08)]
)
def test_non_positive_geometry_is_rejected(
    container_height: float, container_width: float
) -> None:
    """
    A drain of no or negative extent is refused instead of producing inf/NaN drains.
    """
    with pytest.raises(NonPositiveContainerGeometryError):
        PouringEquation(
            container_height=container_height, container_width=container_width
        )


# %% head above the lip


def test_the_head_is_zero_up_to_the_onset_and_the_closed_form_above_it(
    cup: PouringCup,
) -> None:
    """
    No head up to the onset tilt, which the repose angle delays, and the closed-form
    head above it.
    """
    equation = cup.equation(repose_angle=cup.repose_angle)
    onset = float(equation.onset_tilt(fill=cup.fill).evaluate()[0])

    assert _head(equation=equation, tilt=onset - 0.05, fill=cup.fill) == 0.0
    assert _head(equation=equation, tilt=onset, fill=cup.fill) == pytest.approx(
        0.0, abs=1e-12
    )
    assert _head(equation=equation, tilt=onset + 0.05, fill=cup.fill) == pytest.approx(
        cup.expected_head(
            lip_offset=cup.container_width / 2,
            tilt=onset + 0.05,
            fill=cup.fill,
            repose_angle=cup.repose_angle,
        )
    )


# %% drain rate


def test_the_drain_follows_the_closed_form(cup: PouringCup) -> None:
    """
    ``fill_velocity = −outflow_rate_constant · head_above_lip / container_height``.
    """
    equation = cup.equation()
    tilt = cup.tilt + 0.5
    head = cup.expected_head(lip_offset=equation.lip_offset, tilt=tilt, fill=cup.fill)

    assert _drain(equation=equation, tilt=tilt, fill=cup.fill) == pytest.approx(
        -equation.outflow_rate_constant * head / cup.container_height
    )


# %% exit speed


def test_the_exit_speed_is_torricelli_of_the_head(cup: PouringCup) -> None:
    """
    ``C_d · √(2 g h)`` for the head above the lip.
    """
    equation = cup.equation()
    tilt = cup.tilt + 0.5
    head = cup.expected_head(lip_offset=equation.lip_offset, tilt=tilt, fill=cup.fill)

    assert _exit_speed(equation=equation, tilt=tilt, fill=cup.fill) == pytest.approx(
        equation.discharge_coefficient
        * math.sqrt(2 * PouringEquation.standard_gravity * head)
    )


def test_the_exit_speed_of_no_head_is_that_of_the_minimum_head(
    cup: PouringCup,
) -> None:
    """
    Below the onset the speed is computed from the minimum head, so its gradient stays
    finite.
    """
    equation = cup.equation()

    assert _exit_speed(equation=equation, tilt=0.0, fill=cup.fill) == pytest.approx(
        equation.discharge_coefficient
        * math.sqrt(2 * PouringEquation.standard_gravity * equation.minimum_head)
    )
