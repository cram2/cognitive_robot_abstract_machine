"""
Tests of the analytic fill equations of the pouring domain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import krrood.symbolic_math.symbolic_math as sm
import pytest
from semantic_digital_twin.physics.equations.pouring_equations import (
    PouringEquation,
)
from semantic_digital_twin.physics.equations.projectile_motion import ProjectileMotion
from semantic_digital_twin.world_description.geometry import Scale

# %% the cup the equations are evaluated for


@dataclass(kw_only=True)
class PouringCup:
    """
    A rectangular cup and the operating point the equation tests evaluate it at.
    """

    container_scale: Scale = field(default_factory=lambda: Scale(0.08, 0.08, 0.2))
    """
    Inner extents of the cup, in metres.
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
            container_scale=self.container_scale,
            repose_angle=repose_angle,
        )

    def expected_head(
        self, *, lip_offset: float, tilt: float, fill: float, repose_angle: float = 0.0
    ) -> float:
        """
        :return: Closed-form head of the tilted surface above a lip at ``lip_offset``.
        """
        dry_height = self.container_scale.z * (1 - fill)
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
            lip_offset=cup.container_scale.y / 2,
            tilt=onset + 0.05,
            fill=cup.fill,
            repose_angle=cup.repose_angle,
        )
    )


# %% drain rate


def test_the_drain_follows_the_closed_form(cup: PouringCup) -> None:
    """
    The fill rate is the head above the lip, scaled by the outflow rate constant and
    normalized by the cup's height.
    """
    equation = cup.equation()
    tilt = cup.tilt + 0.5
    head = cup.expected_head(lip_offset=equation.lip_offset, tilt=tilt, fill=cup.fill)

    assert _drain(equation=equation, tilt=tilt, fill=cup.fill) == pytest.approx(
        -equation.outflow_rate_constant * head / cup.container_scale.z
    )


# %% exit speed


def test_the_exit_speed_is_torricelli_of_the_head(cup: PouringCup) -> None:
    """
    The discharge coefficient times the speed of a fall through the head above the lip.
    """
    equation = cup.equation()
    tilt = cup.tilt + 0.5
    head = cup.expected_head(lip_offset=equation.lip_offset, tilt=tilt, fill=cup.fill)

    assert _exit_speed(equation=equation, tilt=tilt, fill=cup.fill) == pytest.approx(
        equation.discharge_coefficient
        * math.sqrt(2 * ProjectileMotion.standard_gravity * head)
    )
