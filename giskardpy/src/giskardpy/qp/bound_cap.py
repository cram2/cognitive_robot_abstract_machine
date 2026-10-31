"""
Rules for capping what an equality constraint may demand from one plan
(:class:`BoundCap` and subclasses).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import Scalar


@dataclass
class BoundCap(ABC):
    """
    Limits the bound of an equality constraint to the largest change one plan may
    demand, so that a far away goal does not ask for more motion than is reachable.
    """

    @abstractmethod
    def capped(self, bound: Scalar, reach: Scalar) -> Scalar:
        """
        :param bound: The bound of one constraint.
        :param reach: The largest change of the constraint's expression one plan may
            demand.
        :return: The bound, limited to what one plan may demand.
        """


@dataclass
class ScalarBoundCap(BoundCap):
    """
    Caps a bound on its own, independently of any other constraint.
    """

    def capped(self, bound: Scalar, reach: Scalar) -> Scalar:
        return sm.limit(bound, -reach, reach)


@dataclass
class VectorBoundCap(BoundCap):
    """
    Caps the bounds of the constraints that together describe one vector, such as the
    three axes of a point goal, by the length of that vector.

    Every bound is scaled by the same factor, so the capped vector keeps its direction.
    """

    bounds: sm.Vector
    """
    The bounds of all constraints that make up the vector.
    """

    def capped(self, bound: Scalar, reach: Scalar) -> Scalar:
        scale = sm.min(1, reach.safe_division(self.bounds.norm(), if_nan=1))
        return bound * scale


@dataclass
class RotationBoundCap(BoundCap):
    """
    Caps the bounds of the three constraints of a rotation goal, which demand a change
    of the vector part of the quaternion error, sin(angle / 2) * axis.

    The cap is applied to the rotation angle rather than to the vector part, so that the
    angle one plan may demand does not depend on how far away the goal is. The reachable
    change is read in units of the half angle, which the vector part changes by close to
    the goal, so close to the goal the cap equals that of a single row.
    """

    bounds: sm.Vector
    """
    The bounds of the three constraints, the vector part of the quaternion error.
    """

    def capped(self, bound: Scalar, reach: Scalar) -> Scalar:
        sine_of_half_angle = self.bounds.norm()
        cosine_of_half_angle = sm.max(0, 1 - sine_of_half_angle**2) ** 0.5
        angle = 2 * sm.atan2(sine_of_half_angle, cosine_of_half_angle)
        capped_angle = sm.min(angle, 2 * reach)
        capped_sine_change = sine_of_half_angle - sm.sin((angle - capped_angle) / 2)
        scale = capped_sine_change.safe_division(sine_of_half_angle, if_nan=1)
        return bound * scale
