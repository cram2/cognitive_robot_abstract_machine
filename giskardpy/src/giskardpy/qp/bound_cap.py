"""
Rules for capping what an equality constraint may demand within the control horizon
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
    Limits the bound of an equality constraint to the largest change that may be
    demanded within the control horizon, so that a far away goal does not ask for more
    motion than is reachable.
    """

    @abstractmethod
    def cap_bound(self, bound: Scalar, reachable_change: Scalar) -> Scalar:
        """
        :param bound: The bound of one constraint.
        :param reachable_change: The largest change of the constraint's expression that
            may be demanded within the control horizon.
        :return: The bound, limited to the reachable change.
        """


@dataclass
class ScalarBoundCap(BoundCap):
    """
    Caps a bound on its own, independently of any other constraint.
    """

    def cap_bound(self, bound: Scalar, reachable_change: Scalar) -> Scalar:
        return sm.limit(bound, -reachable_change, reachable_change)


@dataclass
class VectorBoundCap(BoundCap):
    """
    Caps the bounds of the constraints that together describe one vector, such as the
    three axes of a point goal, by the length of that vector.

    Every bound is scaled by the same factor, so the capped vector keeps its direction.
    """

    vector_bounds: sm.Vector
    """
    The bounds of all constraints that make up the vector.
    """

    def cap_bound(self, bound: Scalar, reachable_change: Scalar) -> Scalar:
        scale = sm.min(
            1, reachable_change.safe_division(self.vector_bounds.norm(), if_nan=1)
        )
        return bound * scale


@dataclass
class RotationBoundCap(BoundCap):
    """
    Caps the three constraints of a rotation goal by the rotation angle, so the angle
    demanded within the control horizon does not depend on how far away the goal is.
    """

    quaternion_error_bounds: sm.Vector
    """
    The bounds of the three constraints, the vector part of the quaternion error.
    """

    def cap_bound(self, bound: Scalar, reachable_change: Scalar) -> Scalar:
        sine_of_half_angle = self.quaternion_error_bounds.norm()
        cosine_of_half_angle = sm.max(0, 1 - sine_of_half_angle**2) ** 0.5
        angle = 2 * sm.atan2(sine_of_half_angle, cosine_of_half_angle)
        capped_angle = sm.min(angle, 2 * reachable_change)
        capped_sine_change = sine_of_half_angle - sm.sin((angle - capped_angle) / 2)
        scale = capped_sine_change.safe_division(sine_of_half_angle, if_nan=1)
        return bound * scale
