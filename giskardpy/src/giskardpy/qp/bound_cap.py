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
