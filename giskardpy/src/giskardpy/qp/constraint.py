"""
Constraint description classes for the QP controller (:class:`GiskardConstraint` and
subclasses).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing_extensions import TYPE_CHECKING

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.qp.bound_cap import BoundCap, ScalarBoundCap
from krrood.symbolic_math.symbolic_math import Scalar

if TYPE_CHECKING:
    from giskardpy.qp.enforcement_strategy import EnforcementStrategy


LargeNumber = 1e4


@dataclass(kw_only=True)
class GiskardConstraint(ABC):
    """
    Defines a (slack-relaxed) constraint on expression for a quadratic program.
    """

    name: str
    """
    Human readable name of the constraint, used for debugging.
    """

    expression: Scalar
    """
    The expression that is being constrained.
    """

    quadratic_weight: sm.ScalarData
    """
    The quadratic weight of this constraint.

    Describes how expensive it is to violate.
    """

    linear_weight: sm.ScalarData = field(default=0)
    """
    The linear weight of this constraint.

    It is only here for completeness, no use case has been found for it yet.
    """

    velocity_unit: sm.ScalarData
    """
    The speed at which the expression typically changes in its space, such as the
    velocity limit of a joint.

    It scales the weight so that constraints of different units become comparable, and
    does not depend on the speed the constraint drives at.
    """

    cruise_velocity: sm.ScalarData | None = None
    """
    The velocity a constraint on the expression's position drives at, from which the
    change it may demand within the control horizon is derived.

    None for a constraint on the expression's velocity, which demands no change of
    position.
    """

    enforcement_strategy: type[EnforcementStrategy]
    """
    The strategy used to enforce this constraint within the QP.
    """

    lower_slack_limit: sm.ScalarData = field(default=-LargeNumber)
    """
    How far the constraint may be violated below its bound.
    """

    upper_slack_limit: sm.ScalarData = field(default=LargeNumber)
    """
    How far the constraint may be violated above its bound.
    """

    @abstractmethod
    def kept_fraction_of_demand(self, reachable_change: Scalar) -> Scalar:
        """
        :param reachable_change: The largest change of the expression that may be
            demanded within the control horizon.
        :return: The fraction of the motion this constraint demands that is kept when its
            bounds are capped to the reachable change, 1 for a demand within reach or
            for a constraint that demands no motion.
        """


@dataclass
class GiskardEqualityConstraint(GiskardConstraint):
    """
    A constraint that drives the expression to a single target value.
    """

    bound: Scalar
    """
    The target value the expression should reach.
    """

    bound_cap: BoundCap = field(default_factory=ScalarBoundCap)
    """
    How the bound is limited to what may be demanded within the control horizon.
    """

    def kept_fraction_of_demand(self, reachable_change: Scalar) -> Scalar:
        return self.bound_cap.kept_fraction(self.bound, reachable_change)


@dataclass
class GiskardInequalityConstraint(GiskardConstraint):
    """
    A constraint that keeps the expression between a lower and upper bound.
    """

    lower_bound: Scalar
    """
    The lowest allowed value of the expression.
    """

    upper_bound: Scalar
    """
    The highest allowed value of the expression.
    """

    def kept_fraction_of_demand(self, reachable_change: Scalar) -> Scalar:
        # a positive lower or a negative upper bound demands motion
        demanded_change = sm.max(
            sm.max(sm.Scalar(self.lower_bound), -sm.Scalar(self.upper_bound)), 0
        )
        return ScalarBoundCap().kept_fraction(demanded_change, reachable_change)
