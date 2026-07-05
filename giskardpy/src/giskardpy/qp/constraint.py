"""
Constraint description classes for the QP controller (:class:`GiskardConstraint` and subclasses).
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import krrood.symbolic_math.symbolic_math as sm
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
    The quadratic weight of this constraint. Describes how expensive it is to violate.
    """

    linear_weight: sm.ScalarData = field(default=0)
    """
    The linear weight of this constraint. It is only here for completeness, no use case has been found for it yet.
    """

    normalization_factor: float
    """
    This value is important to make constraints with different units comparable.
    The meaning depends on derivative.
    If the derivative is position, the normalization factor is rough velocity with which the expression can change.
    For example:
        - If you have a joint position constraint, the normalization factor should be the joint velocity limit.
        - If you have a cartesian position constraint, the normalization factor should be the cartesian velocity limit.
    In practice, use joint limits from the URDF for joint space constraints and define two values for cartesian constraints:
        - a m/s limit for translation
        - a rad/s value for rotation
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


@dataclass
class GiskardEqualityConstraint(GiskardConstraint):
    """
    A constraint that drives the expression to a single target value.
    """

    bound: Scalar
    """
    The target value the expression should reach.
    """


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
