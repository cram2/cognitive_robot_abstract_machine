from collections import namedtuple
from copy import copy
from dataclasses import dataclass
from typing import Optional

import semantic_digital_twin.spatial_types.spatial_types as cas
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.derivatives import Derivatives

DebugConstraint = namedtuple("debug", ["expr"])


@dataclass
class Constraint:
    constraint_name: str
    parent_task_name: str

    @property
    def name(self) -> str:
        return str(PrefixedName(self.constraint_name, self.parent_task_name))


@dataclass
class InequalityConstraint(Constraint):
    expression: cas.SymbolicScalar

    velocity_limit: float
    quadratic_weight: cas.ScalarData

    lower_error: cas.ScalarData = -1e4
    upper_error: cas.ScalarData = 1e4

    lower_slack_limit: cas.ScalarData = -1e4
    upper_slack_limit: cas.ScalarData = 1e4

    linear_weight: cas.ScalarData = 0

    def __copy__(self):
        return InequalityConstraint(
            constraint_name=self.constraint_name,
            parent_task_name=self.parent_task_name,
            expression=copy(self.expression),
            lower_error=copy(self.lower_error),
            upper_error=copy(self.upper_error),
            velocity_limit=self.velocity_limit,
            quadratic_weight=self.quadratic_weight,
            linear_weight=self.linear_weight,
            lower_slack_limit=copy(self.lower_slack_limit),
            upper_slack_limit=copy(self.upper_slack_limit),
        )

    def normalized_weight(self, control_horizon: int) -> cas.Expression:
        weight_normalized = self.quadratic_weight * (
            1 / (self.velocity_limit**2 * control_horizon)
        )
        return weight_normalized

    def capped_lower_error(self, dt: float, control_horizon: int) -> cas.Expression:
        return cas.limit(
            self.lower_error,
            -self.velocity_limit * dt * control_horizon,
            self.velocity_limit * dt * control_horizon,
        )

    def capped_upper_error(self, dt: float, control_horizon: int) -> cas.Expression:
        return cas.limit(
            self.upper_error,
            -self.velocity_limit * dt * control_horizon,
            self.velocity_limit * dt * control_horizon,
        )


@dataclass
class EqualityConstraint(Constraint):
    expression: cas.SymbolicScalar

    bound: cas.ScalarData
    velocity_limit: cas.ScalarData
    quadratic_weight: cas.ScalarData

    lower_slack_limit: cas.ScalarData = -1e4
    upper_slack_limit: cas.ScalarData = 1e4

    linear_weight: cas.ScalarData = 0

    def __copy__(self):
        return EqualityConstraint(
            constraint_name=self.constraint_name,
            parent_task_name=self.parent_task_name,
            expression=copy(self.expression),
            bound=copy(self.bound),
            velocity_limit=self.velocity_limit,
            quadratic_weight=self.quadratic_weight,
            linear_weight=self.linear_weight,
            lower_slack_limit=copy(self.lower_slack_limit),
            upper_slack_limit=copy(self.upper_slack_limit),
        )

    def normalized_weight(self, control_horizon: int) -> cas.Expression:
        weight_normalized = self.quadratic_weight * (
            1 / (self.velocity_limit**2 * control_horizon)
        )
        return weight_normalized

    def capped_bound(self, dt: float, control_horizon: int) -> cas.Expression:
        return cas.limit(
            self.bound,
            -self.velocity_limit * dt * control_horizon,
            self.velocity_limit * dt * control_horizon,
        )


@dataclass
class DerivativeInequalityConstraint(Constraint):
    derivative: Derivatives
    expression: cas.Expression
    lower_limit: cas.ScalarData
    upper_limit: cas.ScalarData
    quadratic_weight: cas.ScalarData
    normalization_factor: Optional[cas.ScalarData]
    lower_slack_limit: cas.ScalarData
    upper_slack_limit: cas.ScalarData
    linear_weight: cas.ScalarData = None

    def __copy__(self):
        return DerivativeInequalityConstraint(
            constraint_name=self.constraint_name,
            parent_task_name=self.parent_task_name,
            derivative=self.derivative,
            expression=copy(self.expression),
            lower_limit=copy(self.lower_limit),
            upper_limit=copy(self.upper_limit),
            quadratic_weight=self.quadratic_weight,
            normalization_factor=self.normalization_factor,
            lower_slack_limit=copy(self.lower_slack_limit),
            upper_slack_limit=copy(self.upper_slack_limit),
            linear_weight=self.linear_weight,
        )

    def normalized_weight(self, t) -> float:
        return self.quadratic_weight * (1 / self.normalization_factor) ** 2


@dataclass
class DerivativeEqualityConstraint(Constraint):
    derivative: Derivatives
    expression: cas.Expression
    bound: cas.ScalarData
    quadratic_weight: cas.ScalarData
    normalization_factor: Optional[cas.ScalarData]
    lower_slack_limit: cas.ScalarData
    upper_slack_limit: cas.ScalarData
    linear_weight: cas.ScalarData = None

    def __copy__(self):
        return DerivativeEqualityConstraint(
            constraint_name=self.constraint_name,
            parent_task_name=self.parent_task_name,
            derivative=self.derivative,
            expression=copy(self.expression),
            bound=copy(self.bound),
            quadratic_weight=self.quadratic_weight,
            normalization_factor=self.normalization_factor,
            lower_slack_limit=copy(self.lower_slack_limit),
            upper_slack_limit=copy(self.upper_slack_limit),
            linear_weight=self.linear_weight,
        )

    def normalized_weight(self, t) -> float:
        return self.quadratic_weight * (1 / self.normalization_factor) ** 2
