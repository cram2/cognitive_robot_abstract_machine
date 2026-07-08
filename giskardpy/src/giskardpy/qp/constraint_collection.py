from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from typing_extensions import Optional, TYPE_CHECKING

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.data_types.exceptions import (
    DuplicateNameException,
)
from giskardpy.motion_statechart.data_types import LifeCycleValues
from giskardpy.motion_statechart.exceptions import (
    InvalidConstraintExpressionShapeError,
)
from giskardpy.qp.constraint import (
    GiskardConstraint,
    GiskardEqualityConstraint,
    GiskardInequalityConstraint,
    LargeNumber,
)
from giskardpy.qp.enforcement_strategy import (
    EnforcementStrategy,
    IntegralStrategy,
    VelocityStrategy,
)

if TYPE_CHECKING:
    from giskardpy.motion_statechart.graph_node import MotionStatechartNode


@dataclass
class ConstraintCollection:
    """
    Holds the equality and inequality constraints of a motion task and groups them by the
    enforcement strategy that turns them into the rows of the final quadratic program.
    """

    _constraints: list[GiskardConstraint] = field(default_factory=list, init=False)
    """
    The constraints collected so far, in insertion order.
    """

    def get_equality_constraint_blocks(
        self,
    ) -> dict[type[EnforcementStrategy], list[GiskardEqualityConstraint]]:
        """
        Groups the equality constraints by their enforcement strategy.
        """
        result = defaultdict(list)
        for c in self._constraints:
            if isinstance(c, GiskardEqualityConstraint):
                result[c.enforcement_strategy].append(c)
        return result

    def get_inequality_constraint_blocks(
        self,
    ) -> dict[type[EnforcementStrategy], list[GiskardInequalityConstraint]]:
        """
        Groups the inequality constraints by their enforcement strategy.
        """
        result = defaultdict(list)
        for c in self._constraints:
            if isinstance(c, GiskardInequalityConstraint):
                result[c.enforcement_strategy].append(c)
        return result

    @property
    def equality_constraints(self) -> list[GiskardEqualityConstraint]:
        """
        All collected equality constraints.
        """
        return [
            c for c in self._constraints if isinstance(c, GiskardEqualityConstraint)
        ]

    @property
    def inequality_constraints(self) -> list[GiskardInequalityConstraint]:
        """
        All collected inequality constraints.
        """
        return [
            c for c in self._constraints if isinstance(c, GiskardInequalityConstraint)
        ]

    def merge(self, name_prefix: str, other: ConstraintCollection) -> None:
        """
        Adds the constraints of another collection, prefixing their names to keep them unique.
        """
        for constraint in other._constraints:
            constraint.name = f"{name_prefix}/{constraint.name}"
        self._constraints.extend(other._constraints)
        self._are_names_unique()

    def add_constraint(self, constraint: GiskardConstraint) -> None:
        """
        Appends a constraint, assigning it an index-based name if it has none.

        :raises DuplicateNameException: if a constraint with the same name already exists.
        """
        constraint.name = constraint.name or f"{len(self._constraints)}"
        existing_names = {c.name for c in self._constraints}
        if constraint.name in existing_names:
            raise DuplicateNameException(name=constraint.name)
        self._constraints.append(constraint)

    def _are_names_unique(self) -> None:
        """
        Verifies that all collected constraint names are unique.

        :raises DuplicateNameException: if two constraints share a name.
        """
        names = set()
        for c in self._constraints:
            if c.name in names:
                raise DuplicateNameException(name=c.name)
            names.add(c.name)

    def get_all_float_variable_names(self) -> set[str]:
        """
        The names of all free variables appearing in any constraint expression.
        """
        return {
            v.name for c in self._constraints for v in c.expression.free_variables()
        }

    def link_to_motion_statechart_node(self, node: MotionStatechartNode) -> None:
        """
        Scales every constraint weight so it is only active while the given node is running.
        """
        for constraint in self._constraints:
            is_running = sm.if_eq(
                node.life_cycle_variable,
                LifeCycleValues.RUNNING,
                if_result=sm.Scalar(1),
                else_result=sm.Scalar(0),
            )
            constraint.quadratic_weight *= is_running

    def add_equality_constraint(
        self,
        task_expression: sm.SymbolicScalar,
        equality_bound: sm.ScalarData,
        quadratic_weight: sm.ScalarData,
        reference_velocity: sm.ScalarData,
        name: Optional[str] = None,
        lower_slack_limit: sm.ScalarData = -LargeNumber,
        upper_slack_limit: sm.ScalarData = LargeNumber,
    ) -> None:
        """
        Add a task constraint to the motion problem. This should be used for most constraints.
        It will not strictly stick to the reference velocity, but requires only a single constraint in the final
        optimization problem and is therefore faster.
        :param reference_velocity: used by Giskard to limit the error and normalize the weight, will not be strictly
                                    enforced.
        :param task_expression: defines the task function
        :param equality_bound: goal for the derivative of task_expression
        :param quadratic_weight: how expensive it is to violate this constraint
        :param name: give this constraint a name, required if you add more than one in the same goal
        :param lower_slack_limit: how much the lower error can be violated, don't use unless you know what you are doing
        :param upper_slack_limit: how much the upper error can be violated, don't use unless you know what you are doing
        """
        if task_expression.shape != (1, 1):
            raise InvalidConstraintExpressionShapeError(list(task_expression.shape))

        lower_slack_limit = (
            lower_slack_limit if lower_slack_limit is not None else -float("inf")
        )
        upper_slack_limit = (
            upper_slack_limit if upper_slack_limit is not None else float("inf")
        )
        constraint = GiskardEqualityConstraint(
            name=name,
            expression=task_expression,
            bound=equality_bound,
            normalization_factor=reference_velocity,
            quadratic_weight=quadratic_weight,
            lower_slack_limit=lower_slack_limit,
            upper_slack_limit=upper_slack_limit,
            linear_weight=0,
            enforcement_strategy=IntegralStrategy,
        )
        self.add_constraint(constraint)

    def add_inequality_constraint(
        self,
        reference_velocity: sm.ScalarData,
        lower_error: sm.ScalarData,
        upper_error: sm.ScalarData,
        quadratic_weight: sm.ScalarData,
        task_expression: sm.SymbolicScalar,
        name: Optional[str] = None,
        linear_weight: sm.ScalarData = 0,
        lower_slack_limit: sm.ScalarData = -LargeNumber,
        upper_slack_limit: sm.ScalarData = LargeNumber,
    ) -> None:
        """
        Add a task constraint to the motion problem. This should be used for most constraints.
        It will not strictly stick to the reference velocity, but requires only a single constraint in the final
        optimization problem and is therefore faster.
        :param reference_velocity: used by Giskard to limit the error and normalize the weight, will not be strictly
                                    enforced.
        :param lower_error: lower bound for the error of expression
        :param upper_error: upper bound for the error of expression
        :param quadratic_weight:
        :param task_expression: defines the task function
        :param name: give this constraint a name, required if you add more than one in the same goal
        :param lower_slack_limit: how much the lower error can be violated, don't use unless you know what you are doing
        :param upper_slack_limit: how much the upper error can be violated, don't use unless you know what you are doing
        """
        if task_expression.shape != (1, 1):
            raise InvalidConstraintExpressionShapeError(list(task_expression.shape))
        lower_slack_limit = (
            lower_slack_limit if lower_slack_limit is not None else -float("inf")
        )
        upper_slack_limit = (
            upper_slack_limit if upper_slack_limit is not None else float("inf")
        )
        constraint = GiskardInequalityConstraint(
            name=name,
            expression=task_expression,
            normalization_factor=reference_velocity,
            quadratic_weight=quadratic_weight,
            lower_slack_limit=lower_slack_limit,
            upper_slack_limit=upper_slack_limit,
            linear_weight=linear_weight,
            enforcement_strategy=IntegralStrategy,
            lower_bound=lower_error,
            upper_bound=upper_error,
        )
        self.add_constraint(constraint)

    def add_velocity_constraint(
        self,
        lower_velocity_limit: sm.ScalarData,
        upper_velocity_limit: sm.ScalarData,
        quadratic_weight: sm.ScalarData,
        task_expression: sm.SymbolicScalar,
        velocity_limit: sm.ScalarData,
        name: Optional[str] = None,
        lower_slack_limit: sm.ScalarData = -LargeNumber,
        upper_slack_limit: sm.ScalarData = LargeNumber,
    ) -> None:
        """
        Add a velocity constraint. Internally, this will be converted into multiple constraints, to ensure that the
        velocity stays within the given bounds.
        :param lower_velocity_limit:
        :param upper_velocity_limit:
        :param quadratic_weight:
        :param task_expression:
        :param velocity_limit: Used for normalizing the expression, like reference_velocity, must be positive
        :param name:
        :param lower_slack_limit:
        :param upper_slack_limit:
        """

        constraint = GiskardInequalityConstraint(
            name=name,
            enforcement_strategy=VelocityStrategy,
            expression=task_expression,
            lower_bound=lower_velocity_limit,
            upper_bound=upper_velocity_limit,
            quadratic_weight=quadratic_weight,
            normalization_factor=velocity_limit,
            lower_slack_limit=lower_slack_limit,
            upper_slack_limit=upper_slack_limit,
            linear_weight=0,
        )
        self.add_constraint(constraint)

    def add_velocity_eq_constraint(
        self,
        velocity_goal: sm.ScalarData,
        quadratic_weight: sm.ScalarData,
        task_expression: sm.SymbolicScalar,
        velocity_limit: sm.ScalarData,
        name: Optional[str] = None,
        lower_slack_limit: sm.ScalarData = -LargeNumber,
        upper_slack_limit: sm.ScalarData = LargeNumber,
    ) -> None:
        """
        Add a velocity constraint. Internally, this will be converted into multiple constraints, to ensure that the
        velocity stays within the given bounds.
        :param velocity_goal:
        :param quadratic_weight:
        :param task_expression:
        :param velocity_limit: Used for normalizing the expression, like reference_velocity, must be positive
        :param name:
        :param lower_slack_limit:
        :param upper_slack_limit:
        """

        constraint = GiskardEqualityConstraint(
            name=name,
            enforcement_strategy=VelocityStrategy,
            expression=task_expression,
            bound=velocity_goal,
            quadratic_weight=quadratic_weight,
            normalization_factor=velocity_limit,
            lower_slack_limit=lower_slack_limit,
            upper_slack_limit=upper_slack_limit,
            linear_weight=0,
        )
        self.add_constraint(constraint)
