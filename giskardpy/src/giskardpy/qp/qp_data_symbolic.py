from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.qp import qp_controller_config
from giskardpy.qp.dof_limits import QuadraticProgramDegreeOfFreedomLimits
from giskardpy.qp.enforcement_strategy import (
    EnforcementStrategy,
    SystemDynamicsStrategy,
)
from giskardpy.qp.constraint_collection import ConstraintCollection
from krrood.symbolic_math.symbolic_math import Vector, Matrix
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from giskardpy.qp.constraint import GiskardConstraint
    from giskardpy.qp.qp_controller_config import QPControllerConfig


@dataclass
class QPVariableAccumulator:
    """
    Collects the per-block cost weights, box bounds, and variable names of all decision and slack
    variables while the QP is assembled, so the assembly logic stays free of hidden side effects.
    """

    quadratic_weights: list[Vector] = field(default_factory=list)
    """
    The quadratic cost blocks, one per registered variable block.
    """
    linear_weights: list[Vector] = field(default_factory=list)
    """
    The linear cost blocks, one per registered variable block.
    """
    box_lower_constraints: list[Vector] = field(default_factory=list)
    """
    The lower box-bound blocks, one per registered variable block.
    """
    box_upper_constraints: list[Vector] = field(default_factory=list)
    """
    The upper box-bound blocks, one per registered variable block.
    """
    free_variable_names: list[str] = field(default_factory=list)
    """
    The names of all registered variables, in column order.
    """


@dataclass(eq=False)
class QPDataSymbolic:
    """
    Takes free variables and constraints and converts them to a QP problem in the following format, depending on the
    class attributes:
    min_x 0.5 x^T H x + g^T x
    s.t.  box_lower_constraints <= x <= box_upper_constraints     (box constraints)
          equality_matrix x <= equality_bounds          (equality constraints)
          inequality_lower_bounds <= inequality_matrix x <= inequality_upper_bounds  (lower/upper inequality constraints)
    """

    degrees_of_freedom: list[DegreeOfFreedom]
    """
    The degrees of freedom whose decision variables make up the non-slack part of the QP.
    """
    constraint_collection: ConstraintCollection
    """
    The equality and inequality constraints to encode into the QP.
    """
    qp_controller_config: QPControllerConfig
    """
    Controller configuration, e.g. prediction horizon and time step.
    """

    quadratic_weights: Vector = field(init=False)
    """
    Diagonal of the QP cost matrix H, for all decision and slack variables.
    """
    linear_weights: Vector = field(init=False)
    """
    The linear cost vector g, for all decision and slack variables.
    """

    box_lower_constraints: Vector = field(init=False)
    """
    Lower box bounds lb for all decision and slack variables.
    """
    box_upper_constraints: Vector = field(init=False)
    """
    Upper box bounds ub for all decision and slack variables.
    """

    free_variable_names: list[str] = field(init=False)
    """
    Names of all decision and slack variables, in column order.
    """

    equality_matrix_degrees_of_freedom: Matrix = field(init=False)
    """
    Equality constraint matrix block acting on the degree-of-freedom variables.
    """
    equality_matrix_slack: Matrix = field(init=False)
    """
    Equality constraint matrix block acting on the equality slack variables.
    """
    equality_bounds: Vector = field(init=False)
    """
    Right-hand side bounds of the equality constraints.
    """
    equality_constraint_names: list[str] = field(init=False)
    """
    Names of the equality constraints, in row order.
    """

    inequality_matrix_degrees_of_freedom: Matrix = field(init=False)
    """
    Inequality constraint matrix block acting on the degree-of-freedom variables.
    """
    inequality_matrix_slack: Matrix = field(init=False)
    """
    Inequality constraint matrix block acting on the inequality slack variables.
    """
    inequality_lower_bounds: Vector = field(init=False)
    """
    Lower bounds of the inequality constraints.
    """
    inequality_upper_bounds: Vector = field(init=False)
    """
    Upper bounds of the inequality constraints.
    """
    inequality_constraint_names: list[str] = field(init=False)
    """
    Names of the inequality constraints, in row order.
    """

    @staticmethod
    def _append_slack_block(
        strategy: EnforcementStrategy,
        constraint_names: list[str],
        accumulator: QPVariableAccumulator,
    ) -> tuple[Matrix, Matrix]:
        """
        Appends the strategy's slack weights, box bounds, and names to the accumulator and the given
        constraint-name list, and returns its constraint matrix and slack matrix.
        """
        slack_variables = strategy.create_slack_variables()
        accumulator.quadratic_weights.append(slack_variables.quadratic_weights)
        accumulator.linear_weights.append(slack_variables.linear_weights)
        accumulator.box_lower_constraints.append(slack_variables.lower_bounds)
        accumulator.box_upper_constraints.append(slack_variables.upper_bounds)
        constraint_names.extend(strategy.create_names())
        accumulator.free_variable_names.extend(slack_variables.names)
        return strategy.create_matrix(), strategy.create_slack_matrix()

    def __post_init__(self):
        accumulator = self._collect_degree_of_freedom_limits()
        self._assemble_equality_constraints(accumulator)
        self._assemble_inequality_constraints(accumulator)
        self._store_box_constraints(accumulator)

    def _collect_degree_of_freedom_limits(self) -> QPVariableAccumulator:
        """
        Creates the variable accumulator seeded with the box bounds, weights, and names of the
        degree-of-freedom decision variables.
        """
        direct_limits = QuadraticProgramDegreeOfFreedomLimits.create(
            self.degrees_of_freedom, self.qp_controller_config
        )
        return QPVariableAccumulator(
            quadratic_weights=[direct_limits.quadratic_weights],
            linear_weights=[direct_limits.linear_weights],
            box_lower_constraints=[direct_limits.lower_bounds],
            box_upper_constraints=[direct_limits.upper_bounds],
            free_variable_names=list(direct_limits.names),
        )

    def _instantiate_strategy(
        self,
        enforcement_strategy: type[EnforcementStrategy],
        constraints: list[GiskardConstraint],
    ) -> EnforcementStrategy:
        """
        Instantiates an enforcement strategy for a constraint block with this QP's degrees of
        freedom and controller configuration.
        """
        return enforcement_strategy(
            degrees_of_freedom=self.degrees_of_freedom,
            qp_controller_config=self.qp_controller_config,
            constraints=constraints,
        )

    def _assemble_equality_constraints(
        self, accumulator: QPVariableAccumulator
    ) -> None:
        """
        Builds and stores the equality constraint matrices, slack blocks, bounds, and row names,
        starting with the system dynamics block and followed by the per-strategy equality blocks.
        """
        system_dynamics_strategy = self._instantiate_strategy(
            SystemDynamicsStrategy, []
        )
        degree_of_freedom_matrices = [system_dynamics_strategy.create_matrix()]
        slack_matrices = [system_dynamics_strategy.create_slack_matrix()]
        bounds = [system_dynamics_strategy.create_equality_bounds()]
        self.equality_constraint_names = list(system_dynamics_strategy.create_names())

        for (
            enforcement_strategy,
            constraints,
        ) in self.constraint_collection.get_equality_constraint_blocks().items():
            strategy = self._instantiate_strategy(enforcement_strategy, constraints)
            matrix, slack_matrix = self._append_slack_block(
                strategy, self.equality_constraint_names, accumulator
            )
            degree_of_freedom_matrices.append(matrix)
            slack_matrices.append(slack_matrix)
            bounds.append(strategy.create_equality_bounds())

        self.equality_matrix_degrees_of_freedom = sm.vstack(degree_of_freedom_matrices)
        self.equality_matrix_slack = sm.diag_stack(slack_matrices)
        self.equality_bounds = sm.concatenate(*bounds)

    def _assemble_inequality_constraints(
        self, accumulator: QPVariableAccumulator
    ) -> None:
        """
        Builds and stores the inequality constraint matrices, slack blocks, lower/upper bounds,
        and row names from the per-strategy inequality blocks.
        """
        degree_of_freedom_matrices = []
        slack_matrices = []
        lower_bounds = []
        upper_bounds = []
        self.inequality_constraint_names = []

        for (
            enforcement_strategy,
            constraints,
        ) in self.constraint_collection.get_inequality_constraint_blocks().items():
            strategy = self._instantiate_strategy(enforcement_strategy, constraints)
            matrix, slack_matrix = self._append_slack_block(
                strategy, self.inequality_constraint_names, accumulator
            )
            degree_of_freedom_matrices.append(matrix)
            slack_matrices.append(slack_matrix)
            lower_bounds.append(strategy.create_lower_bounds())
            upper_bounds.append(strategy.create_upper_bounds())

        self.inequality_matrix_degrees_of_freedom = self._stack_rows(
            degree_of_freedom_matrices
        )
        self.inequality_matrix_slack = self._diagonally_stack(slack_matrices)
        self.inequality_lower_bounds = self._concatenate_blocks(lower_bounds)
        self.inequality_upper_bounds = self._concatenate_blocks(upper_bounds)

    def _store_box_constraints(self, accumulator: QPVariableAccumulator) -> None:
        """
        Concatenates the accumulated box bounds, weights, and names into the QP's box constraint
        fields.
        """
        self.free_variable_names = accumulator.free_variable_names
        self.quadratic_weights = sm.concatenate(*accumulator.quadratic_weights)
        self.linear_weights = sm.concatenate(*accumulator.linear_weights)
        self.box_lower_constraints = sm.concatenate(*accumulator.box_lower_constraints)
        self.box_upper_constraints = sm.concatenate(*accumulator.box_upper_constraints)

    @staticmethod
    def _stack_rows(blocks: list[Matrix]) -> Matrix:
        """
        Vertically stacks the constraint matrix blocks, returning an empty matrix when there are
        no blocks.
        """
        return sm.vstack(blocks) if blocks else sm.Matrix()

    @staticmethod
    def _diagonally_stack(blocks: list[Matrix]) -> Matrix:
        """
        Diagonally stacks the slack matrix blocks, returning an empty matrix when there are no
        blocks.
        """
        return sm.diag_stack(blocks) if blocks else sm.Matrix()

    @staticmethod
    def _concatenate_blocks(blocks: list[Vector]) -> Vector:
        """
        Concatenates the bound blocks, returning an empty vector when there are no blocks.
        """
        return sm.concatenate(*blocks) if blocks else sm.Vector()

    @property
    def number_degrees_of_freedom(self) -> int:
        """
        The number of degrees of freedom.
        """
        return len(self.degrees_of_freedom)

    @property
    def number_equality_slack_variables(self) -> int:
        """
        The number of slack columns introduced by the equality constraints.
        """
        return self.equality_matrix_slack.shape[1]

    @property
    def number_inequality_slack_variables(self) -> int:
        """
        The number of slack columns introduced by the inequality constraints.
        """
        return self.inequality_matrix_slack.shape[1]

    @property
    def number_slack_variables(self) -> int:
        """
        The total number of slack columns.
        """
        return (
            self.number_equality_slack_variables
            + self.number_inequality_slack_variables
        )

    @property
    def number_non_slack_variables(self) -> int:
        """
        The number of degree-of-freedom decision variable columns, i.e. all non-slack columns.
        """
        return self.quadratic_weights.shape[0] - self.number_slack_variables
