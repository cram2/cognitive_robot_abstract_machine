from __future__ import annotations

import logging
from dataclasses import dataclass, field, InitVar
from typing_extensions import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.qp.constraint_collection import ConstraintCollection
from giskardpy.qp.qp_data_factories import QPDataFactory
from giskardpy.qp.exceptions import InfeasibleException, SolverReturnedFailureError
from giskardpy.qp.qp_data import QPData, QPDataExplicit
from giskardpy.qp.qp_data_symbolic import QPDataSymbolic
from giskardpy.qp.qp_debugger import QuadraticProgramDebugger
from giskardpy.qp.solvers.qp_solver import QPSolver
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from giskardpy.qp.qp_controller_config import QPControllerConfig


@dataclass
class QPController:
    """
    Wraps around QP Solver.

    Builds the required matrices from constraints.
    """

    config: QPControllerConfig
    degrees_of_freedom: InitVar[list[DegreeOfFreedom]]
    active_degrees_of_freedom: list[DegreeOfFreedom] = field(init=False)
    constraint_collection: ConstraintCollection
    world_state_symbols: list[sm.FloatVariable]
    life_cycle_variables: list[sm.FloatVariable]
    float_variables: list[sm.FloatVariable]

    qp_data_factory: QPDataFactory = field(default=None, init=False)
    qp_solver: QPSolver = field(default=None, init=False)
    debugger: QuadraticProgramDebugger = field(default=None, init=False)
    dof_filter: np.ndarray = field(default=None, init=False)

    def __post_init__(self, degrees_of_freedom: list[DegreeOfFreedom]):
        self.qp_solver = self.config.qp_solver_class()
        if self.config.verbose:
            logger.info(
                f"Initialized QP Controller:\n"
                f'sample period: "{self.config.control_time_step.total_seconds()}"s\n'
                f'max derivative: "{self.config.max_derivative.name}"\n'
                f'prediction horizon: "{self.config.prediction_horizon}"\n'
                f'QP solver: "{self.config.qp_solver_class.__name__}"'
            )
        self._set_active_dofs(degrees_of_freedom)
        generic_qp_data_symbolic = QPDataSymbolic(
            degrees_of_freedom=self.active_degrees_of_freedom,
            constraint_collection=self.constraint_collection,
            qp_controller_config=self.config,
        )

        qp_data_factory_class = QPDataFactory.get_factory_from_qp_data_type(
            self.qp_solver.qp_data_type()
        )
        self.qp_data_factory = qp_data_factory_class(generic_qp_data_symbolic)
        self.qp_data_factory.compile(
            world_state_symbols=self.world_state_symbols,
            life_cycle_symbols=self.life_cycle_variables,
            float_variables=self.float_variables,
        )
        self.debugger = QuadraticProgramDebugger(
            qp_data_symbolic=self.qp_data_factory.qp_data
        )

    def _set_active_dofs(self, degrees_of_freedom: list[DegreeOfFreedom]) -> None:
        all_active_float_variables = set().union(
            *[
                {
                    dof.variables.position.name,
                    dof.variables.velocity.name,
                    dof.variables.acceleration.name,
                    dof.variables.jerk.name,
                }
                for dof in degrees_of_freedom
            ]
        )
        float_variable_names = self.constraint_collection.get_all_float_variable_names()
        active_float_variables = all_active_float_variables & float_variable_names

        def dof_used(dof: DegreeOfFreedom) -> bool:
            vars_ = dof.variables
            return (
                vars_.position.name in float_variable_names
                or vars_.velocity.name in float_variable_names
                or vars_.acceleration.name in float_variable_names
                or vars_.jerk.name in float_variable_names
            )

        self.dof_filter = np.array(
            [
                i
                for i, v in enumerate(self.world_state_symbols)
                if v.name in active_float_variables
            ]
        )
        self.active_degrees_of_freedom = [
            dof for dof in degrees_of_freedom if dof_used(dof)
        ]

    def has_not_free_variables(self) -> bool:
        return len(self.active_degrees_of_freedom) == 0

    def compute_command(
        self,
        world_state: np.ndarray,
        life_cycle_state: np.ndarray,
        float_variables: np.ndarray,
    ) -> np.ndarray:
        """
        Uses substitutions for each symbol to compute the next commands for each joint.
        """
        qp_data_raw = self.qp_data_factory.evaluate(
            world_state, life_cycle_state, float_variables
        )
        qp_data_filtered = qp_data_raw.apply_filters()
        stopped = self._unconstrained_degrees_of_freedom_able_to_stop(qp_data_filtered)
        if not stopped.any():
            return self.extract_control_commands(self._solve(qp_data_filtered))
        return self.extract_control_commands(
            self._solve_without_stopped(qp_data_filtered, stopped)
        )

    def _solve_without_stopped(
        self, qp_data: QPDataExplicit, stopped: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        """
        :param qp_data: The problem of this control cycle.
        :param stopped: Which degrees of freedom are stopped instead of solved for,
            one flag per active degree of freedom.
        :return: The value of every variable of `qp_data`: the solution of the problem
            without the stopped degrees of freedom, and for those the values that
            bring them to rest.
        """
        number_of_variables = len(qp_data.quadratic_weights)
        number_of_steps = self._number_of_degree_of_freedom_variables // len(stopped)
        removed_variables = np.zeros(number_of_variables, dtype=bool)
        removed_variables[: self._number_of_degree_of_freedom_variables] = np.tile(
            stopped, number_of_steps
        )
        solution = np.zeros(number_of_variables)
        solution[: self._number_of_degree_of_freedom_variables] = (
            self._stopping_solution(qp_data)
        )
        if not removed_variables.all():
            solution[~removed_variables] = self._solve(
                qp_data.without_variables(removed_variables)
            )
        return solution

    def _solve(self, qp_data: QPData) -> np.ndarray:
        """
        :param qp_data: The problem to solve.
        :return: The value of every variable of `qp_data`.
        :raises InfeasibleException: If the problem has no solution.
        :raises SolverReturnedFailureError: If the solver gave up on the problem.
        """
        try:
            return self.qp_solver.solver_call(qp_data)
        except (InfeasibleException, SolverReturnedFailureError):
            self.report_unsolvable_problem(qp_data)
            raise

    @property
    def _number_of_velocity_variables(self) -> int:
        """
        :return: How many velocity variables the degrees of freedom have over the
            prediction horizon.
        """
        return len(self.active_degrees_of_freedom) * (
            self.config.prediction_horizon - 2
        )

    @property
    def _number_of_degree_of_freedom_variables(self) -> int:
        """
        :return: How many velocity and jerk variables the degrees of freedom have over
            the prediction horizon.
        """
        return self.qp_data_factory.qp_data.number_non_slack_variables

    def _stopping_solution(self, qp_data: QPDataExplicit) -> NDArray[np.float64]:
        """
        :param qp_data: The problem of this control cycle.
        :return: The value of every velocity and jerk variable with which each degree
            of freedom is at rest after one control cycle, wherever the constraints
            would rather have it go.
        """
        number_of_degrees_of_freedom = len(self.active_degrees_of_freedom)
        first_jerk = self._number_of_velocity_variables
        solution = np.zeros(self._number_of_degree_of_freedom_variables)
        # The first two rows of the system dynamics hold what the first two jerks have
        # to make up for when every velocity is zero.
        solution[first_jerk : first_jerk + 2 * number_of_degrees_of_freedom] = (
            qp_data.equality_bounds[: 2 * number_of_degrees_of_freedom]
        )
        return solution

    def _unconstrained_degrees_of_freedom_able_to_stop(
        self, qp_data: QPData
    ) -> NDArray[np.bool_]:
        """
        A degree of freedom that no constraint of the problem refers to only has to
        stand still. A solver finds that within its tolerance only, which lets the
        degree of freedom creep, so it is stopped without the solver once its limits
        allow coming to rest within one control cycle.

        :param qp_data: The problem of this control cycle, without the constraints of
            nodes that are not running.
        :return: Which degrees of freedom are stopped without the solver, one flag per
            active degree of freedom. None is stopped for a problem that is not a
            :class:`~giskardpy.qp.qp_data.QPDataExplicit`.
        """
        number_of_degrees_of_freedom = len(self.active_degrees_of_freedom)
        if not isinstance(qp_data, QPDataExplicit):
            return np.zeros(number_of_degrees_of_freedom, dtype=bool)
        number_of_variables = self._number_of_degree_of_freedom_variables
        number_of_system_dynamics_rows = (
            number_of_variables - self._number_of_velocity_variables
        )
        unconstrained = qp_data.variables_without_coefficients(
            number_of_variables=number_of_variables,
            number_of_ignored_equalities=number_of_system_dynamics_rows,
        )
        stopping_solution = self._stopping_solution(qp_data)
        within_limits = (
            qp_data.box_lower_constraints[:number_of_variables] <= stopping_solution
        ) & (stopping_solution <= qp_data.box_upper_constraints[:number_of_variables])
        return (
            (unconstrained & within_limits)
            .reshape(-1, number_of_degrees_of_freedom)
            .all(axis=0)
        )

    @staticmethod
    def report_unsolvable_problem(qp_data: QPData) -> None:
        """
        Log a problem the solver failed on, so it can be turned into a test case.

        Only :class:`~giskardpy.qp.qp_data.QPDataExplicit` is printed and its numerical
        problems logged; other formats only report the failure.

        :param qp_data: The problem the solver failed on.
        """
        if not isinstance(qp_data, QPDataExplicit):
            logger.warning(
                f"The QP solver failed on a {type(qp_data).__name__}, which cannot be "
                f"printed."
            )
            return
        logger.warning(qp_data.pretty_print_problem())
        qp_data.analyze_well_posedness()

    def extract_control_commands(self, solution: np.ndarray) -> np.ndarray:
        """
        Reads the control command of every degree of freedom out of the QP solution.

        :param solution: The QP decision vector returned by the solver.
        :return: One control command per degree of freedom of the world state, zero for
            degrees of freedom that are not active in this QP.
        """
        offset = len(self.active_degrees_of_freedom) * (
            self.config.prediction_horizon - 2
        )
        offset_end = offset + len(self.active_degrees_of_freedom)
        control_commands = (
            solution[offset:offset_end]
            / self.config.control_time_step.total_seconds() ** 2
        )
        # divide by 4 because the world state has pos/vel/acc/jerk variables
        full_control_commands = np.zeros(len(self.world_state_symbols) // 4)
        full_control_commands[self.dof_filter] = control_commands
        return full_control_commands
