from __future__ import annotations

import logging
from dataclasses import dataclass, field, InitVar
from typing import List, TYPE_CHECKING

import numpy as np

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.qp.constraint_collection import ConstraintCollection
from giskardpy.qp.qp_data_factories import QPDataFactory
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
    degrees_of_freedom: InitVar[List[DegreeOfFreedom]]
    active_dofs: List[DegreeOfFreedom] = field(init=False)
    constraint_collection: ConstraintCollection
    world_state_symbols: List[sm.FloatVariable]
    life_cycle_variables: List[sm.FloatVariable]
    float_variables: List[sm.FloatVariable]

    qp_data_factory: QPDataFactory = field(default=None, init=False)
    qp_solver: QPSolver = field(default=None, init=False)
    debugger: QuadraticProgramDebugger = field(default=None, init=False)
    dof_filter: np.ndarray = field(default=None, init=False)

    def __post_init__(self, degrees_of_freedom: List[DegreeOfFreedom]):
        self.qp_solver = self.config.qp_solver_class()
        if self.config.verbose:
            logger.info(
                f"Initialized QP Controller:\n"
                f'sample period: "{self.config.model_predictive_control_time_step}"s\n'
                f'max derivative: "{self.config.max_derivative.name}"\n'
                f'prediction horizon: "{self.config.prediction_horizon}"\n'
                f'QP solver: "{self.config.qp_solver_class.__name__}"'
            )
        self._set_active_dofs(degrees_of_freedom)
        generic_qp_data_symbolic = QPDataSymbolic(
            degrees_of_freedom=self.active_dofs,
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

    def _set_active_dofs(self, degrees_of_freedom: List[DegreeOfFreedom]) -> None:
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
        self.active_dofs = [dof for dof in degrees_of_freedom if dof_used(dof)]

    def has_not_free_variables(self) -> bool:
        return len(self.active_dofs) == 0

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
        solution = self.qp_solver.solver_call(qp_data_filtered)
        return self.xdot_to_control_commands(solution)

    def xdot_to_control_commands(self, xdot: np.ndarray) -> np.ndarray:
        offset = len(self.active_dofs) * (self.config.prediction_horizon - 2)
        offset_end = offset + len(self.active_dofs)
        control_cmds = (
            xdot[offset:offset_end] / self.config.model_predictive_control_time_step**2
        )
        # divide by 4 because the world state has pos/vel/acc/jerk variables
        full_control_cmds = np.zeros(len(self.world_state_symbols) // 4)
        full_control_cmds[self.dof_filter] = control_cmds
        return full_control_cmds
