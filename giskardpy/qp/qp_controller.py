from __future__ import annotations

import datetime
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

import numpy as np
from line_profiler import profile

from giskardpy.data_types.exceptions import HardConstraintsViolatedException, InfeasibleException
from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.qp.adapters.qp_adapter import GiskardToQPAdapter, QPData
from giskardpy.qp.constraint import DerivativeEqualityConstraint
from giskardpy.qp.constraint import InequalityConstraint, EqualityConstraint, DerivativeInequalityConstraint
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.qp.next_command import NextCommands
from giskardpy.qp.weight_gain import QuadraticWeightGain, LinearWeightGain
from giskardpy.utils.utils import create_path
from semantic_digital_twin.spatial_types.symbol_manager import SymbolManager

if TYPE_CHECKING:
    from giskardpy.qp.qp_controller_config import QPControllerConfig

# used for saving pandas in the same folder every time within a run
date_str = datetime.datetime.now().strftime('%Yy-%mm-%dd--%Hh-%Mm-%Ss')


def save_pandas(dfs, names, path, time: float, folder_name: Optional[str] = None):
    import pandas as pd
    if folder_name is None:
        folder_name = ''
    folder_name = f'{path}/pandas/{folder_name}_{date_str}/{time}/'
    create_path(folder_name)
    for df, name in zip(dfs, names):
        csv_string = 'name\n'
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            if df.shape[1] > 1:
                for column_name, column in df.T.items():
                    zero_filtered_column = column.replace(0, np.nan).dropna(how='all').replace(np.nan, 0)
                    csv_string += zero_filtered_column.add_prefix(column_name + '||').to_csv(float_format='%.6f')
            else:
                csv_string += df.to_csv(float_format='%.6f')
        file_name2 = f'{folder_name}{name}.csv'
        with open(file_name2, 'w') as f:
            f.write(csv_string)


@dataclass
class QPController:
    """
    Wraps around QP Solver. Builds the required matrices from constraints.
    """
    config: QPControllerConfig
    qp_adapter: GiskardToQPAdapter = field(default=None)

    @profile
    def __post_init__(self):
        self.qp_solver = self.config.qp_solver_class()
        if self.config.verbose:
            get_middleware().loginfo(f'Initialized QP Controller:\n'
                                     f'sample period: "{self.config.mpc_dt}"s\n'
                                     f'max derivative: "{self.config.max_derivative.name}"\n'
                                     f'prediction horizon: "{self.config.prediction_horizon}"\n'
                                     f'QP solver: "{self.config.qp_solver_id.name}"')
        self.reset()

    def reset(self):
        self.degrees_of_freedoms = []
        self.equality_constraints = []
        self.inequality_constraints = []
        self.derivative_constraints = []
        self.eq_derivative_constraints = []
        self.quadratic_weight_gains = []
        self.linear_weight_gains = []
        self.xdot_full = None

    def init(self,
             degrees_of_freedom: List[FreeVariable] = None,
             equality_constraints: List[EqualityConstraint] = None,
             inequality_constraints: List[InequalityConstraint] = None,
             derivative_constraints: List[DerivativeInequalityConstraint] = None,
             eq_derivative_constraints: List[DerivativeEqualityConstraint] = None,
             quadratic_weight_gains: List[QuadraticWeightGain] = None,
             linear_weight_gains: List[LinearWeightGain] = None):
        self.degrees_of_freedoms = list(sorted(degrees_of_freedom, key=lambda dof: str(dof.name)))
        self.dof_filter = np.array([god_map.world.state.keys().index(dof.name) for dof in self.degrees_of_freedoms])
        self.qp_adapter = self.qp_solver.required_adapter_type(
            world_state_symbols=god_map.world.get_world_state_symbols(),
            task_life_cycle_symbols=god_map.motion_statechart_manager.task_state.get_life_cycle_state_symbols(),
            goal_life_cycle_symbols=god_map.motion_statechart_manager.goal_state.get_life_cycle_state_symbols(),
            external_collision_symbols=god_map.collision_scene.get_external_collision_symbol(),
            self_collision_symbols=god_map.collision_scene.get_self_collision_symbol(),
            free_variables=self.degrees_of_freedoms,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
            derivative_constraints=derivative_constraints,
            eq_derivative_constraints=eq_derivative_constraints,
            config=self.config)

        get_middleware().loginfo('Done compiling controller:')
        # get_middleware().loginfo(f'  #free variables: {self.num_free_variables}')
        # get_middleware().loginfo(f'  #equality constraints: {self.num_eq_constraints}')
        # get_middleware().loginfo(f'  #inequality constraints: {self.num_ineq_constraints}')

    def save_all_pandas(self, folder_name: Optional[str] = None):
        self._create_debug_pandas(self.qp_solver)
        save_pandas(
            [self.p_weights, self.p_b,
             self.p_E, self.p_bE,
             self.p_A, self.p_lbA, self.p_ubA,
             god_map.debug_expression_manager.to_pandas(), self.p_xdot],
            ['weights', 'b', 'E', 'bE', 'A', 'lbA', 'ubA', 'debug'],
            god_map.tmp_folder,
            god_map.time,
            folder_name)

    def _print_pandas_array(self, array):
        import pandas as pd
        if len(array) > 0:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(array)

    @profile
    def get_cmd(self, symbol_manager: SymbolManager) -> np.ndarray:
        """
        Uses substitutions for each symbol to compute the next commands for each joint.
        """
        try:
            qp_data = self.qp_adapter.evaluate(god_map.world.state.data,
                                               god_map.motion_statechart_manager.task_state.life_cycle_state,
                                               god_map.motion_statechart_manager.goal_state.life_cycle_state,
                                               god_map.collision_scene.get_external_collision_data(),
                                               god_map.collision_scene.get_self_collision_data(),
                                               symbol_manager)
            try:
                self.xdot_full = self.qp_solver.solver_call(qp_data.filtered)
            except InfeasibleException as e:
                self.config.retries_with_relaxed_constraints -= 1
                relaxed_solution = self.qp_solver.solver_call(qp_data.relaxed())
                if self.config.retries_with_relaxed_constraints < 0:
                    raise HardConstraintsViolatedException(f'Hard constraints were violated too often.')
                self.xdot_full = self.qp_solver.solver_call(qp_data.partially_relaxed(relaxed_solution))
            return self.xdot_to_control_commands(self.xdot_full)

        except InfeasibleException as e_original:
            self.xdot_full = None
            self._create_debug_pandas()
            # self._has_nan()
            # self._print_iis()
            if isinstance(e_original, HardConstraintsViolatedException):
                raise
            self.xdot_full = None
            self._are_hard_limits_violated(str(e_original))
            raise

    def xdot_to_control_commands(self, xdot: np.ndarray) -> np.ndarray:
        offset = len(self.degrees_of_freedoms) * (self.config.prediction_horizon - 2)
        offset_end = offset + len(self.degrees_of_freedoms)
        control_cmds = xdot[offset:offset_end]
        full_control_cmds = np.zeros(len(god_map.world.state))
        full_control_cmds[self.dof_filter] = control_cmds
        return full_control_cmds

    def _has_nan(self):
        nan_entries = self.p_A[0].isnull().stack()
        row_col_names = nan_entries[nan_entries].index.tolist()
        pass

    def _are_hard_limits_violated(self, error_message):
        self._create_debug_pandas()
        try:
            lower_violations = self.p_lb[self.qp_solver.lb_filter]
            upper_violations = self.p_ub[self.qp_solver.ub_filter]
            if len(upper_violations) > 0 or len(lower_violations) > 0:
                error_message += '\n'
                if len(upper_violations) > 0:
                    error_message += 'upper slack bounds of following constraints might be too low: {}\n'.format(
                        list(upper_violations.index))
                if len(lower_violations) > 0:
                    error_message += 'lower slack bounds of following constraints might be too high: {}'.format(
                        list(lower_violations.index))
                raise HardConstraintsViolatedException(error_message)
        except AttributeError:
            pass
        get_middleware().loginfo('No slack limit violation detected.')

    @profile
    def _create_debug_pandas(self) -> None:
        import pandas as pd

        qp_data: QPData = self.qp_adapter.qp_data_raw
        free_variable_names = self.qp_adapter.free_variable_bounds.names
        equality_constr_names = self.qp_adapter.equality_bounds.names
        inequality_constr_names = self.qp_adapter.inequality_bounds.names
        num_vel_constr = len(self.qp_adapter.derivative_constraints) * (self.config.prediction_horizon - 2)
        num_eq_vel_constr = len(self.qp_adapter.eq_derivative_constraints) * (self.config.prediction_horizon - 2)
        num_neq_constr = len(self.qp_adapter.inequality_constraints)
        num_eq_constr = len(self.qp_adapter.equality_constraints)
        num_constr = num_vel_constr + num_neq_constr + num_eq_constr + num_eq_vel_constr

        self.p_weights = pd.DataFrame(qp_data.quadratic_weights, free_variable_names, ['data'], dtype=float)
        self.p_g = pd.DataFrame(qp_data.linear_weights, free_variable_names, ['data'], dtype=float)
        self.p_lb = pd.DataFrame(qp_data.box_lower_constraints, free_variable_names, ['data'], dtype=float)
        self.p_ub = pd.DataFrame(qp_data.box_upper_constraints, free_variable_names, ['data'], dtype=float)
        self.p_b = pd.DataFrame({'lb': qp_data.box_lower_constraints, 'ub': qp_data.box_upper_constraints},
                                free_variable_names, dtype=float)
        if len(qp_data.eq_bounds) > 0:
            self.p_bE_raw = pd.DataFrame(qp_data.eq_bounds, equality_constr_names, ['data'], dtype=float)
            self.p_bE = deepcopy(self.p_bE_raw)
            self.p_bE[len(self.qp_adapter.equality_bounds.names_derivative_links):] /= self.config.mpc_dt
        else:
            self.p_bE = pd.DataFrame()
        if len(qp_data.neq_lower_bounds) > 0:
            self.p_lbA_raw = pd.DataFrame(qp_data.neq_lower_bounds, inequality_constr_names, ['data'],
                                          dtype=float)
            self.p_lbA = deepcopy(self.p_lbA_raw)
            self.p_lbA /= self.config.mpc_dt

            self.p_ubA_raw = pd.DataFrame(qp_data.neq_upper_bounds, inequality_constr_names, ['data'],
                                          dtype=float)
            self.p_ubA = deepcopy(self.p_ubA_raw)
            self.p_ubA /= self.config.mpc_dt

            self.p_bA_raw = pd.DataFrame({'lbA': qp_data.neq_lower_bounds, 'ubA': qp_data.neq_upper_bounds},
                                         inequality_constr_names, dtype=float)
            self.p_bA = deepcopy(self.p_bA_raw)
            self.p_bA /= self.config.mpc_dt
        else:
            self.p_lbA = pd.DataFrame()
            self.p_ubA = pd.DataFrame()
        # remove sample period factor
        if len(qp_data.dense_eq_matrix) > 0:
            self.p_E = pd.DataFrame(qp_data.dense_eq_matrix, equality_constr_names, free_variable_names,
                                    dtype=float)
        else:
            self.p_E = pd.DataFrame()
        if len(qp_data.dense_neq_matrix) > 0:
            self.p_A = pd.DataFrame(qp_data.dense_neq_matrix, inequality_constr_names, free_variable_names,
                                    dtype=float)
        else:
            self.p_A = pd.DataFrame()
        self.p_xdot = None
        if self.xdot_full is not None:
            xdot_full = np.ones(qp_data.zero_quadratic_weight_filter.shape) * np.nan
            xdot_full[qp_data.zero_quadratic_weight_filter] = self.xdot_full
            self.p_xdot = pd.DataFrame(xdot_full, free_variable_names, ['data'], dtype=float)
            self.p_b['xdot'] = self.p_xdot
            self.p_b = self.p_b[['lb', 'xdot', 'ub']]
            self.p_pure_xdot = deepcopy(self.p_xdot)
            self.p_pure_xdot[-num_constr:] = 0
            # self.p_Ax = pd.DataFrame(self.p_A.dot(self.p_xdot), self.inequality_constr_names, ['data'], dtype=float)
            if len(self.p_A) > 0:
                self.p_Ax = pd.DataFrame(self.p_A.dot(self.p_pure_xdot), inequality_constr_names,
                                         ['data'], dtype=float)
                bA_slack = self.p_xdot[len(self.p_xdot) - num_neq_constr - num_vel_constr:]
                self.p_bA_raw.insert(1, 'Ax', self.p_Ax)
                self.p_bA_raw.insert(2, 'slack', bA_slack.values)
            else:
                self.p_Ax = pd.DataFrame()
            # self.p_Ax_without_slack = deepcopy(self.p_Ax_without_slack_raw)
            # self.p_Ax_without_slack[-num_constr:] /= self.sample_period
            if len(self.p_E) > 0:
                self.p_Ex = pd.DataFrame(self.p_E.dot(self.p_pure_xdot), equality_constr_names,
                                         ['data'], dtype=float)
            else:
                self.p_Ex = pd.DataFrame()

        else:
            self.p_xdot = None
        self.p_debug = god_map.debug_expression_manager.to_pandas()

    def _print_iis(self):
        import pandas as pd

        def print_iis_matrix(row_filter: np.ndarray, column_filter: np.ndarray, matrix: pd.DataFrame,
                             bounds: pd.DataFrame):
            if len(row_filter) == 0:
                return
            filtered_matrix = matrix.loc[row_filter, column_filter]
            filtered_matrix['bounds'] = bounds.loc[row_filter]
            print(filtered_matrix)

        result = self.qp_solver.analyze_infeasibility()
        if result is None:
            get_middleware().loginfo(f'Can only compute possible causes with gurobi, '
                                     f'but current solver is {self.config.qp_solver_id.name}.')
            return
        lb_ids, ub_ids, eq_ids, lbA_ids, ubA_ids = result
        b_ids = lb_ids | ub_ids
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
            get_middleware().loginfo('Irreducible Infeasible Subsystem:')
            get_middleware().loginfo('  Free variable bounds')
            free_variables = self.p_lb[b_ids]
            free_variables['ub'] = self.p_ub[b_ids]
            free_variables = free_variables.rename(columns={'data': 'lb'})
            print(free_variables)
            get_middleware().loginfo('  Equality constraints:')
            print_iis_matrix(eq_ids, b_ids, self.p_E, self.p_bE)
            get_middleware().loginfo('  Inequality constraint lower bounds:')
            print_iis_matrix(lbA_ids, b_ids, self.p_A, self.p_lbA)
            get_middleware().loginfo('  Inequality constraint upper bounds:')
            print_iis_matrix(ubA_ids, b_ids, self.p_A, self.p_ubA)
