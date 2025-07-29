from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional, List, Union, Dict, Callable, Iterable, overload, DefaultDict, TypeVar

import numpy as np

import semantic_world.spatial_types.spatial_types as cas
from giskardpy.data_types.exceptions import GoalInitalizationException, DuplicateNameException
from giskardpy.god_map import god_map
from giskardpy.data_types.data_types import Derivatives, LifeCycleState
from giskardpy.qp.constraint import EqualityConstraint, InequalityConstraint, DerivativeInequalityConstraint
from semantic_world.spatial_types.symbol_manager import symbol_manager
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.qp.weight_gain import QuadraticWeightGain, LinearWeightGain
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.qp.constraint import DerivativeEqualityConstraint

WEIGHT_MAX = 10000
WEIGHT_ABOVE_CA = 2500
WEIGHT_COLLISION_AVOIDANCE = 50
WEIGHT_BELOW_CA = 1
WEIGHT_MIN = 0

T = TypeVar('T', EqualityConstraint, InequalityConstraint, DerivativeInequalityConstraint,
            DerivativeEqualityConstraint)


@dataclass
class Task(MotionStatechartNode):
    """
    Tasks are a set of constraints with the same predicates.
    """
    eq_constraints: List[EqualityConstraint] = field(default_factory=list, init=False)
    neq_constraints: List[InequalityConstraint] = field(default_factory=list, init=False)

    derivative_constraints: List[DerivativeInequalityConstraint] = field(default_factory=list, init=False)
    eq_derivative_constraints: List[DerivativeEqualityConstraint] = field(default_factory=list, init=False)

    quadratic_gains: List[QuadraticWeightGain] = field(default_factory=list, init=False)
    linear_weight_gains: List[LinearWeightGain] = field(default_factory=list, init=False)

    @cached_property
    def observation_state_symbol(self) -> cas.Symbol:
        symbols_name = f'{self.name}.observation_state'
        return symbol_manager.register_symbol_provider(symbols_name, lambda
            n=self.name: god_map.motion_statechart_manager.task_state.get_observation_state(n))

    @cached_property
    def life_cycle_state_symbol(self):
        symbols_name = f'{self.name}.life_cycle_state'
        return symbol_manager.register_symbol_provider(symbols_name, lambda
            n=self.name: god_map.motion_statechart_manager.task_state.get_life_cycle_state(n))

    @property
    def ref_str(self) -> str:
        """
        A string referring to self on the god_map. Used with symbol manager.
        """
        return f'god_map.motion_statechart_manager.task_state.get_node(\'{str(self)}\')'

    def get_eq_constraints(self) -> List[EqualityConstraint]:
        return self._apply_monitors_to_constraints(self.eq_constraints)

    def get_neq_constraints(self) -> List[InequalityConstraint]:
        return self._apply_monitors_to_constraints(self.neq_constraints)

    def get_derivative_constraints(self) -> List[DerivativeInequalityConstraint]:
        return self._apply_monitors_to_constraints(self.derivative_constraints)

    def get_eq_derivative_constraints(self) -> List[DerivativeEqualityConstraint]:
        return self._apply_monitors_to_constraints(self.eq_derivative_constraints)

    def get_quadratic_gains(self) -> List[QuadraticWeightGain]:
        return self.quadratic_gains

    def get_linear_gains(self) -> List[LinearWeightGain]:
        return self.linear_weight_gains

    def _apply_monitors_to_constraints(self, constraints: List[T]) -> List[T]:
        output_constraints = []
        for constraint in constraints:
            is_running = cas.if_eq(self.life_cycle_state_symbol, int(LifeCycleState.running),
                                   if_result=1,
                                   else_result=0)
            constraint.quadratic_weight *= is_running
            output_constraints.append(constraint)
        return output_constraints

    def add_quadratic_weight_gain(self, name: str, gains: List[DefaultDict[Derivatives, Dict[FreeVariable, float]]]):
        q_gain = QuadraticWeightGain(name=name,
                                     gains=gains)
        self.quadratic_gains.append(q_gain)

    def add_linear_weight_gain(self, name: str, gains: List[DefaultDict[Derivatives, Dict[FreeVariable, float]]]):
        q_gain = LinearWeightGain(name=name,
                                  gains=gains)
        self.linear_weight_gains.append(q_gain)

    def add_equality_constraint(self,
                                reference_velocity: cas.symbol_expr_float,
                                equality_bound: cas.symbol_expr_float,
                                weight: cas.symbol_expr_float,
                                task_expression: cas.symbol_expr,
                                name: str = None,
                                lower_slack_limit: Optional[cas.symbol_expr_float] = None,
                                upper_slack_limit: Optional[cas.symbol_expr_float] = None):
        """
        Add a task constraint to the motion problem. This should be used for most constraints.
        It will not strictly stick to the reference velocity, but requires only a single constraint in the final
        optimization problem and is therefore faster.
        :param reference_velocity: used by Giskard to limit the error and normalize the weight, will not be strictly
                                    enforced.
        :param task_expression: defines the task function
        :param equality_bound: goal for the derivative of task_expression
        :param weight:
        :param name: give this constraint a name, required if you add more than one in the same goal
        :param lower_slack_limit: how much the lower error can be violated, don't use unless you know what you are doing
        :param upper_slack_limit: how much the upper error can be violated, don't use unless you know what you are doing
        """
        if task_expression.shape != (1, 1):
            raise GoalInitalizationException(f'expression must have shape (1, 1), has {task_expression.shape}')
        name = name or f'{len(self.eq_constraints)}'
        lower_slack_limit = lower_slack_limit if lower_slack_limit is not None else -float('inf')
        upper_slack_limit = upper_slack_limit if upper_slack_limit is not None else float('inf')
        constraint = EqualityConstraint(constraint_name=name,
                                        parent_task_name=self.name,
                                        expression=task_expression,
                                        bound=equality_bound,
                                        velocity_limit=reference_velocity,
                                        quadratic_weight=weight,
                                        lower_slack_limit=lower_slack_limit,
                                        upper_slack_limit=upper_slack_limit)
        if constraint.name in self.eq_constraints:
            raise DuplicateNameException(f'Constraint named {constraint.name} already exists.')
        self.eq_constraints.append(constraint)

    def add_inequality_constraint(self,
                                  reference_velocity: cas.symbol_expr_float,
                                  lower_error: cas.symbol_expr_float,
                                  upper_error: cas.symbol_expr_float,
                                  weight: cas.symbol_expr_float,
                                  task_expression: cas.symbol_expr,
                                  name: Optional[str] = None,
                                  lower_slack_limit: Optional[cas.symbol_expr_float] = None,
                                  upper_slack_limit: Optional[cas.symbol_expr_float] = None):
        """
        Add a task constraint to the motion problem. This should be used for most constraints.
        It will not strictly stick to the reference velocity, but requires only a single constraint in the final
        optimization problem and is therefore faster.
        :param reference_velocity: used by Giskard to limit the error and normalize the weight, will not be strictly
                                    enforced.
        :param lower_error: lower bound for the error of expression
        :param upper_error: upper bound for the error of expression
        :param weight:
        :param task_expression: defines the task function
        :param name: give this constraint a name, required if you add more than one in the same goal
        :param lower_slack_limit: how much the lower error can be violated, don't use unless you know what you are doing
        :param upper_slack_limit: how much the upper error can be violated, don't use unless you know what you are doing
        """
        if task_expression.shape != (1, 1):
            raise GoalInitalizationException(f'expression must have shape (1,1), has {task_expression.shape}')
        name = name or ''
        lower_slack_limit = lower_slack_limit if lower_slack_limit is not None else -float('inf')
        upper_slack_limit = upper_slack_limit if upper_slack_limit is not None else float('inf')
        constraint = InequalityConstraint(constraint_name=name,
                                          parent_task_name=self.name,
                                          expression=task_expression,
                                          lower_error=lower_error,
                                          upper_error=upper_error,
                                          velocity_limit=reference_velocity,
                                          quadratic_weight=weight,
                                          lower_slack_limit=lower_slack_limit,
                                          upper_slack_limit=upper_slack_limit)
        if name in self.neq_constraints:
            raise DuplicateNameException(f'A constraint with name \'{name}\' already exists. '
                                         f'You need to set a name, if you add multiple constraints.')
        self.neq_constraints.append(constraint)

    def add_inequality_constraint_vector(self,
                                         reference_velocities: Union[
                                             cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr_float]],
                                         lower_errors: Union[
                                             cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr_float]],
                                         upper_errors: Union[
                                             cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr_float]],
                                         weights: Union[
                                             cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr_float]],
                                         task_expression: Union[
                                             cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr]],
                                         names: List[str],
                                         lower_slack_limits: Optional[List[cas.symbol_expr_float]] = None,
                                         upper_slack_limits: Optional[List[cas.symbol_expr_float]] = None):
        """
        Calls add_constraint for a list of expressions.
        """
        if len(lower_errors) != len(upper_errors) \
                or len(lower_errors) != len(task_expression) \
                or len(lower_errors) != len(reference_velocities) \
                or len(lower_errors) != len(weights) \
                or (names is not None and len(lower_errors) != len(names)) \
                or (lower_slack_limits is not None and len(lower_errors) != len(lower_slack_limits)) \
                or (upper_slack_limits is not None and len(lower_errors) != len(upper_slack_limits)):
            raise GoalInitalizationException('All parameters must have the same length.')
        for i in range(len(lower_errors)):
            name_suffix = names[i] if names else None
            lower_slack_limit = lower_slack_limits[i] if lower_slack_limits else None
            upper_slack_limit = upper_slack_limits[i] if upper_slack_limits else None
            self.add_inequality_constraint(reference_velocity=reference_velocities[i],
                                           lower_error=lower_errors[i],
                                           upper_error=upper_errors[i],
                                           weight=weights[i],
                                           task_expression=task_expression[i],
                                           name=name_suffix,
                                           lower_slack_limit=lower_slack_limit,
                                           upper_slack_limit=upper_slack_limit)

    def add_equality_constraint_vector(self,
                                       reference_velocities: Union[
                                           cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr_float]],
                                       equality_bounds: Union[
                                           cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr_float]],
                                       weights: Union[
                                           cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr_float]],
                                       task_expression: Union[
                                           cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr]],
                                       names: List[str],
                                       lower_slack_limits: Optional[List[cas.symbol_expr_float]] = None,
                                       upper_slack_limits: Optional[List[cas.symbol_expr_float]] = None):
        """
        Calls add_constraint for a list of expressions.
        """
        for i in range(len(equality_bounds)):
            name_suffix = names[i] if names else None
            lower_slack_limit = lower_slack_limits[i] if lower_slack_limits else None
            upper_slack_limit = upper_slack_limits[i] if upper_slack_limits else None
            self.add_equality_constraint(reference_velocity=reference_velocities[i],
                                         equality_bound=equality_bounds[i],
                                         weight=weights[i],
                                         task_expression=task_expression[i],
                                         name=name_suffix,
                                         lower_slack_limit=lower_slack_limit,
                                         upper_slack_limit=upper_slack_limit)

    def add_point_goal_constraints(self,
                                   frame_P_current: cas.Point3,
                                   frame_P_goal: cas.Point3,
                                   reference_velocity: cas.symbol_expr_float,
                                   weight: cas.symbol_expr_float,
                                   name: str = ''):
        """
        Adds three constraints to move frame_P_current to frame_P_goal.
        Make sure that both points are expressed relative to the same frame!
        :param frame_P_current: a vector describing a 3D point
        :param frame_P_goal: a vector describing a 3D point
        :param reference_velocity: m/s
        :param weight:
        :param name:
        """
        frame_V_error = frame_P_goal - frame_P_current
        self.add_equality_constraint_vector(reference_velocities=[reference_velocity] * 3,
                                            equality_bounds=frame_V_error[:3],
                                            weights=[weight] * 3,
                                            task_expression=frame_P_current[:3],
                                            names=[f'{name}/x',
                                                   f'{name}/y',
                                                   f'{name}/z'])

    def add_position_constraint(self,
                                expr_current: Union[cas.symbol_expr, float],
                                expr_goal: Union[cas.symbol_expr_float, float],
                                reference_velocity: Union[cas.symbol_expr_float, float],
                                weight: Union[cas.symbol_expr_float, float] = WEIGHT_BELOW_CA,
                                name: str = ''):
        """
        A wrapper around add_constraint. Will add a constraint that tries to move expr_current to expr_goal.
        """
        error = expr_goal - expr_current
        self.add_equality_constraint(reference_velocity=reference_velocity,
                                     equality_bound=error,
                                     weight=weight,
                                     task_expression=expr_current,
                                     name=name)

    def add_position_range_constraint(self,
                                      expr_current: Union[cas.symbol_expr, float],
                                      expr_min: Union[cas.symbol_expr_float, float],
                                      expr_max: Union[cas.symbol_expr_float, float],
                                      reference_velocity: Union[cas.symbol_expr_float, float],
                                      weight: Union[cas.symbol_expr_float, float] = WEIGHT_BELOW_CA,
                                      name: str = ''):
        """
        A wrapper around add_constraint. Will add a constraint that tries to move expr_current to expr_goal.
        """
        error_min = expr_min - expr_current
        error_max = expr_max - expr_current
        self.add_inequality_constraint(reference_velocity=reference_velocity,
                                       lower_error=error_min,
                                       upper_error=error_max,
                                       weight=weight,
                                       task_expression=expr_current,
                                       name=name)

    def add_vector_goal_constraints(self,
                                    frame_V_current: cas.Vector3,
                                    frame_V_goal: cas.Vector3,
                                    reference_velocity: cas.symbol_expr_float,
                                    weight: cas.symbol_expr_float = WEIGHT_BELOW_CA,
                                    name: str = ''):
        """
        Adds constraints to align frame_V_current with frame_V_goal. Make sure that both vectors are expressed
        relative to the same frame and are normalized to a length of 1.
        :param frame_V_current: a vector describing a 3D vector
        :param frame_V_goal: a vector describing a 3D vector
        :param reference_velocity: rad/s
        :param weight:
        :param name:
        """
        angle = cas.save_acos(frame_V_current.dot(frame_V_goal))
        # avoid singularity by staying away from pi
        angle_limited = cas.min(cas.max(angle, -reference_velocity), reference_velocity)
        angle_limited = cas.save_division(angle_limited, angle)
        root_V_goal_normal_intermediate = cas.slerp(frame_V_current, frame_V_goal, angle_limited)

        error = root_V_goal_normal_intermediate - frame_V_current

        self.add_equality_constraint_vector(reference_velocities=[reference_velocity] * 3,
                                            equality_bounds=error[:3],
                                            weights=[weight] * 3,
                                            task_expression=frame_V_current[:3],
                                            names=[f'{name}/trans/x',
                                                   f'{name}/trans/y',
                                                   f'{name}/trans/z'])

    def add_rotation_goal_constraints(self,
                                      frame_R_current: cas.RotationMatrix,
                                      frame_R_goal: cas.RotationMatrix,
                                      reference_velocity: Union[cas.Symbol, float],
                                      weight: Union[cas.Symbol, float],
                                      name: str = ''):
        """
        Adds constraints to move frame_R_current to frame_R_goal. Make sure that both are expressed relative to the same
        frame.
        :param frame_R_current: current rotation as rotation matrix
        :param frame_R_goal: goal rotation as rotation matrix
        :param reference_velocity: rad/s
        :param weight:
        :param name:
        """
        # avoid singularity
        # the sign determines in which direction the robot moves when in singularity.
        # -0.0001 preserves the old behavior from before this goal was refactored
        hack = cas.RotationMatrix.from_axis_angle(cas.Vector3((0, 0, 1)), -0.0001)
        frame_R_current = frame_R_current.dot(hack)
        q_actual = frame_R_current.to_quaternion()
        q_goal = frame_R_goal.to_quaternion()
        q_goal = cas.if_less(q_goal.dot(q_actual), 0, -q_goal, q_goal)
        q_error = cas.quaternion_multiply(q_goal, cas.quaternion_conjugate(q_actual))

        # w is redundant
        self.add_equality_constraint_vector(reference_velocities=[reference_velocity] * 3,
                                            equality_bounds=-q_error[:3],
                                            weights=[weight] * 3,
                                            task_expression=q_error[:3],
                                            names=[f'{name}/rot/x',
                                                   f'{name}/rot/y',
                                                   f'{name}/rot/z'])

    def add_velocity_constraint(self,
                                lower_velocity_limit: Union[cas.symbol_expr_float, List[cas.symbol_expr_float]],
                                upper_velocity_limit: Union[cas.symbol_expr_float, List[cas.symbol_expr_float]],
                                weight: cas.symbol_expr_float,
                                task_expression: cas.symbol_expr,
                                velocity_limit: cas.symbol_expr_float,
                                name: Optional[str] = None,
                                lower_slack_limit: Union[cas.symbol_expr_float, List[cas.symbol_expr_float]] = -1e4,
                                upper_slack_limit: Union[cas.symbol_expr_float, List[cas.symbol_expr_float]] = 1e4):
        """
        Add a velocity constraint. Internally, this will be converted into multiple constraints, to ensure that the
        velocity stays within the given bounds.
        :param lower_velocity_limit:
        :param upper_velocity_limit:
        :param weight:
        :param task_expression:
        :param velocity_limit: Used for normalizing the expression, like reference_velocity, must be positive
        :param name:
        :param lower_slack_limit:
        :param upper_slack_limit:
        """
        name = name or ''
        constraint = DerivativeInequalityConstraint(constraint_name=name,
                                                    parent_task_name=self.name,
                                                    derivative=Derivatives.velocity,
                                                    expression=task_expression,
                                                    lower_limit=lower_velocity_limit,
                                                    upper_limit=upper_velocity_limit,
                                                    quadratic_weight=weight,
                                                    normalization_factor=velocity_limit,
                                                    lower_slack_limit=lower_slack_limit,
                                                    upper_slack_limit=upper_slack_limit)
        if constraint.name in self.derivative_constraints:
            raise KeyError(f'a constraint with name \'{name}\' already exists')
        self.derivative_constraints.append(constraint)

    def add_velocity_eq_constraint(self,
                                   velocity_goal: Union[cas.symbol_expr_float, List[cas.symbol_expr_float]],
                                   weight: cas.symbol_expr_float,
                                   task_expression: cas.symbol_expr,
                                   velocity_limit: cas.symbol_expr_float,
                                   name: Optional[str] = None,
                                   lower_slack_limit: Union[cas.symbol_expr_float, List[cas.symbol_expr_float]] = -1e4,
                                   upper_slack_limit: Union[cas.symbol_expr_float, List[cas.symbol_expr_float]] = 1e4):
        """
        Add a velocity constraint. Internally, this will be converted into multiple constraints, to ensure that the
        velocity stays within the given bounds.
        :param lower_velocity_limit:
        :param upper_velocity_limit:
        :param weight:
        :param task_expression:
        :param velocity_limit: Used for normalizing the expression, like reference_velocity, must be positive
        :param name:
        :param lower_slack_limit:
        :param upper_slack_limit:
        """
        name = name or ''
        constraint = DerivativeEqualityConstraint(constraint_name=name,
                                                  parent_task_name=self.name,
                                                  derivative=Derivatives.velocity,
                                                  expression=task_expression,
                                                  bound=velocity_goal,
                                                  quadratic_weight=weight,
                                                  normalization_factor=velocity_limit,
                                                  lower_slack_limit=lower_slack_limit,
                                                  upper_slack_limit=upper_slack_limit)
        if constraint.name in self.eq_derivative_constraints:
            raise KeyError(f'a constraint with name \'{name}\' already exists')
        self.eq_derivative_constraints.append(constraint)

    def add_velocity_eq_constraint_vector(self,
                                          velocity_goals: Union[
                                              cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr_float]],
                                          reference_velocities: Union[
                                              cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr_float]],
                                          weights: Union[
                                              cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr_float]],
                                          task_expression: Union[
                                              cas.Expression, cas.Vector3, cas.Point3, List[cas.symbol_expr]],
                                          names: List[str]):
        for i in range(len(velocity_goals)):
            name_suffix = names[i] if names else None
            self.add_velocity_eq_constraint(velocity_goal=velocity_goals[i],
                                            weight=weights[i],
                                            velocity_limit=reference_velocities[i],
                                            task_expression=task_expression[i],
                                            name=name_suffix,
                                            lower_slack_limit=-np.inf,
                                            upper_slack_limit=np.inf)

    def add_translational_velocity_limit(self,
                                         frame_P_current: cas.Point3,
                                         max_velocity: cas.symbol_expr_float,
                                         weight: cas.symbol_expr_float,
                                         max_violation: cas.symbol_expr_float = np.inf,
                                         name=''):
        """
        Adds constraints to limit the translational velocity of frame_P_current. Be aware that the velocity is relative
        to frame.
        :param frame_P_current: a vector describing a 3D point
        :param max_velocity:
        :param weight:
        :param max_violation: m/s
        :param name:
        """
        trans_error = cas.norm(frame_P_current)
        trans_error = cas.if_eq_zero(trans_error, 0.01, trans_error)
        god_map.debug_expression_manager.add_debug_expression('trans_error', trans_error)
        self.add_velocity_constraint(upper_velocity_limit=max_velocity,
                                     lower_velocity_limit=-max_velocity,
                                     weight=weight,
                                     task_expression=trans_error,
                                     lower_slack_limit=-max_violation,
                                     upper_slack_limit=max_violation,
                                     velocity_limit=max_velocity,
                                     name=f'{name}/vel')

    def add_rotational_velocity_limit(self,
                                      frame_R_current: cas.RotationMatrix,
                                      max_velocity: Union[cas.Symbol, float],
                                      weight: Union[cas.Symbol, float],
                                      max_violation: Union[cas.Symbol, float] = 1e4,
                                      name: str = ''):
        """
        Add velocity constraints to limit the velocity of frame_R_current. Be aware that the velocity is relative to
        frame.
        :param frame_R_current: Rotation matrix describing the current rotation.
        :param max_velocity: rad/s
        :param weight:
        :param max_violation:
        :param name:
        """
        root_Q_tipCurrent = frame_R_current.to_quaternion()
        angle_error = root_Q_tipCurrent.to_axis_angle()[1]
        self.add_velocity_constraint(upper_velocity_limit=max_velocity,
                                     lower_velocity_limit=-max_velocity,
                                     weight=weight,
                                     task_expression=angle_error,
                                     lower_slack_limit=-max_violation,
                                     upper_slack_limit=max_violation,
                                     name=f'{name}/q/vel',
                                     velocity_limit=max_velocity)
