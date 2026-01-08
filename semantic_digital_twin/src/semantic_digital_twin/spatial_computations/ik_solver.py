from __future__ import annotations

from ctypes import c_int
from dataclasses import dataclass, field
from enum import Enum

import daqp
import numpy as np
from typing_extensions import Dict, TYPE_CHECKING, List, Tuple

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import (
    Vector,
    FloatVariable,
    Matrix,
    Scalar,
    VariableParameters,
)
from ..spatial_types import HomogeneousTransformationMatrix, RotationMatrix, Vector3
from ..world_description.degree_of_freedom import DegreeOfFreedom

if TYPE_CHECKING:
    from ..world import World
    from ..world_description.world_entity import KinematicStructureEntity

_large_value = np.inf
"""
Used as bounds for slack variables. 
Only needs to be changed when a different QP solver is used, as some can't handle inf.
"""


class IKSolverException(Exception):
    pass


class UnreachableException(IKSolverException):
    iterations: int
    """
    After how many iterations the solver converged.
    """

    def __init__(self, iterations: int):
        self.iterations = iterations
        super().__init__(
            f"Converged after {self.iterations}, but target pose not reached."
        )


class MaxIterationsException(IKSolverException):
    iterations: int
    """
    After how many iterations the solver did not converge.
    """

    def __init__(self, iterations: int):
        super().__init__(f"Failed to converge in {iterations} iterations.")


class DAQPSolverExitFlag(Enum):
    """
    Exit flags for the DAQP solver.
    """

    SOFT_OPTIMAL = (2, "Soft optimal")
    OPTIMAL = (1, "Optimal")
    INFEASIBLE = (-1, "Infeasible")
    CYCLING_DETECTED = (-2, "Cycling detected")
    UNBOUNDED_PROBLEM = (-3, "Unbounded problem")
    ITERATION_LIMIT_REACHED = (-4, "Iteration limit reached")
    NONCONVEX_PROBLEM = (-5, "Nonconvex problem")
    INITIAL_WORKING_SET_OVERDETERMINED = (-6, "Initial working set overdetermined")

    def __init__(self, code, description):
        self.code = code
        self.description = description

    @classmethod
    def from_code(cls, code):
        for flag in cls:
            if flag.code == code:
                return flag
        raise ValueError(f"Unknown exit flag code: {code}")


class QPSolverException(IKSolverException):
    def __init__(self, exit_flag_code):
        self.exit_flag = DAQPSolverExitFlag.from_code(exit_flag_code)
        super().__init__(
            f"QP solver failed with exit flag: {self.exit_flag.description}"
        )


@dataclass
class InverseKinematicsSolver:
    """
    Quadratic Programming-based Inverse Kinematics solver.

    This class handles the setup and solving of inverse kinematics problems
    using quadratic programming optimization.
    General idea:
    min_{dof_v, slack} low_weight * (dof_v1² + ... + dof_vn²) + high_weight * (slack_trans_x² + ... + slack_rot_z²)
    subject to
        position_error = J_translation * dof_v + translation slack
        rotation_error = J_rotation * dof_v + rotation slack
    where
        dof_v = velocity of dofs, e.g., joint velocity
        slack = auxiliary variables used to make constraints violable, in this case the remaining error that can not be achieved with the velocity.
        J_translation is the jacobian of the forward kinematics position.
        J_rotation is the jacobian of the forward kinematics rotation.
        position_error = target_position - current_position
        rotation_error = target_rotation - current_rotation

    The slack variables ensure that the QP stays solvable, even if the target velocity cannot be reached.
    Slack variables have a high weight, such that the solver only violates the constraints, if necessary.
    If we solve this QP at the current state, we get a velocity for the dofs, apply this to the current state with a dt and solve again.
    dof_v and slack below threshold:
        target reached
    dof_v below threshold, slack above threshold:
        target unreachable
    dof_v above threshold, slack below threshold:
        almost done
    dof_v above threshold, slack above threshold:
        neither close to the target, nor converged
    """

    world: World
    """
    Backreference to semantic digital twin.
    """

    iterations: int = field(default=-1, init=False)
    """
    The current iteration of the solver.
    """

    _convergence_velocity_tolerance: float = field(default=1e-4)
    """
    If the velocity of the active DOFs is below this threshold, the solver is considered to have converged.
    Unit depends on the DOF, e.g. rad/s for revolute joints or m/s for prismatic joints.
    """

    _convergence_slack_tolerance: float = field(default=1e-3)
    """
    The slack variables describe how much the target is violated. 
    If all slack variables are below this threshold, the solver found a solution.
    Unit is m for the position target or rad for the orientation target.
    """

    def solve(
        self,
        root: KinematicStructureEntity,
        tip: KinematicStructureEntity,
        target: HomogeneousTransformationMatrix,
        dt: float = 0.05,
        max_iterations: int = 200,
        translation_velocity: float = 0.2,
        rotation_velocity: float = 0.2,
    ) -> Dict[DegreeOfFreedom, float]:
        """
        Solve inverse kinematics problem.

        :param root: Root body of the kinematic chain
        :param tip: Tip body of the kinematic chain
        :param target: Desired tip pose relative to the root body
        :param dt: Time step for integration
        :param max_iterations: Maximum number of iterations
        :param translation_velocity: Maximum translation velocity
        :param rotation_velocity: Maximum rotation velocity
        :return: Dictionary mapping DOFs to their computed positions
        """
        target = root._world.transform(target, root)
        qp_problem = QPProblem(
            world=self.world,
            root=root,
            tip=tip,
            target=target,
            dt=dt,
            max_translation_velocity=translation_velocity,
            max_rotation_velocity=rotation_velocity,
        )

        # Initialize solver state
        solver_state = SolverState(
            position=np.array(
                [self.world.state[dof.id].position for dof in qp_problem.active_dofs]
            ),
            passive_position=np.array(
                [self.world.state[dof.id].position for dof in qp_problem.passive_dofs]
            ),
        )

        # Run iterative solver
        final_position = self._solve_iteratively(
            qp_problem, solver_state, dt, max_iterations
        )

        return {dof: final_position[i] for i, dof in enumerate(qp_problem.active_dofs)}

    def _solve_iteratively(
        self,
        qp_problem: QPProblem,
        solver_state: SolverState,
        dt: float,
        max_iterations: int,
    ) -> np.ndarray:
        """
        Tries to solve the inverse kinematics problem iteratively.
        :param qp_problem: Problem definition.
        :param solver_state: Initial state.
        :param dt: Step size per iteration. Unit is seconds.
                    Too large values can lead to instability, too small values can lead to slow convergence.
        :param max_iterations: Maximum number of iterations. A lower dt requires more iterations.
        :return: The final state after max_iterations.
        """
        for self.iteration in range(max_iterations):
            velocity, slack = self._solve_qp_step(qp_problem, solver_state)

            if self._check_convergence(velocity, slack):
                break

            solver_state.update_position(velocity, dt)
        else:
            raise MaxIterationsException(max_iterations)
        return solver_state.position

    def _solve_qp_step(
        self, qp_problem: QPProblem, solver_state: SolverState
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the QP matrices at the current state and solve the QP.
        :param qp_problem: Problem definition.
        :param solver_state: Current state
        :return: Velocities for the DOFs, and slack values.
        """

        # Evaluate QP matrices at current state
        qp_matrices = qp_problem.evaluate_at_state(solver_state)

        # Setup constraint sense (equality for last 6 constraints)
        sense = np.zeros(qp_matrices.l.shape, dtype=c_int)
        sense[-6:] = 5  # equality constraints

        # Solve QP
        (xstar, fval, exitflag, info) = daqp.solve(
            qp_matrices.H,
            qp_matrices.g,
            qp_matrices.A,
            qp_matrices.u,
            qp_matrices.l,
            sense,
        )

        if exitflag != 1:
            raise QPSolverException(exitflag)

        return (
            xstar[: len(qp_problem.active_variables)],
            xstar[len(qp_problem.active_variables) :],
        )

    def _check_convergence(self, velocity: np.ndarray, slack: np.ndarray) -> bool:
        """
        :param velocity: Current velocity of the DOFs.
        :param slack: Current slack values.
        :return: Whether the solver has converged.
        """
        vel_below_threshold = (
            np.max(np.abs(velocity)) < self._convergence_velocity_tolerance
        )
        slack_below_threshold = (
            np.max(np.abs(slack)) < self._convergence_slack_tolerance
        )
        if vel_below_threshold and slack_below_threshold:
            return True
        if vel_below_threshold and not slack_below_threshold:
            raise UnreachableException(self.iterations)
        return False


@dataclass
class QPProblem:
    """
    Represents a quadratic programming problem for inverse kinematics.
    """

    world: World
    """
    Backreference to semantic digital twin.
    """

    root: KinematicStructureEntity
    """
    Root body of the kinematic chain.
    """

    tip: KinematicStructureEntity
    """
    Tip body of the kinematic chain.
    """

    target: HomogeneousTransformationMatrix
    """
    Desired tip pose relative to the root body.
    """

    dt: float
    """
    Time step for integration.
    """

    max_translation_velocity: float
    max_rotation_velocity: float

    def __post_init__(self):
        # Extract DOFs and setup problem
        (
            self.active_dofs,
            self.passive_dofs,
            self.active_variables,
            self.passive_variables,
        ) = self._extract_dofs()
        self._setup_constraints()
        self._setup_weights()
        self._compile_functions()

    def _extract_dofs(
        self,
    ) -> Tuple[
        list[DegreeOfFreedom],
        list[DegreeOfFreedom],
        list[FloatVariable],
        list[FloatVariable],
    ]:
        """
        Extract active and passive DOFs from the kinematic chain.
        :return: Active Dofs, Passive Dofs, Active Variables, Passive Variables.
        """
        active_dofs_set = set()
        passive_dofs_set = set()
        root_to_common_link, common_link_to_tip = (
            self.world.compute_split_chain_of_connections(self.root, self.tip)
        )
        for connection in root_to_common_link + common_link_to_tip:
            active_dofs_set.update(connection.active_dofs)
            passive_dofs_set.update(connection.passive_dofs)

        active_dofs: List[DegreeOfFreedom] = list(
            sorted(active_dofs_set, key=lambda d: str(d.id))
        )
        passive_dofs: List[DegreeOfFreedom] = list(
            sorted(passive_dofs_set, key=lambda d: str(d.id))
        )

        active_variables = [dof.variables.position for dof in active_dofs]
        passive_variables = [dof.variables.position for dof in passive_dofs]

        return active_dofs, passive_dofs, active_variables, passive_variables

    def _setup_constraints(self):
        """Setup all constraints for the QP problem."""
        self.constraint_builder = ConstraintBuilder(
            self.world,
            self.root,
            self.tip,
            self.target,
            self.dt,
            self.max_translation_velocity,
            self.max_rotation_velocity,
        )

        # Box constraints
        self.lower_box_constraints, self.upper_box_constraints = (
            self.constraint_builder.build_box_constraints(self.active_dofs)
        )
        self.box_constraint_matrix = Matrix.eye(len(self.lower_box_constraints))

        # Goal constraints
        self.eq_bound_expr, self.neq_matrix = (
            self.constraint_builder.build_goal_constraints(self.active_variables)
        )

        # Combine constraints
        self.l = Matrix.vstack([self.lower_box_constraints, self.eq_bound_expr])
        self.u = Matrix.vstack([self.upper_box_constraints, self.eq_bound_expr])
        self.A = Matrix.vstack([self.box_constraint_matrix, self.neq_matrix])

    def _setup_weights(self):
        """Setup quadratic and linear weights for the QP problem."""
        dof_weights = [
            0.001 * (1.0 / min(1.0, dof.upper_limits.velocity)) ** 2
            for dof in self.active_dofs
        ]
        slack_weights = [2500 * (1.0 / 0.2) ** 2] * 6

        self.quadratic_weights = Matrix(dof_weights + slack_weights)
        self.linear_weights = Matrix.zeros(*self.quadratic_weights.shape)

    def _compile_functions(self):
        """Compile all symbolic expressions into functions."""
        variable_args = [self.active_variables, self.passive_variables]

        self.l_f = self.l.compile(VariableParameters.from_lists(*variable_args))
        self.u_f = self.u.compile(VariableParameters.from_lists(*variable_args))
        self.A_f = self.A.compile(VariableParameters.from_lists(*variable_args))
        self.quadratic_weights_f = self.quadratic_weights.compile(
            VariableParameters.from_lists(*variable_args)
        )
        self.linear_weights_f = self.linear_weights.compile(
            VariableParameters.from_lists(*variable_args)
        )

    def evaluate_at_state(self, solver_state) -> QPMatrices:
        """Evaluate QP matrices at the current solver state."""
        return QPMatrices(
            H=np.diag(
                self.quadratic_weights_f(
                    solver_state.position, solver_state.passive_position
                )
            ),
            g=self.linear_weights_f(
                solver_state.position, solver_state.passive_position
            ),
            A=self.A_f(solver_state.position, solver_state.passive_position),
            l=self.l_f(solver_state.position, solver_state.passive_position),
            u=self.u_f(solver_state.position, solver_state.passive_position),
        )


@dataclass
class ConstraintBuilder:
    """
    Builds constraints for the inverse kinematics QP problem.
    """

    world: World
    """
    Backreference to semantic digital twin.
    """

    root: KinematicStructureEntity
    """
    Root body of the kinematic chain.
    """

    tip: KinematicStructureEntity
    """
    Tip body of the kinematic chain.
    """

    target: HomogeneousTransformationMatrix
    """
    Desired tip pose relative to the root body.
    """

    dt: float
    """
    Time step for integration.
    """

    max_translation_velocity: float
    max_rotation_velocity: float

    maximum_velocity: float = field(default=1.0, init=False)
    """
    Used to limit the velocity of the DOFs, because the default values defined in the semantic digital twin are sometimes unreasonably high.
    """

    def build_box_constraints(
        self, active_dofs: List[DegreeOfFreedom]
    ) -> Tuple[Vector, Vector]:
        """Build position and velocity limit constraints for DOFs."""
        lower_constraints = []
        upper_constraints = []

        for dof in active_dofs:
            ll = max(-self.maximum_velocity, dof.lower_limits.velocity)
            ul = min(self.maximum_velocity, dof.upper_limits.velocity)

            if dof.has_position_limits():
                ll = sm.max(
                    dof.lower_limits.position - dof.variables.position, Scalar(ll)
                )
                ul = sm.min(
                    dof.upper_limits.position - dof.variables.position, Scalar(ul)
                )

            lower_constraints.append(ll)
            upper_constraints.append(ul)

        # Add slack variables
        global _large_value
        lower_constraints.extend([-_large_value] * 6)
        upper_constraints.extend([_large_value] * 6)

        return Vector(lower_constraints), Vector(upper_constraints)

    def build_goal_constraints(
        self, active_variables: List[FloatVariable]
    ) -> Tuple[Matrix, Matrix]:
        """Build position and rotation goal constraints."""
        root_T_tip = self.world.compose_forward_kinematics_expression(
            self.root, self.tip
        )

        # Position and rotation errors
        position_state, position_error = self._compute_position_error(root_T_tip)
        rotation_state, rotation_error = self._compute_rotation_error(root_T_tip)

        # Current state and jacobian
        current_expr = Matrix.vstack([position_state, rotation_state])
        eq_bound_expr = Matrix.vstack([position_error, rotation_error])

        J = current_expr.jacobian(active_variables)
        neq_matrix = Matrix.hstack([J * self.dt, Matrix.eye(6) * self.dt])

        return eq_bound_expr, neq_matrix

    def _compute_position_error(
        self, root_T_tip: HomogeneousTransformationMatrix
    ) -> Tuple[Vector, Vector]:
        """
        Compute position error with velocity limits.
        :param root_T_tip: Forward kinematics expression.
        :return: Expression describing the position, and the error vector.
        """
        root_P_tip = root_T_tip.to_position()
        root_T_tip_goal = HomogeneousTransformationMatrix(self.target)
        root_P_tip_goal = root_T_tip_goal.to_position()

        translation_cap = self.max_translation_velocity * self.dt
        position_error = root_P_tip_goal[:3] - root_P_tip[:3]

        for i in range(3):
            position_error[i] = sm.limit(
                position_error[i], -translation_cap, translation_cap
            )

        return root_P_tip[:3], position_error

    def _compute_rotation_error(
        self, root_T_tip: HomogeneousTransformationMatrix
    ) -> Tuple[Vector, Vector]:
        """
        Compute rotation error with velocity limits.
        :param root_T_tip: Forward kinematics expression.
        :return: Expression describing the rotation, and the error vector.
        """
        rotation_cap = self.max_rotation_velocity * self.dt

        hack = RotationMatrix.from_axis_angle(Vector3.Z(), -0.0001)
        root_R_tip = root_T_tip.to_rotation_matrix().dot(hack)
        q_actual = HomogeneousTransformationMatrix(self.target).to_quaternion()
        q_goal = root_R_tip.to_quaternion()
        q_goal = sm.if_less(q_goal.dot(q_actual), 0, -q_goal, q_goal)
        q_error = q_actual.diff(q_goal)

        rotation_error = -q_error
        for i in range(3):
            rotation_error[i] = sm.limit(rotation_error[i], -rotation_cap, rotation_cap)

        return q_error[:3], rotation_error[:3]


class SolverState:
    """
    Represents the state of the IK solver during iteration.
    """

    def __init__(self, position: np.ndarray, passive_position: np.ndarray):
        self.position = position
        self.passive_position = passive_position
        self.positions_history = []
        self.velocities_history = []

    def update_position(self, velocity: np.ndarray, dt: float):
        self.positions_history.append(self.position.copy())
        self.velocities_history.append(velocity.copy())
        self.position += velocity * dt


@dataclass
class QPMatrices:
    """
    Container for QP problem matrices at a specific state.
    min_{x} x^T H x + g^T x
    subject to
        l <= Ax <= u

    Find an x that minimizes the cost function, while satisfying the constraints.
    H (for Hessian) describes a quadratic cost function and g (for gradient) a linear one.
    Ax is the constraint space velocity, e.g. if A is the Jacobian, then Ax is the Cartesian velocity.
    l and u are the lower and upper bounds of the constraint space, e.g., translational or rotational velocity
    if l == u, we have equality constraints.
    """

    H: np.ndarray
    g: np.ndarray
    A: np.ndarray
    l: np.ndarray
    u: np.ndarray
