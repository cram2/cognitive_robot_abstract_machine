from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, Type

from typing_extensions import TYPE_CHECKING

from giskardpy.qp.exceptions import BrakingTimeExceedsHorizonError
from giskardpy.qp.jerk_limited_braking import JerkLimitedBraking
from giskardpy.qp.solvers.qp_solver_piqp import QPSolverPIQP
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.spatial_types.derivatives import Derivatives

if TYPE_CHECKING:
    from giskardpy.qp.solvers.qp_solver import QPSolver

logger = logging.getLogger(__name__)

NUMBER_OF_RESTING_STEPS = 2
"""
Number of final prediction horizon steps whose velocity is fixed at zero, so that every
plan ends at rest.
"""


@dataclass
class QPControllerConfig:
    """
    Configuration for the QPController.

    .. warning:: Giskard relies on the robot tracking velocity commands well. Make sure it does!
    .. note::
    Tuning works the following way:
        1. Look at the frequency you get feedback from the robot and choose a frequency slightly below it.
            e.g. joint_states publishes at 100hz -> start with 90hz for the controller.
        2. Leave the prediction horizon unset, so it is derived from the braking time.
        3. If the robot is NOT moving smoothly, increase the braking time until it does.
        4. If Giskard cannot keep up with the frequency, reduce hz and go back to step 2.
    """

    target_frequency: float
    """
    Target frequency of the control loop in Hz.

    A higher value will result in a more responsive and thus smoother control signal, but the QP will have to be solved more
    often per second. If the value is too low, the QP might start running into
    infeasiblity issues.

    .. note::
    On a real robot:
        Pick a value equal to or below the frequency at which we get feedback.
        Computing control commands at a higher frequency than the robot can provide feedback can result in instability.
        If you cannot match the frequency due to hardware limitations, pick one that is as close to it as possible.

    .. note::
    In simulation:
        Pick 20. It is high enough to be stable and low enough for quick simulations.
    """

    braking_time: float = field(default=0.3)
    """
    Time, in seconds, a degree of freedom without a jerk limit of its own takes to brake
    from its velocity limit to rest.

    It sets that degree of freedom's jerk limit to ``4 * velocity_limit / braking_time**2``,
    independent of the control frequency. Increasing it makes the motion smoother and
    less aggressive, and lengthens the derived prediction horizon.
    """

    prediction_horizon: int | None = field(default=None)
    """
    The prediction horizon in time steps used for the QP formulation.

    Each step will have a length of 1/hz, meaning the prediction horizon in seconds is
    prediction_horizon / hz. Every plan must come to rest within it, so it has to cover
    the braking time. ``None`` derives the shortest horizon that does.

    .. note:: Larger values increase the computational cost of the controller and slow
        down tracking of moving goals.
    .. warning:: Minimum value is 4, otherwise it becomes impossible to integrate jerk into the QP formulation.
    """

    dof_weights: Dict[PrefixedName, DerivativeMap[float]] = field(
        default_factory=lambda: defaultdict(
            lambda: DerivativeMap(None, 0.01, None, None)
        )
    )
    """
    Weights for the derivatives of the DOFs.

    A lower weight for a dof will make it cheaper for Giskard to use it.
    If you think Giskard is using a certain DOF too much, you can increase its weight here.
    .. warning:: If you increase the weights too much, Giskard might prefer violating goals over moving Dofs.
    """

    horizon_weight_gain_scalar: float = 0.1
    """
    Fraction of the dof_weights applied at the first step of the prediction horizon.

    The weights grow linearly from this fraction to the full weight at the end of the
    horizon, so the controller prefers using the early steps, whose commands are the
    ones that get executed.

    .. warning:: Only change if you really know what you are doing.
    """

    max_derivative: Derivatives = field(default=Derivatives.jerk)
    """
    The highest derivative that will be considered in the QP formulation.

    ..warning:: Only change if you really know what you are doing.
    """

    verbose: bool = field(default=True)
    """
    If True, prints config.
    """

    # %% init false
    model_predictive_control_time_step: float = field(init=False)
    """
    The time step of the MPC in seconds.

    control_dt == mpc_dt:
        default
    control_dt > mpc_dt:
        The control commands apply over longer intervals than expected, almost guaranteeing overshoot or in stability.
    control_dt < mpc_dt:
        The MPC formulation underestimates real kinematics based on mpc_dt. If the control loop runs faster,
        the actual system evolves more frequently, potentially causing overshooting as velocity
        integrals exceed the controller’s estimate. In extreme cases, QPs may become infeasible due to excessive
        velocity/acceleration demands.
    .. warning:: Don't change this.
    """

    qp_solver_class: Type[QPSolver] = field(default=QPSolverPIQP)
    """
    Reference to the resolved QP solver class.
    """

    def __post_init__(self):
        if self.target_frequency < 20:
            logging.warning(
                f"Hertz ({self.target_frequency}) is below 20Hz. This might cause instability."
            )
        self.model_predictive_control_time_step = self.control_dt

        minimum_prediction_horizon = (
            self.number_of_braking_steps + NUMBER_OF_RESTING_STEPS
        )
        if self.prediction_horizon is None:
            self.prediction_horizon = minimum_prediction_horizon
        if self.prediction_horizon < 4:
            raise ValueError("prediction horizon must be >= 4.")
        if self.prediction_horizon < minimum_prediction_horizon:
            raise BrakingTimeExceedsHorizonError(
                prediction_horizon=self.prediction_horizon,
                minimum_prediction_horizon=minimum_prediction_horizon,
                braking_time=self.braking_time,
                time_step=self.control_dt,
            )

    @cached_property
    def control_dt(self) -> float:
        """
        Time step of the control loop in seconds.
        """
        return 1 / self.target_frequency

    @property
    def number_of_braking_steps(self) -> int:
        """
        Number of time steps a degree of freedom without a jerk limit of its own needs to
        brake from its velocity limit to rest.
        """
        return JerkLimitedBraking.number_of_steps_for_braking_time(
            braking_time=self.braking_time, time_step=self.control_dt
        )

    @property
    def control_horizon(self) -> int:
        """
        Number of time steps over which commands are applied, fewer than the prediction
        horizon by the final steps that only bring the system to rest.
        """
        return self.prediction_horizon - NUMBER_OF_RESTING_STEPS

    @classmethod
    def create_with_simulation_defaults(cls):
        """
        Creates a configuration with the default values used for kinematic simulation.
        """
        return cls(target_frequency=20)

    def set_dof_weight(
        self, dof_name: PrefixedName, derivative: Derivatives, weight: float
    ):
        """
        Sets the objective weight of a single degree-of-freedom derivative.
        """
        self.dof_weights[dof_name][derivative] = weight

    def set_dof_weights(self, dof_name: PrefixedName, weight_map: DerivativeMap[float]):
        """
        Sets the objective weights of all derivatives of a degree of freedom.
        """
        self.dof_weights[dof_name] = weight_map

    def get_degree_of_freedom_weight(
        self, dof_name: PrefixedName, derivative: Derivatives
    ) -> float:
        """
        Returns the objective weight of a single degree-of-freedom derivative.
        """
        return self.dof_weights[dof_name][derivative]
