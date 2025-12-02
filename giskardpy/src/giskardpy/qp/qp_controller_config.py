from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Dict, Type

import numpy as np

from giskardpy.middleware import get_middleware
from giskardpy.qp.exceptions import QPSolverException
from giskardpy.qp.qp_formulation import QPFormulation
from giskardpy.qp.solvers.qp_solver import QPSolver
from giskardpy.qp.solvers.qp_solver_ids import SupportedQPSolver
from giskardpy.utils.utils import get_all_classes_in_module
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.spatial_types.derivatives import Derivatives
import logging

logger = logging.getLogger(__name__)

available_solvers: Dict[SupportedQPSolver, Type[QPSolver]] = {}


def detect_solvers():
    global available_solvers
    qp_solver_class: Type[QPSolver]
    for qp_solver_name in SupportedQPSolver:
        module_name = f"giskardpy.qp.solvers.qp_solver_{qp_solver_name.name}"
        try:
            qp_solver_class = list(
                get_all_classes_in_module(module_name, QPSolver).items()
            )[0][1]
            available_solvers[qp_solver_name] = qp_solver_class
        except Exception:
            continue
    solver_names = [solver_name.name for solver_name in available_solvers.keys()]
    print(f"Found these qp solvers: {solver_names}")


detect_solvers()


@dataclass
class QPControllerConfig:
    control_dt: Optional[float]
    """
    The time step of the control loop.
    A lower value will result in a smoother control signal, but the QP will have to be solved more often.
    If the value is too large, the QP might start running into infeasiblity issues.

    On a real robot:
        Pick a value equal to or below the frequency at which we get feedback.
        Computing control commands at a higher frequency than the robot can provide feedback can result in instability.
        If you cannot match the frequency due to hardware limitations, pick one that is as close to it as possible.

    In simulation:
        Pick 0.05. It is low enough to be stable and high enough for quick simulations.
    """

    dof_weights: Dict[PrefixedName, DerivativeMap[float]] = field(
        default_factory=lambda: defaultdict(
            lambda: DerivativeMap([None, 0.01, None, None])
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
    Decides how much the dof_weights decrease over the prediction horizon.
    .. warning:: Only change if you really know what you are doing.
    """

    max_derivative: Derivatives = field(default=Derivatives.jerk)
    """
    The highest derivative that will be considered in the QP formulation.
    ..warning:: Only change if you really know what you are doing.
    """

    qp_solver_id: SupportedQPSolver = field(default=SupportedQPSolver.qpSWIFT)
    """
    The solver used for solving the QP formulation.
    .. warning:: only qpSWIFT is well tested.
    """

    prediction_horizon: int = field(default=7)
    """
    The prediction horizon used for the QP formulation.
    Increasing this value will:
        - make the commands produced by Giskard smoother
        - increase the computational cost of the controller.
    You'll want a value that is as high as necessary and as low as possible.
    .. warning:: Minimum value is 4, otherwise it becomes impossible to integrate jerk into the QP formulation.
    """

    qp_formulation: Optional[QPFormulation] = field(default_factory=QPFormulation)
    """
    Changes the formulation of the QP problem.
    Check QPFormulation for more information.
    """

    retries_with_relaxed_constraints: int = field(default=5)
    """
    If the QP insolvable, the constraints will be relaxed with high weight slack variables 
    up to 'retries_with_relaxed_constraints' many times.
    """

    verbose: bool = field(default=True)
    """
    If True, prints config.
    """

    # %% init false
    mpc_dt: Optional[float] = field(init=False)
    """
    The time step of the MPC.
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

    qp_solver_class: Type[QPSolver] = field(init=False)
    """
    Reference to the resolved QP solver class.
    """

    def __post_init__(self):
        self.mpc_dt = self.control_dt
        if not self.qp_formulation.is_mpc:
            self.prediction_horizon = 1
            self.max_derivative = Derivatives.velocity

        if self.prediction_horizon < 4:
            raise ValueError("prediction horizon must be >= 4.")
        self.set_qp_solver()

    @classmethod
    def create_default_with_20hz(cls):
        return cls(
            control_dt=0.05,
            prediction_horizon=7,
        )

    def set_qp_solver(self) -> None:
        if self.qp_solver_id is not None:
            self.qp_solver_class = available_solvers[self.qp_solver_id]
        else:
            for qp_solver_id in SupportedQPSolver:
                if qp_solver_id in available_solvers:
                    self.qp_solver_class = available_solvers[qp_solver_id]
                    break
            else:
                raise QPSolverException(f"No qp solver found")
            self.qp_solver_id = self.qp_solver_class.solver_id
        get_middleware().loginfo(
            f'QP Solver set to "{self.qp_solver_class.solver_id.name}"'
        )

    def set_dof_weight(
            self, dof_name: PrefixedName, derivative: Derivatives, weight: float
    ):
        """Set weight for a specific DOF derivative."""
        self.dof_weights[dof_name].data[derivative] = weight

    def set_dof_weights(self, dof_name: PrefixedName, weight_map: DerivativeMap[float]):
        """Set multiple weights for a DOF."""
        self.dof_weights[dof_name] = weight_map

    def get_dof_weight(self, dof_name: PrefixedName, derivative: Derivatives) -> float:
        """Get weight for a specific DOF derivative."""
        return self.dof_weights[dof_name].data[derivative]
