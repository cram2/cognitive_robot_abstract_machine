import logging

import pytest

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.qp.solvers.qp_solver import QPSolver
from semantic_digital_twin.datastructures.joint_state import JointState

logger = logging.getLogger(__name__)

installed_qp_solvers: list[type[QPSolver]] = []

try:
    from giskardpy.qp.solvers.qp_solver_qpSWIFT import QPSolverQPSwift

    installed_qp_solvers.append(QPSolverQPSwift)
except Exception as e:
    logger.warning(f"Could not import QP solver: {e}")

try:
    from giskardpy.qp.solvers.qp_solver_gurobi import QPSolverGurobi

    installed_qp_solvers.append(QPSolverGurobi)
except Exception as e:
    logger.warning(f"Could not import QP solver: {e}")

try:
    from giskardpy.qp.solvers.qp_solver_qpalm import QPSolverQPalm

    installed_qp_solvers.append(QPSolverQPalm)
except Exception as e:
    logger.warning(f"Could not import QP solver: {e}")

try:
    from giskardpy.qp.solvers.qp_solver_piqp import QPSolverPIQP

    installed_qp_solvers.append(QPSolverPIQP)
except Exception as e:
    logger.warning(f"Could not import QP solver: {e}")


@pytest.mark.parametrize("solver", installed_qp_solvers)
def test_joint_goal(solver, pr2_world_state_reset):
    msc = MotionStatechart()
    msc.add_node(
        sequence := Sequence(
            [
                JointPositionList(
                    goal_state=JointState.from_str_dict(
                        {"torso_lift_joint": 0.1}, world=pr2_world_state_reset
                    )
                ),
                JointPositionList(
                    goal_state=JointState.from_str_dict(
                        {"torso_lift_joint": 0.2}, world=pr2_world_state_reset
                    )
                ),
            ]
        )
    )
    msc.add_node(EndMotion.when_true(sequence))

    kin_sim = Executor(
        MotionStatechartContext(
            world=pr2_world_state_reset,
            qp_controller_config=QPControllerConfig(
                target_frequency=20,
                prediction_horizon=7,
                qp_solver_class=solver,
            ),
        )
    )
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()
