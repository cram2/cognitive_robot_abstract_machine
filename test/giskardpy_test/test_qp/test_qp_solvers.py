import logging

import pytest

from cramph.composites import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion
from cramph.statechart import Statechart
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.qp.solvers.qp_solver import QPSolver
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.robots.pr2 import PR2Joint
from giskardpy.motion_control import MotionControl
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor

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
    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[
            MotionControl(
                qp_controller_config=QPControllerConfig(
                    target_frequency=20,
                    prediction_horizon=7,
                    qp_solver_class=solver,
                )
            )
        ],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        sequence := Sequence(
            nodes=[
                JointPositionList(
                    goal_state=JointState.from_str_dict(
                        {PR2Joint.TORSO_LIFT: 0.1}, world=pr2_world_state_reset
                    )
                ),
                JointPositionList(
                    goal_state=JointState.from_str_dict(
                        {PR2Joint.TORSO_LIFT: 0.2}, world=pr2_world_state_reset
                    )
                ),
            ]
        )
    )
    statechart.add_node(EndMotion.when_true(sequence))

    executor.compile(statechart=statechart)
    executor.tick_until_end()
