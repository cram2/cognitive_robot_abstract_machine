from dataclasses import dataclass
from typing import Union

from giskardpy.god_map import god_map
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.qp.solvers.qp_solver_ids import SupportedQPSolver


@dataclass
class SetQPSolver(MotionStatechartNode):
    qp_solver_id: Union[SupportedQPSolver, int]

    def __post_init__(self):
        qp_solver_id = SupportedQPSolver(self.qp_solver_id)
        god_map.qp_controller.set_qp_solver(qp_solver_id)

    def __call__(self):
        self.state = True
