from dataclasses import dataclass

import numpy as np
import rustworkx as rx
from typing_extensions import List

from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
    StateTransitionCondition,
)


@dataclass
class MotionStatechartGraph:
    rx_graph: rx.PyDiGraph[MotionStatechartNode]
    life_cycle_state: np.ndarray
    observable_state: np.ndarray

    def add_node(self, node: MotionStatechartNode):
        self.rx_graph.add_node(node)

    def add_transition(self, transition: StateTransitionCondition):
        pass

    def compile(self):
        pass

    def transition(self):
        pass

    def evaluate(self):
        pass
