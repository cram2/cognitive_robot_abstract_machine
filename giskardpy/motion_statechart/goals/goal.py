from __future__ import annotations

from dataclasses import field, dataclass

from typing_extensions import List

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.motion_statechart.graph_node import MotionStatechartNode


@dataclass(eq=False, repr=False)
class Goal(MotionStatechartNode):
    nodes: List[MotionStatechartNode] = field(default_factory=list)
    _plot_style: str = field(default="filled", kw_only=True)
    _plot_shape: str = field(default="none", kw_only=True)

    def add_node(self, node: MotionStatechartNode) -> None:
        self.nodes.append(node)

    def arrange_in_sequence(self, nodes: List[MotionStatechartNode]) -> None:
        first_node = nodes[0]
        first_node.end_condition = first_node
        for node in nodes[1:]:
            node.start_condition = first_node
            node.end_condition = node
            first_node = node

    def apply_goal_conditions_to_children(self):
        for node in self.nodes:
            self.apply_start_condition_to_node(node)
            self.apply_pause_condition_to_node(node)
            self.apply_end_condition_to_node(node)
            self.apply_reset_condition_to_node(node)
            if isinstance(node, Goal):
                node.apply_goal_conditions_to_children()

    def apply_start_condition_to_node(self, node: MotionStatechartNode):
        if cas.is_trinary_true_symbol(node.start_condition):
            node.start_condition = self.start_condition

    def apply_pause_condition_to_node(self, node: MotionStatechartNode):
        if cas.is_trinary_false_symbol(node.pause_condition):
            node.pause_condition = node.pause_condition
        elif not cas.is_trinary_false_symbol(node.pause_condition):
            node.pause_condition = cas.trinary_logic_or(
                node.pause_condition, node.pause_condition
            )

    def apply_end_condition_to_node(self, node: MotionStatechartNode):
        if cas.is_trinary_false_symbol(node.end_condition):
            node.end_condition = self.end_condition
        elif not cas.is_trinary_false_symbol(self.end_condition):
            node.end_condition = cas.trinary_logic_or(
                node.end_condition, self.end_condition
            )

    def apply_reset_condition_to_node(self, node: MotionStatechartNode):
        if cas.is_trinary_false_symbol(node.reset_condition):
            node.reset_condition = node.reset_condition
        elif not cas.is_trinary_false_symbol(node.pause_condition):
            node.reset_condition = cas.trinary_logic_or(
                node.reset_condition, node.reset_condition
            )
