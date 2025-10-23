from __future__ import annotations

from dataclasses import field
from functools import cached_property
from typing import Optional, Union

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.god_map import god_map
from giskardpy.utils.decorators import validated_dataclass
from giskardpy.utils.utils import string_shortener, quote_node_names


@validated_dataclass
class MotionStatechartNode:
    name: str

    _plot: bool = field(default=True, kw_only=True)
    _observation_expression: cas.Expression = field(
        default_factory=lambda: cas.TrinaryUnknown, init=False
    )

    _unparsed_start_condition: Optional[str] = field(default=None, init=False)
    _unparsed_pause_condition: Optional[str] = field(default=None, init=False)
    _unparsed_end_condition: Optional[str] = field(default=None, init=False)
    _unparsed_reset_condition: Optional[str] = field(default=None, init=False)

    logic3_start_condition: Optional[cas.Expression] = field(default=None, init=False)
    logic3_pause_condition: Optional[cas.Expression] = field(default=None, init=False)
    logic3_end_condition: Optional[cas.Expression] = field(default=None, init=False)
    logic3_reset_condition: Optional[cas.Expression] = field(default=None, init=False)

    def __str__(self):
        return self.name

    def set_unparsed_conditions(
        self,
        start_condition: Optional[str] = None,
        pause_condition: Optional[str] = None,
        end_condition: Optional[str] = None,
        reset_condition: Optional[str] = None,
    ):
        if start_condition is not None:
            self._unparsed_start_condition = start_condition
        if pause_condition is not None:
            self._unparsed_pause_condition = pause_condition
        if end_condition is not None:
            self._unparsed_end_condition = end_condition
        if reset_condition is not None:
            self._unparsed_reset_condition = reset_condition

    def set_conditions(
        self,
        start_condition: cas.Expression,
        pause_condition: cas.Expression,
        end_condition: cas.Expression,
        reset_condition: cas.Expression,
    ):
        self.logic3_start_condition = start_condition
        self.logic3_pause_condition = pause_condition
        self.logic3_end_condition = end_condition
        self.logic3_reset_condition = reset_condition

    def __eq__(self, other: MotionStatechartNode) -> bool:
        return self.name == other.name

    def formatted_name(self, quoted: bool = False) -> str:
        formatted_name = string_shortener(
            original_str=str(self.name), max_lines=4, max_line_length=25
        )
        result = (
            f"{formatted_name}\n"
            f"----start_condition----\n"
            f"{god_map.motion_statechart_manager.format_condition(self.start_condition)}\n"
            f"----pause_condition----\n"
            f"{god_map.motion_statechart_manager.format_condition(self.pause_condition)}\n"
            f"----end_condition----\n"
            f"{god_map.motion_statechart_manager.format_condition(self.end_condition)}"
        )
        if quoted:
            return '"' + result + '"'
        return result

    def update_expression_on_starting(
        self, expression: cas.GenericSymbolicType
    ) -> cas.GenericSymbolicType:
        if len(expression.free_symbols()) == 0:
            return expression
        return god_map.motion_statechart_manager.register_expression_updater(
            expression, self
        )

    @property
    def observation_expression(self) -> cas.Expression:
        return self._observation_expression

    @observation_expression.setter
    def observation_expression(self, expression: cas.Expression) -> None:
        self._observation_expression = expression

    @cached_property
    def observation_state_symbol(self) -> cas.Symbol:
        raise NotImplementedError("observation_state property is not implemented")

    @cached_property
    def life_cycle_state_symbol(self) -> cas.Symbol:
        raise NotImplementedError("life_cycle_state property is not implemented")

    @property
    def start_condition(self) -> str:
        if self._unparsed_start_condition is None:
            return "True"
        return quote_node_names(self._unparsed_start_condition)

    @start_condition.setter
    def start_condition(self, value: Union[str, MotionStatechartNode]) -> None:
        if isinstance(value, MotionStatechartNode):
            value = value.name
        if value == "":
            value = "True"
        self._unparsed_start_condition = value

    @property
    def pause_condition(self) -> str:
        if self._unparsed_pause_condition is None:
            return "False"
        return quote_node_names(self._unparsed_pause_condition)

    @pause_condition.setter
    def pause_condition(self, value: Union[str, MotionStatechartNode]) -> None:
        if isinstance(value, MotionStatechartNode):
            value = value.name
        if value == "":
            value = "False"
        self._unparsed_pause_condition = value

    @property
    def end_condition(self) -> str:
        if self._unparsed_end_condition is None:
            return "False"
        return quote_node_names(self._unparsed_end_condition)

    @end_condition.setter
    def end_condition(self, value: Union[str, MotionStatechartNode]) -> None:
        if isinstance(value, MotionStatechartNode):
            value = value.name
        if value == "":
            value = "False"
        self._unparsed_end_condition = value

    @property
    def reset_condition(self) -> str:
        if self._unparsed_reset_condition is None:
            return "False"
        return quote_node_names(self._unparsed_reset_condition)

    @reset_condition.setter
    def reset_condition(self, value: Union[str, MotionStatechartNode]) -> None:
        if isinstance(value, MotionStatechartNode):
            value = value.name
        if value == "":
            value = "False"
        self._unparsed_reset_condition = value
