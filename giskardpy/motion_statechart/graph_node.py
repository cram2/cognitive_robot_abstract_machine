from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import field, dataclass

from random_events.utils import SubclassJSONSerializer
from typing_extensions import (
    Dict,
    Any,
    Self,
    Optional,
    TYPE_CHECKING,
    List,
    TypeVar,
    Union,
)

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.god_map import god_map
from giskardpy.utils.utils import string_shortener
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World

if TYPE_CHECKING:
    from giskardpy.motion_statechart.motion_statechart import (
        MotionStatechart,
        ObservationState,
    )


@dataclass
class StateTransitionCondition:
    """Encapsulates both string and expression representations of a condition."""

    motion_statechart: MotionStatechart
    expression: cas.Expression = cas.TrinaryUnknown
    _parents: List[MotionStatechartNode] = field(default_factory=list, init=False)
    _child: MotionStatechartNode = field(default=None, init=False)

    @classmethod
    def create_true(cls, motion_statechart: MotionStatechart) -> Self:
        return cls(expression=cas.TrinaryTrue, motion_statechart=motion_statechart)

    @classmethod
    def create_false(cls, motion_statechart: MotionStatechart) -> Self:
        return cls(expression=cas.TrinaryFalse, motion_statechart=motion_statechart)

    @classmethod
    def create_unknown(cls, motion_statechart: MotionStatechart) -> Self:
        return cls(expression=cas.TrinaryUnknown, motion_statechart=motion_statechart)

    def update_expression(
        self, new_expression: cas.Expression, child: MotionStatechartNode
    ) -> None:
        self.expression = new_expression
        self._parents = [
            x
            for x in new_expression.free_symbols()
            if isinstance(x, MotionStatechartNode)
        ]
        self._child = child
        self.apply_to_motion_state_chart()

    def apply_to_motion_state_chart(self):
        self.motion_statechart.add_transition(self)

    def __str__(self):
        """
        Takes a logical expression, replaces the state symbols with monitor names and formats it nicely.
        """
        free_symbols = self.expression.free_symbols()
        if not free_symbols:
            return str(cas.is_true_symbol(self.expression))
        return cas.trinary_logic_to_str(self.expression)


@dataclass(repr=False, eq=False)
class MotionStatechartNode(cas.Symbol, SubclassJSONSerializer):
    name: PrefixedName = field(kw_only=True)

    motion_statechart: MotionStatechart = field(kw_only=True)
    """
    Back reference to the motion statechart that owns this node.
    """
    index: Optional[int] = field(default=None, init=False)
    """
    The index of the entity in `_world.kinematic_structure`.
    """

    life_cycle_symbol: cas.Symbol = field(init=False)
    observation_expression: cas.Expression = field(
        default_factory=lambda: cas.TrinaryUnknown, init=False
    )

    _plot: bool = field(default=True, kw_only=True)
    _plot_style: str = field(kw_only=True)
    _plot_shape: str = field(kw_only=True)
    _plot_extra_boarder_styles: List[str] = field(default_factory=list, kw_only=True)

    _start_condition: StateTransitionCondition = field(init=False)
    _pause_condition: StateTransitionCondition = field(init=False)
    _end_condition: StateTransitionCondition = field(init=False)
    _reset_condition: StateTransitionCondition = field(init=False)

    def __post_init__(self):
        self.life_cycle_symbol = cas.Symbol(
            name=PrefixedName("life_cycle", str(self.name))
        )
        self._start_condition = StateTransitionCondition.create_true(
            motion_statechart=self.motion_statechart
        )
        self._pause_condition = StateTransitionCondition.create_false(
            motion_statechart=self.motion_statechart
        )
        self._end_condition = StateTransitionCondition.create_false(
            motion_statechart=self.motion_statechart
        )
        self._reset_condition = StateTransitionCondition.create_false(
            motion_statechart=self.motion_statechart
        )
        self.motion_statechart.add_node(self)

    def resolve(self) -> float:
        return self.motion_statechart.observation_state[self]

    @property
    def world(self) -> World:
        return self.motion_statechart.world

    def __hash__(self):
        return hash(self.name)

    @property
    def start_condition(self) -> cas.Expression:
        return self._start_condition.expression

    @start_condition.setter
    def start_condition(self, expression: cas.Expression) -> None:
        self._start_condition.update_expression(expression, self)

    @property
    def pause_condition(self) -> cas.Expression:
        return self._pause_condition.expression

    @pause_condition.setter
    def pause_condition(self, expression: cas.Expression) -> None:
        self._pause_condition.update_expression(expression, self)

    @property
    def end_condition(self) -> cas.Expression:
        return self._end_condition.expression

    @end_condition.setter
    def end_condition(self, expression: cas.Expression) -> None:
        self._end_condition.update_expression(expression, self)

    @property
    def reset_condition(self) -> cas.Expression:
        return self._reset_condition.expression

    @reset_condition.setter
    def reset_condition(self, expression: cas.Expression) -> None:
        self._reset_condition.update_expression(expression, self)

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "name": self.name, "_plot": self._plot}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(name=data["name"], prefix=data["prefix"])

    def formatted_name(self, quoted: bool = False) -> str:
        formatted_name = string_shortener(
            original_str=str(self.name), max_lines=4, max_line_length=25
        )
        result = (
            f"{formatted_name}\n"
            f"----start_condition----\n"
            f"{str(self._start_condition)}\n"
            f"----pause_condition----\n"
            f"{str(self._pause_condition)}\n"
            f"----end_condition----\n"
            f"{str(self._end_condition)}\n"
            f"----reset_condition----\n"
            f"{str(self._end_condition)}"
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


GenericMotionStatechartNode = TypeVar(
    "GenericMotionStatechartNode", bound=MotionStatechartNode
)


@dataclass(eq=False, repr=False)
class Monitor(MotionStatechartNode):
    _plot_style: str = field(default="filled, rounded", kw_only=True)
    _plot_shape: str = field(default="rectangle", kw_only=True)


@dataclass(eq=False, repr=False)
class PayloadMonitor(Monitor, ABC):
    observation_expression: cas.Expression = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.observation_expression = cas.Expression(self)

    @abstractmethod
    def compute_observation(
        self,
    ) -> Union[
        ObservationState.TrinaryFalse,
        ObservationState.TrinaryTrue,
        ObservationState.TrinaryUnknown,
    ]: ...


@dataclass(eq=False)
class EndMotion(MotionStatechartNode):
    observation_expression: cas.Expression = field(
        default_factory=lambda: cas.TrinaryTrue, init=False
    )
    _plot_boarder_styles: List[str] = field(
        default_factory=lambda: ["rounded"], kw_only=True
    )
    _plot_style: str = field(default="filled, rounded", kw_only=True)
    _plot_shape: str = field(default="rectangle", kw_only=True)


@dataclass(eq=False)
class CancelMotion(MotionStatechartNode):
    exception: Exception = field(kw_only=True)
    observation_expression: cas.Expression = field(
        default_factory=lambda: cas.TrinaryTrue, init=False
    )
    _plot_extra_boarder_styles: List[str] = field(
        default_factory=lambda: ["dashed, rounded"], kw_only=True
    )
    _plot_style: str = field(default="filled, rounded", kw_only=True)
    _plot_shape: str = field(default="rectangle", kw_only=True)
