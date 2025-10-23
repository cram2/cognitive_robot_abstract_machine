from __future__ import annotations

import abc
from abc import ABC
from dataclasses import field
from functools import cached_property

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.data_types.exceptions import GiskardException
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.data_types import ObservationState
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.utils.decorators import validated_dataclass
from semantic_digital_twin.spatial_types.symbol_manager import symbol_manager


@validated_dataclass
class Monitor(MotionStatechartNode):

    @cached_property
    def observation_state_symbol(self) -> cas.Symbol:
        symbol_name = f"{self.name}.observation_state"
        return symbol_manager.register_symbol_provider(
            symbol_name,
            lambda name=self.name: god_map.motion_statechart_manager.monitor_state.get_observation_state(
                name
            ),
        )

    @cached_property
    def life_cycle_state_symbol(self) -> cas.Symbol:
        symbol_name = f"{self.name}.life_cycle_state"
        return symbol_manager.register_symbol_provider(
            symbol_name,
            lambda name=self.name: god_map.motion_statechart_manager.monitor_state.get_life_cycle_state(
                name
            ),
        )


@validated_dataclass
class PayloadMonitor(Monitor, ABC):
    """
    A monitor which executes its __call__ function when start_condition becomes True.
    Subclass this and implement __init__.py and __call__. The __call__ method should change self.state to True when
    it's done.
    """

    state: ObservationState = field(init=False, default=ObservationState.unknown)

    @abc.abstractmethod
    def __call__(self):
        pass


@validated_dataclass
class ThreadedPayloadMonitor(Monitor, ABC):
    """
    A monitor which executes its __call__ function when start_condition becomes True.
    Subclass this and implement __init__.py and __call__. The __call__ method should change self.state to True when
    it's done.
    Calls __call__ in a separate thread. Use for expensive operations
    """

    state: ObservationState = field(init=False, default=ObservationState.unknown)

    @abc.abstractmethod
    def __call__(self):
        pass


@validated_dataclass
class EndMotion(PayloadMonitor):

    def __call__(self):
        self.state = ObservationState.true


@validated_dataclass
class CancelMotion(PayloadMonitor):
    exception: Exception = field(default_factory=GiskardException)

    def __call__(self):
        self.state = ObservationState.true
        raise self.exception


@validated_dataclass
class LocalMinimumReached(Monitor):
    min_cut_off: float = 0.01
    max_cut_off: float = 0.06
    joint_convergence_threshold: float = 0.01
    windows_size: int = 1

    def __post_init__(self):
        ref = []
        symbols = []
        for dof in god_map.world.active_degrees_of_freedom:
            velocity_limit = dof.upper_limits.velocity
            velocity_limit *= self.joint_convergence_threshold
            velocity_limit = min(
                max(self.min_cut_off, velocity_limit), self.max_cut_off
            )
            ref.append(velocity_limit)
            symbols.append(dof.symbols.velocity)
        ref = cas.Expression(ref)
        vel_symbols = cas.Expression(symbols)

        traj_longer_than_1_sec = god_map.time_symbol > 1
        self.observation_expression = cas.logic_and(
            traj_longer_than_1_sec, cas.logic_all(cas.abs(vel_symbols) < ref)
        )


@validated_dataclass
class TimeAbove(Monitor):
    threshold: float

    def __post_init__(self):
        traj_length_in_sec = god_map.time_symbol
        condition = traj_length_in_sec > self.threshold
        self.observation_expression = condition


@validated_dataclass
class Alternator(Monitor):
    mod: int = 2

    def __post_init__(self):
        time = god_map.time_symbol
        expr = cas.fmod(cas.floor(time), self.mod) == 0
        self.observation_expression = expr


@validated_dataclass
class TrueMonitor(Monitor):
    def __post_init__(self):
        self.observation_expression = cas.BinaryTrue


@validated_dataclass
class FalseMonitor(Monitor):
    def __post_init__(self):
        self.observation_expression = cas.BinaryFalse
