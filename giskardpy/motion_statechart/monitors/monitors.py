from __future__ import annotations

import abc
from abc import ABC
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional
from line_profiler import profile

import numpy as np

import semantic_world.spatial_types.spatial_types as cas
from giskardpy.data_types.data_types import Derivatives, PrefixName, ObservationState
from giskardpy.data_types.exceptions import GiskardException, MonitorInitalizationException
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from semantic_world.spatial_types.symbol_manager import symbol_manager


@dataclass
class Monitor(MotionStatechartNode):
    obs_symbol: cas.Symbol = field(init=False)
    life_symbol: cas.Symbol = field(init=False)

    @cached_property
    def observation_state_symbol(self) -> cas.Symbol:
        symbol_name = f'{self.name}.observation_state'
        return symbol_manager.register_symbol_provider(symbol_name,
                                                       lambda
                                                           name=self.name: god_map.motion_statechart_manager.monitor_state.get_observation_state(
                                                           name))

    @cached_property
    def life_cycle_state_symbol(self) -> cas.Symbol:
        symbol_name = f'{self.name}.life_cycle_state'
        return symbol_manager.register_symbol_provider(symbol_name, lambda
            name=self.name: god_map.motion_statechart_manager.monitor_state.get_life_cycle_state(name))


@dataclass
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


@dataclass
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


@dataclass
class EndMotion(PayloadMonitor):

    def __call__(self):
        self.state = ObservationState.true


@dataclass
class CancelMotion(PayloadMonitor):
    exception: Exception = field(default_factory=GiskardException)

    def __call__(self):
        self.state = ObservationState.true
        raise self.exception


@dataclass
class LocalMinimumReached(Monitor):
    min_cut_off: float = 0.01
    max_cut_off: float = 0.06
    joint_convergence_threshold: float = 0.01
    windows_size: int = 1

    def __post_init__(self):
        ref = []
        symbols = []
        for free_variable in god_map.world.active_degrees_of_freedom:
            velocity_limit = god_map.qp_controller.config.dof_upper_limits_overwrite[free_variable.name].velocity
            if free_variable.upper_limits.velocity is not None:
                velocity_limit = min(velocity_limit, free_variable.upper_limits.velocity)
            velocity_limit *= self.joint_convergence_threshold
            velocity_limit = min(max(self.min_cut_off, velocity_limit), self.max_cut_off)
            ref.append(velocity_limit)
            symbols.append(free_variable.symbols.velocity)
        ref = cas.Expression(ref)
        vel_symbols = cas.Expression(symbols)

        traj_longer_than_1_sec = cas.greater(god_map.time_symbol, 1)
        self.observation_expression = cas.logic_and(traj_longer_than_1_sec,
                                                    cas.logic_all(cas.less(vel_symbols, ref)))


@dataclass
class TimeAbove(Monitor):
    threshold: float

    def __post_init__(self):
        traj_length_in_sec = god_map.time_symbol
        condition = cas.greater(traj_length_in_sec, self.threshold)
        self.observation_expression = condition


class Alternator(Monitor):

    def __init__(self,
                 name: Optional[str] = None,
                 mod: int = 2,
                 plot: bool = True):
        super().__init__(name=name,
                         plot=plot)
        time = god_map.time_symbol
        expr = cas.equal(cas.fmod(cas.floor(time), mod), 0)
        self.observation_expression = expr


class TrueMonitor(Monitor):
    def __init__(self,
                 name: Optional[str] = None,
                 plot: bool = True):
        super().__init__(name=name,
                         plot=plot)
        self.observation_expression = cas.BinaryTrue


class FalseMonitor(Monitor):
    def __init__(self,
                 name: Optional[str] = None,
                 plot: bool = True):
        super().__init__(name=name,
                         plot=plot)
        self.observation_expression = cas.BinaryFalse
