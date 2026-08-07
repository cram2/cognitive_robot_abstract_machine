from time import perf_counter

import numpy as np

import pytest

from giskardpy.data_types.exceptions import NonPositiveRealTimeFactorError
from giskardpy.executor import (
    Executor,
    NoPacing,
    RealTimePacer,
    SimulationPacer,
)
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, EndMotion
from giskardpy.motion_statechart.monitors.payload_monitors import CountSeconds
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world import World


def test_simulation_pacer_timing_real_time(monkeypatch):
    pacer = SimulationPacer(real_time_factor=1.0)
    pacer.target_frequency = 50
    start_time = perf_counter()
    for i in range(50):
        pacer.sleep()
    assert np.isclose(perf_counter() - start_time, 1.0, rtol=0.01)


def test_simulation_pacer_timing_2x(monkeypatch):
    pacer = SimulationPacer(real_time_factor=2.0)
    pacer.target_frequency = 50
    start_time = perf_counter()
    for i in range(50):
        pacer.sleep()
    actual = perf_counter() - start_time
    assert np.isclose(actual, 0.5, rtol=0.01)


def test_simulation_pacer_timing_halfx(monkeypatch):
    pacer = SimulationPacer(real_time_factor=0.5)
    pacer.target_frequency = 50
    start_time = perf_counter()
    for i in range(50):
        pacer.sleep()
    assert np.isclose(perf_counter() - start_time, 2.0, rtol=0.01)


def test_no_pacing_does_not_wait():
    pacer = NoPacing()
    pacer.target_frequency = 50
    start_time = perf_counter()
    for i in range(50):
        pacer.sleep()
    assert perf_counter() - start_time < 0.01


def test_real_time_pacer_holds_the_target_frequency():
    pacer = RealTimePacer()
    pacer.target_frequency = 50
    start_time = perf_counter()
    for i in range(50):
        pacer.sleep()
    assert np.isclose(perf_counter() - start_time, 1.0, rtol=0.01)


def test_a_simulation_cannot_be_configured_to_stand_still():
    with pytest.raises(NonPositiveRealTimeFactorError):
        SimulationPacer(real_time_factor=0.0)


def test_with_executor():
    msc = MotionStatechart()
    msc.add_node(counter := CountSeconds(seconds=1.0))
    msc.add_node(EndMotion.when_true(counter))

    kin_sim = Executor(
        context=MotionStatechartContext(
            world=World(),
            qp_controller_config=QPControllerConfig.create_with_simulation_defaults(),
        ),
        pacer=SimulationPacer(real_time_factor=2.0),
    )
    kin_sim.compile(msc)
    kin_sim.tick_until_end(timeout=1000)
    # we tick 20 (hz) * 2 (real_time_factor) per second and sleep for 1s.
    # +2 because the endmotion needs to extra ticks
    assert kin_sim.control_cycles == 42
