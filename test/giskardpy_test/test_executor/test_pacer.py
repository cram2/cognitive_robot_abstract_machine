from time import perf_counter

import numpy as np

from giskardpy.executor import SimulationPacer, Executor
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


def test_simulation_pacer_timing_inf(monkeypatch):
    pacer = SimulationPacer(real_time_factor=None)
    pacer.target_frequency = 50
    start_time = perf_counter()
    for i in range(50):
        pacer.sleep()
    assert perf_counter() - start_time < 0.01


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
