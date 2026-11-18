"""
Tests for timing the control cycle on the ROS clock.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from rclpy.clock import ROSClock
from rclpy.time import Time
from rclpy.time_source import TimeSource

from giskardpy.middleware.ros2.cycle_timing import CycleTiming

# %% a simulated clock


@dataclass(kw_only=True)
class SimulatedClock:
    """
    A ROS clock that only advances when a test sets it, as under ``use_sim_time``.
    """

    clock: ROSClock
    """
    The clock the timed code reads.
    """

    time_source: TimeSource
    """
    Keeps ROS time active on :attr:`clock`.
    """

    def set_seconds(self, *, seconds: float) -> None:
        """
        Let the clock report ``seconds`` from now on.
        """
        self.clock.set_ros_time_override(
            Time(seconds=seconds, clock_type=self.clock.clock_type)
        )


@pytest.fixture()
def simulated_clock() -> SimulatedClock:
    """
    A simulated ROS clock standing at zero.
    """
    clock = ROSClock()
    time_source = TimeSource()
    time_source.ros_time_is_active = True
    time_source.attach_clock(clock)
    simulated = SimulatedClock(clock=clock, time_source=time_source)
    simulated.set_seconds(seconds=0.0)
    return simulated


# %% timing cycles


def test_the_computation_time_is_measured_on_the_ros_clock(
    simulated_clock: SimulatedClock,
):
    """
    A simulation running slower or faster than real time scales the computation time
    with it.
    """
    timing = CycleTiming(clock=simulated_clock.clock)
    simulated_clock.set_seconds(seconds=10.0)
    timing.start_cycle()
    simulated_clock.set_seconds(seconds=10.04)

    timing.finish_cycle()

    assert timing.computation_time == pytest.approx(0.04)


def test_the_publish_time_is_expected_one_computation_time_after_the_cycle_start(
    simulated_clock: SimulatedClock,
):
    """
    The running cycle is assumed to take as long as the previous one.
    """
    timing = CycleTiming(clock=simulated_clock.clock)
    simulated_clock.set_seconds(seconds=10.0)
    timing.start_cycle()
    simulated_clock.set_seconds(seconds=10.04)
    timing.finish_cycle()
    simulated_clock.set_seconds(seconds=10.1)

    timing.start_cycle()

    assert timing.expected_publish_time == pytest.approx(10.1 + 0.04)


def test_the_publish_time_is_expected_no_earlier_than_the_inputs_were_read(
    simulated_clock: SimulatedClock,
):
    """
    A cycle that takes longer to read its inputs than the previous cycle took to publish
    cannot publish before it has read them.
    """
    timing = CycleTiming(clock=simulated_clock.clock)
    simulated_clock.set_seconds(seconds=10.0)
    timing.start_cycle()
    simulated_clock.set_seconds(seconds=10.04)
    timing.finish_cycle()
    simulated_clock.set_seconds(seconds=10.1)
    timing.start_cycle()
    simulated_clock.set_seconds(seconds=10.2)

    timing.mark_inputs_read()

    assert timing.expected_publish_time == pytest.approx(10.2)


def test_finishing_a_cycle_returns_the_publish_time(simulated_clock: SimulatedClock):
    """
    The commands are stamped with the time read from the same clock the cycle start was
    read from.
    """
    timing = CycleTiming(clock=simulated_clock.clock)
    simulated_clock.set_seconds(seconds=10.0)
    timing.start_cycle()
    simulated_clock.set_seconds(seconds=10.04)

    assert timing.finish_cycle() == pytest.approx(10.04)


def test_a_clock_jumping_back_during_a_cycle_counts_as_no_computation_time(
    simulated_clock: SimulatedClock,
):
    """
    A restarted simulation's clock starts over; a negative computation time would expect
    the commands before the cycle even started.
    """
    timing = CycleTiming(clock=simulated_clock.clock)
    simulated_clock.set_seconds(seconds=1000.0)
    timing.start_cycle()
    simulated_clock.set_seconds(seconds=0.5)

    timing.finish_cycle()

    assert timing.computation_time == 0.0
