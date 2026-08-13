#!/usr/bin/env python3

import time
from typing import List, Optional

import pytest

from physics_simulators.base_simulator import (
    BaseSimulator,
    SimulatorState,
    SimulatorConstraints,
    SimulatorStopReason,
    SimulatorCallback,
    SimulatorCallbackResult,
)


class TestBaseSimulator:
    file_path: str = ""
    world_path: str = ""
    robots_path: Optional[str] = None
    headless: bool = False
    step_size: float = 1e-3
    Simulator = BaseSimulator
    number_of_envs: int = 1

    @pytest.fixture
    def simulator_factory(self):
        def _make_simulator(
            callbacks: Optional[List[SimulatorCallback]] = None,
            step_size: Optional[float] = None,
        ) -> BaseSimulator:
            callbacks = callbacks or []
            simulator = self.Simulator(
                self.headless,
                self.step_size if step_size is None else step_size,
                callbacks,
            )
            return simulator

        return _make_simulator

    def _assert_initialized(self, simulator: BaseSimulator) -> None:
        assert simulator.state is SimulatorState.STOPPED
        assert simulator.headless is self.headless
        assert simulator.stop_reason is None
        assert simulator.simulation_thread is None

    def _start_and_stop_simulator(self, simulator: BaseSimulator) -> BaseSimulator:
        simulator.start()
        assert simulator.state is SimulatorState.RUNNING

        simulator.stop()
        assert simulator.state is SimulatorState.STOPPED
        assert simulator.stop_reason is SimulatorStopReason.STOP
        assert not simulator.renderer.is_running()
        assert not simulator.simulation_thread.is_alive()
        return simulator

    def _pause_and_unpause_simulator(self, simulator: BaseSimulator) -> BaseSimulator:
        simulator.start()
        assert simulator.state is SimulatorState.RUNNING

        for _ in range(10):
            simulator.pause()
            assert simulator.state is SimulatorState.PAUSED

            simulator.unpause()
            assert simulator.state is SimulatorState.RUNNING

        simulator.unpause()
        assert simulator.state is SimulatorState.RUNNING

        simulator.stop()
        return simulator

    def _run_with_constraints(
        self,
        simulator: BaseSimulator,
        constraints: Optional[SimulatorConstraints] = None,
    ) -> BaseSimulator:
        simulator.start(constraints=constraints)

        while simulator.state == SimulatorState.RUNNING:
            if constraints is None:
                simulator.renderer.close()
            else:
                if (
                    constraints.max_number_of_steps is not None
                    and simulator.current_number_of_steps
                    > constraints.max_number_of_steps + 10
                ):
                    raise Exception("Constraints max_number_of_steps are not working")

                if (
                    constraints.max_simulation_time is not None
                    and simulator.current_simulation_time
                    > constraints.max_simulation_time + 10 * simulator.step_size
                ):
                    raise Exception("Constraints max_simulation_time are not working")

                if (
                    constraints.max_real_time is not None
                    and simulator.current_real_time - simulator.start_real_time
                    > constraints.max_real_time + 1.0
                ):
                    raise Exception("Constraints max_real_time are not working")

        if constraints is None:
            assert simulator.stop_reason is SimulatorStopReason.VIEWER_IS_CLOSED
        else:
            if constraints.max_number_of_steps is not None:
                assert (
                    simulator.current_number_of_steps
                    <= constraints.max_number_of_steps
                )

            if constraints.max_simulation_time is not None:
                assert (
                    simulator.current_simulation_time
                    <= constraints.max_simulation_time + simulator.step_size
                )

            if constraints.max_real_time is not None:
                assert (
                    simulator.current_real_time - simulator.start_real_time
                    <= constraints.max_real_time + 1.0
                )

            assert simulator.stop_reason is not None

        return simulator

    def test_initialize_simulator(self, simulator_factory):
        simulator = simulator_factory()
        self._assert_initialized(simulator)

    def test_start_and_stop_simulator(self, simulator_factory):
        simulator = simulator_factory()
        self._assert_initialized(simulator)
        self._start_and_stop_simulator(simulator)

    def test_pause_and_unpause_simulator(self, simulator_factory):
        simulator = simulator_factory()
        self._assert_initialized(simulator)
        self._pause_and_unpause_simulator(simulator)

    def test_step_simulator(self, simulator_factory):
        simulator = simulator_factory()
        self._assert_initialized(simulator)

        simulator.start(simulate_in_thread=False)
        assert simulator.stop_reason is None

        for i in range(10):
            assert simulator.current_number_of_steps == i

            if (simulator.current_simulation_time - i * simulator.step_size) > 1e-6:
                print(simulator.current_simulation_time, i * simulator.step_size)

            assert simulator.current_simulation_time == pytest.approx(
                i * simulator.step_size
            )

            simulator.step()
            assert simulator.stop_reason is None

        simulator.stop()
        assert simulator.stop_reason is SimulatorStopReason.STOP

    def test_reset_simulator(self, simulator_factory):
        simulator = simulator_factory()
        self._assert_initialized(simulator)

        simulator.start(simulate_in_thread=False)
        simulator.reset()

        assert simulator.current_number_of_steps == 0
        assert simulator.current_simulation_time == 0.0

        simulator.stop()
        assert simulator.stop_reason is SimulatorStopReason.STOP

    def test_run_with_constraints_simulator(self, simulator_factory):
        simulator = simulator_factory()
        self._assert_initialized(simulator)

        constraints = SimulatorConstraints(max_number_of_steps=10)
        self._run_with_constraints(simulator, constraints=constraints)

    def test_run_with_multiple_constraints_simulator(self, simulator_factory):
        max_number_of_steps = 10
        max_simulation_time = 0.01
        max_real_time = 0.1

        simulator = simulator_factory()
        self._assert_initialized(simulator)
        constraints = SimulatorConstraints(max_number_of_steps=max_number_of_steps)
        self._run_with_constraints(simulator, constraints=constraints)
        assert simulator.current_number_of_steps == max_number_of_steps
        assert simulator.stop_reason is SimulatorStopReason.MAX_NUMBER_OF_STEPS

        simulator = simulator_factory()
        self._assert_initialized(simulator)
        constraints = SimulatorConstraints(max_simulation_time=max_simulation_time)
        self._run_with_constraints(simulator, constraints=constraints)
        assert simulator.current_simulation_time == pytest.approx(max_simulation_time)

        simulator = simulator_factory()
        self._assert_initialized(simulator)
        constraints = SimulatorConstraints(max_real_time=max_real_time)
        self._run_with_constraints(simulator, constraints=constraints)
        assert (
            simulator.current_real_time - simulator.start_real_time
            <= max_real_time + 1.0
        )

        simulator = simulator_factory()
        self._assert_initialized(simulator)
        constraints = SimulatorConstraints(
            max_number_of_steps=max_number_of_steps,
            max_simulation_time=max_simulation_time,
            max_real_time=max_real_time,
        )
        self._run_with_constraints(simulator, constraints=constraints)
        assert simulator.stop_reason is not None

    def test_real_time(self, simulator_factory):
        simulator = simulator_factory(step_size=1e-4)
        self._assert_initialized(simulator)

        constraints = SimulatorConstraints(max_real_time=1.0)
        simulator.start(constraints=constraints)

        while simulator.state == SimulatorState.RUNNING:
            time.sleep(1)

        assert simulator.state is SimulatorState.STOPPED

    def test_making_functions(self, simulator_factory):
        result_1 = SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info="Test function 1",
            result="Hello, World!",
        )

        def function_1(
            physics_simulator: BaseSimulator,
        ) -> SimulatorCallbackResult:
            return result_1

        function_1 = SimulatorCallback(function_1)

        result_2 = SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.FAILURE_AFTER_EXECUTION_ON_DATA,
            info="Test function 2",
            result="Hello, World!",
        )

        def function_2(
            physics_simulator: BaseSimulator,
        ) -> SimulatorCallbackResult:
            return result_2

        function_2 = SimulatorCallback(function_2)

        simulator = simulator_factory(callbacks=[function_1, function_2])
        self._assert_initialized(simulator)

        assert simulator.callbacks["function_1"]() == result_1
        assert simulator.callbacks["function_2"]() == result_2

        with pytest.raises(Exception) as exc_info:
            simulator_factory(callbacks=[function_1, function_2, function_2])

        assert f"Function {function_2.__name__} is already defined" in str(
            exc_info.value
        )