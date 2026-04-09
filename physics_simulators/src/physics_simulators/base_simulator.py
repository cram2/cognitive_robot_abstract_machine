#!/usr/bin/env python3

import atexit
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from threading import Thread
from typing import Optional, List, Any, Callable, Union, ClassVar


class SimulatorState(Enum):
    """Simulator State Enum"""

    STOPPED = 0
    PAUSED = 1
    RUNNING = 2


class SimulatorStopReason(Enum):
    """Simulator Stop Reason"""

    STOP = 0
    MAX_REAL_TIME = 1
    MAX_SIMULATION_TIME = 2
    MAX_NUMBER_OF_STEPS = 3
    VIEWER_IS_CLOSED = 4
    OTHER = 5


@dataclass
class SimulatorConstraints:
    max_real_time: float = None
    max_simulation_time: float = None
    max_number_of_steps: int = None


@dataclass
class SimulatorRenderer:
    """Base class for Renderer"""

    _is_running: bool = False

    def __init__(self):
        self._is_running = True
        atexit.register(self.close)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def is_running(self) -> bool:
        """Check if the renderer is running"""
        return self._is_running

    def sync(self):
        """Update the renderer"""
        pass

    def close(self):
        """Close the renderer"""
        self._is_running = False


@dataclass
class SimulatorCallbackResult:
    """
    Container for callback result, including its type, info, and result.
    """

    class OutputType(str, Enum):
        """
        Output type for SimulatorCallbackResult
        """

        MUJOCO = "mujoco"
        PYBULLET = "pybullet"
        ISAACSIM = "isaacsim"

    class ResultType(Enum):
        """
        Result type for SimulatorCallbackResult
        """

        SUCCESS_WITHOUT_EXECUTION = 0
        SUCCESS_AFTER_EXECUTION_ON_MODEL = 1
        SUCCESS_AFTER_EXECUTION_ON_DATA = 2
        FAILURE_WITHOUT_EXECUTION = 3
        FAILURE_BEFORE_EXECUTION_ON_MODEL = 4
        FAILURE_AFTER_EXECUTION_ON_MODEL = 5
        FAILURE_BEFORE_EXECUTION_ON_DATA = 6
        FAILURE_AFTER_EXECUTION_ON_DATA = 7

    type: ResultType
    """Result type"""
    info: str = None
    """Information about the result"""
    result: Any = None
    """Result of the callback"""

    def __call__(self):
        self.result = self.result()
        return self


@dataclass
class SimulatorCallback:
    """Base class for Simulator Callback"""

    def __init__(self, callback: Callable):
        """
        Initialize the function with the callback

        :param callback: callback function
        """
        self._call = callback
        self.__name__ = callback.__name__

    def __call__(self, *args, render: bool = True, **kwargs) -> SimulatorCallbackResult:
        result = self._call(*args, **kwargs)
        if not isinstance(result, SimulatorCallbackResult):
            raise TypeError("Callback function must return SimulatorCallbackResult")
        simulator = args[0]
        if not isinstance(simulator, BaseSimulator):
            raise TypeError("First argument must be of type BaseSimulator")
        if render:
            simulator.renderer.sync()
        return result

@dataclass
class BaseSimulator:
    """
    Base class for Base Simulator

    This class is intended as an abstract foundation for all specific simulators in your project.
    You do not use BaseSimulator directly to run a concrete physics engine; instead, you subclass it and implement engine-specific logic.
    """

    _headless: bool = field(repr=False)

    _step_size: float = field(default=1e-3, repr=False)

    _callbacks: List[SimulatorCallback] = field(default_factory=list, repr=False)

    config: dict = field(default_factory=dict)
    """Configuration for the simulator, it can be used to store any information that is needed for the simulator"""

    name: ClassVar[str] = "Base Simulation"
    """Name of the simulator"""

    ext: ClassVar[str] = ""
    """Extension of the simulator description file"""

    logger: ClassVar[logging.Logger] = logging.getLogger(__name__)
    """Logger for the simulator"""

    simulation_thread: Optional[Thread] = field(init=False, default=None)
    """Simulation thread, run step() method in this thread"""

    render_thread: Optional[Thread] = field(init=False, default=None)
    """Render thread, run render() method in this thread"""

    class_level_callbacks: ClassVar[List[SimulatorCallback]] = []
    """Class level callback functions"""

    instance_level_callbacks: List[SimulatorCallback] = field(init=False, default_factory=list)
    """Instance level callback functions"""

    _current_number_of_steps: int = field(init=False, default=0, repr=False)
    """Current number of steps in the simulation"""

    _start_real_time: float = field(init=False, repr=False)
    """Real time when the simulation starts, used for calculating the elapsed real time during the simulation"""

    _current_render_time: float = field(init=False, repr=False)
    """Real time when the renderer is last updated, used for calculating the elapsed real time during rendering"""

    _state: SimulatorState = field(init=False, repr=False)
    """Current state of the simulator"""

    _stop_reason: Optional[SimulatorStopReason] = field(init=False, default=None, repr=False)
    """Reason for stopping the simulator"""

    _renderer: SimulatorRenderer = field(init=False, repr=False)
    """Renderer for the simulator, it can be used to render the simulation in real time, and it can also be used to check if the renderer is still running or not."""

    def __post_init__(self):
        self._start_real_time = self.current_real_time
        self._state = SimulatorState.STOPPED
        self._stop_reason = None
        self._renderer = SimulatorRenderer()
        self._current_render_time = self.current_real_time
        self.instance_level_callbacks = []
        for func in self._callbacks:
            self.add_instance_callback(func)
        atexit.register(self.stop)

    @property
    def callbacks(self):
        return {
            callback.__name__: partial(callback, self)
            for callback in [
                *self.class_level_callbacks,
                *self.instance_level_callbacks,
            ]
        }

    def start(
        self,
        simulate_in_thread: bool = True,
        render_in_thread: bool = False,
        constraints: SimulatorConstraints = None,
        time_out_in_seconds: float = 10.0,
    ):
        """
        Start the simulator, if run_in_thread is True, run the simulator in a thread until the constraints are met

        :param constraints: constraints for stopping the simulator
        :param simulate_in_thread: True to simulate the simulator in a thread
        :param render_in_thread: True to render the simulator in a thread
        :param constraints: constraints for stopping the simulator
        :param time_out_in_seconds: timeout for starting the renderer
        """
        self.start_callback()
        self.reset()
        for i in range(int(10 * time_out_in_seconds)):
            if self.renderer.is_running():
                break
            time.sleep(0.1)
            if i % 10 == 0:
                self.log_info(f"Waiting for {self.renderer.__name__} to start")
        else:
            self.log_error(f"{self.renderer.__name__} is not running")
            return
        self._current_number_of_steps = 0
        self._start_real_time = self.current_real_time
        self._state = SimulatorState.RUNNING
        self._stop_reason = None
        if simulate_in_thread:
            self.simulation_thread = Thread(target=self.run, args=(constraints,))
            self.simulation_thread.start()
        if not self.headless and render_in_thread:
            def render():
                with self.renderer:
                    while self.renderer.is_running():
                        self.renderer.sync()
                        time.sleep(1.0 / 60.0)

            self.render_thread = Thread(target=render)
            self.render_thread.start()

    def run(self, constraints: SimulatorConstraints = None):
        """
        Run the simulator while the state is RUNNING or until the constraints are met.

        :param constraints: constraints for stopping the simulator
        """
        with self.renderer:
            while self.state != SimulatorState.STOPPED:
                self._stop_reason = self.should_stop(constraints)
                if self.stop_reason is not None:
                    self._state = SimulatorState.STOPPED
                    break
                match self.state:
                    case SimulatorState.RUNNING:
                        if self.current_simulation_time == 0.0:
                            self.reset()
                        self.step()

                    case SimulatorState.PAUSED:
                        self.pause_callback()
                        time.sleep(self.step_size)  # avoid busy waiting

                    case _:
                        time.sleep(self.step_size)
                if (
                    self.render_thread is None
                    and self.current_real_time - self._current_render_time > 1.0 / 60.0
                ):
                    self._current_render_time = self.current_real_time
                    self.render()
        self.stop_callback()

    def step(self):
        """
        Step the simulator. It reads the data from the viewer and writes the data to the simulator,
        then it reads the data from the simulator and writes the data to the viewer.
        It also increments the current simulation time and the current number of steps.
        """
        self.pre_step_callback()
        self.write_data_to_simulator()
        self.step_callback()
        self.read_data_from_simulator()

    def write_data_to_simulator(self):
        """
        Write data to the simulator.
        """
        pass

    def read_data_from_simulator(self):
        """
        Read data from the simulator.
        """
        pass

    def stop(self):
        """
        Stop the simulator, close the renderer and join the simulation thread if it exists and is alive.
        """
        if self.renderer.is_running():
            self.renderer.close()
        if self.render_thread is not None and self.render_thread.is_alive():
            self.render_thread.join()
        self._state = SimulatorState.STOPPED
        if self.simulation_thread is not None and self.simulation_thread.is_alive():
            self.simulation_thread.join()
        self._stop_reason = SimulatorStopReason.STOP

    def pause(self):
        """
        Pause the simulator. It doesn't pause the renderer.
        """
        if self.state != SimulatorState.RUNNING:
            self.log_warning("Cannot pause when the simulator is not running")
        else:
            self._state = SimulatorState.PAUSED

    def unpause(self):
        """
        Unpause the simulator and run the simulator.
        """
        if self.state == SimulatorState.PAUSED:
            self.unpause_callback()
            self._state = SimulatorState.RUNNING
        else:
            self.log_warning("Cannot unpause when the simulator is not paused")

    def reset(self):
        """
        Reset the simulator, set the start_real_time to current_real_time, current_simulate_time to 0.0,
        current_number_of_steps to 0, and run the simulator
        """
        self.reset_callback()
        self._current_number_of_steps = 0
        self._start_real_time = self.current_real_time

    def should_stop(
        self, constraints: SimulatorConstraints = None
    ) -> Optional[SimulatorStopReason]:
        """
        Check if the simulator should stop based on the constraints.

        :param constraints: constraints for stopping the simulator

        :return: bool, True if the simulator should stop, False otherwise
        """
        if constraints is None:
            return self.should_stop_callback()
        if (
            constraints.max_real_time is not None
            and self.current_real_time - self.start_real_time
            >= constraints.max_real_time
        ):
            self.log_info(
                f"Stopping simulation because max_real_time [{constraints.max_real_time}] reached"
            )
            return SimulatorStopReason.MAX_REAL_TIME
        if (
            constraints.max_simulation_time is not None
            and self.current_simulation_time >= constraints.max_simulation_time
        ):
            self.log_info(
                f"Stopping simulation because max_simulation_time [{constraints.max_simulation_time}] reached"
            )
            return SimulatorStopReason.MAX_SIMULATION_TIME
        if (
            constraints.max_number_of_steps is not None
            and self.current_number_of_steps >= constraints.max_number_of_steps
        ):
            self.log_info(
                f"Stopping simulation because max_number_of_steps [{constraints.max_number_of_steps}] reached"
            )
            return SimulatorStopReason.MAX_NUMBER_OF_STEPS
        return self.should_stop_callback()

    def start_callback(self):
        """
        This function is called when the simulator starts. It initializes the current simulation time and the renderer.
        """
        self._current_simulation_time = 0.0
        self._renderer = SimulatorRenderer()

    def render(self):
        self.renderer.sync()

    def pre_step_callback(self):
        pass

    def step_callback(self):
        """
        This function is called after the step function.
        It increments the current simulation time and the current number of steps.
        """
        if self.state == SimulatorState.RUNNING:
            self._current_simulation_time += self.step_size
            self._current_number_of_steps += 1

    def stop_callback(self):
        """
        This function is called when the simulator stops.
        It closes the renderer.
        """
        if self.renderer.is_running():
            self.renderer.close()

    def pause_callback(self):
        """
        This function is called when the simulator is paused.
        It updates the start_real_time to current_real_time - current_simulation_time.
        """
        self._start_real_time += (
            self.current_real_time - self.current_simulation_time - self.start_real_time
        )

    def unpause_callback(self):
        """
        This function is called when the simulator is unpaused.
        """
        pass

    def reset_callback(self):
        """
        This function is called when the simulator is reset.
        It sets the current simulation time to 0.0.
        """
        self._current_simulation_time = 0.0

    def should_stop_callback(self) -> Optional[SimulatorStopReason]:
        """
        This function is called when the simulator should stop.
        It returns None if the renderer is running, otherwise it returns SimulatorStopReason.VIEWER_IS_CLOSED.
        """
        return (
            None if self.renderer.is_running() else SimulatorStopReason.VIEWER_IS_CLOSED
        )

    @classmethod
    def log_info(cls, message: str):
        cls.logger.info(f"[{cls.name}] {message}")

    @classmethod
    def log_warning(cls, message: str):
        cls.logger.warning(f"[{cls.name}] {message}")

    @classmethod
    def log_error(cls, message: str):
        cls.logger.error(f"[{cls.name}] {message}")

    @property
    def headless(self) -> bool:
        return self._headless

    @property
    def step_size(self) -> float:
        return self._step_size

    @property
    def state(self) -> SimulatorState:
        return self._state

    @property
    def stop_reason(self) -> SimulatorStopReason:
        return self._stop_reason

    @property
    def start_real_time(self) -> float:
        return self._start_real_time

    @property
    def current_real_time(self) -> float:
        return time.time()

    @property
    def current_simulation_time(self) -> float:
        return self._current_simulation_time

    @property
    def current_number_of_steps(self) -> int:
        return self._current_number_of_steps

    @property
    def renderer(self) -> SimulatorRenderer:
        return self._renderer

    @classmethod
    def add_callback(
        cls,
        callback: Union[Callable, SimulatorCallback],
        callbacks: List[SimulatorCallback],
    ):
        if not isinstance(callback, SimulatorCallback):
            if isinstance(callback, Callable):
                callback = SimulatorCallback(callback=callback)
            else:
                raise TypeError(
                    f"Function {callback} must be an instance of SimulatorCallback or Callable, "
                    f"got {type(callback)}"
                )
        if callback.__name__ in [callback.__name__ for callback in callbacks]:
            raise AttributeError(f"Function {callback.__name__} is already defined")
        callbacks.append(callback)
        cls.log_info(f"Function {callback.__name__} is registered")

    def add_instance_callback(self, callback: Union[Callable, SimulatorCallback]):
        self.add_callback(callback, self.instance_level_callbacks)

    @classmethod
    def add_class_callback(cls, callback: Union[Callable, SimulatorCallback]):
        cls.add_callback(callback, cls.class_level_callbacks)

    @classmethod
    def simulator_callback(cls, callback):
        cls.add_class_callback(callback)
        return callback
