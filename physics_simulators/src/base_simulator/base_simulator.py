#!/usr/bin/env python3

import atexit
import logging
import time
from dataclasses import dataclass
from enum import Enum
from functools import partial
from threading import Thread
from typing import Optional, Dict, List, Tuple, Any, Callable, Union

import numpy


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
class SimulatorAttribute:
    """Base class for Simulator Attribute"""

    default_value: numpy.ndarray
    """Default value of the attribute"""
    _values: numpy.ndarray = None
    """Values of the attribute"""

    def __init__(self, default_value: numpy.ndarray):
        self.default_value = default_value

    def initialize_data(self, number_of_envs: int):
        """
        Initialize the data for the attribute

        :param number_of_envs: int, number of environments
        """
        self._values = numpy.array([self.default_value for _ in range(number_of_envs)])

    @property
    def values(self):
        if self._values is None:
            raise ValueError("Values are not set, call initialize_data() first.")
        return self._values


class SimulatorViewer:
    """Base class for Simulator Viewer"""

    def __init__(
        self,
        write_objects: Optional[
            Dict[str, Dict[str, Union[numpy.ndarray, List[float]]]]
        ] = None,
        read_objects: Optional[
            Dict[str, Dict[str, Union[numpy.ndarray, List[float]]]]
        ] = None,
    ):
        self._write_objects = (
            self.from_array(write_objects) if write_objects is not None else {}
        )
        self._write_data = numpy.array([])

        self._read_objects = (
            self.from_array(read_objects) if read_objects is not None else {}
        )
        self._read_data = numpy.array([])

    @staticmethod
    def from_array(
        data: Dict[str, Dict[str, Union[numpy.ndarray, List[float]]]],
    ) -> Dict[str, Dict[str, SimulatorAttribute]]:
        """
        Convert the data array to SimulatorAttribute objects

        :param data: Dict[str, Dict[str, Union[numpy.ndarray, List[float]]]], data array
        :return: Dict[str, Dict[str, SimulatorAttribute]], SimulatorAttribute objects
        """
        return {
            key: {
                key2: SimulatorAttribute(default_value=value)
                for key2, value in value.items()
            }
            for key, value in data.items()
        }

    def initialize_data(self, number_of_envs: int) -> "SimulatorViewer":
        """
        Initialize the data for the viewer

        :param number_of_envs: int, number of environments
        """
        self._write_data = numpy.array(
            [self._initialize_data(self._write_objects) for _ in range(number_of_envs)]
        )
        self._read_data = numpy.array(
            [self._initialize_data(self._read_objects) for _ in range(number_of_envs)]
        )
        for objects in [self._write_objects, self._read_objects]:
            for attrs in objects.values():
                for attr in attrs.values():
                    attr.initialize_data(number_of_envs)
        return self

    @staticmethod
    def _initialize_data(
        objects: Dict[str, Dict[str, SimulatorAttribute]],
    ) -> numpy.ndarray:
        """
        Flatten attribute values into a NumPy array.

        :param objects: Dict[str, Dict[str, SimulatorAttribute]], objects with attributes
        :return: numpy.ndarray, flattened attribute values
        """
        return numpy.array(
            [
                i
                for attrs in objects.values()
                for attr in attrs.values()
                for i in attr.default_value
            ]
        )

    @property
    def write_objects(self) -> Dict[str, Dict[str, SimulatorAttribute]]:
        self._update_objects_from_data(self._write_objects, self.write_data)
        return self._write_objects

    @write_objects.setter
    def write_objects(
        self,
        send_objects: Dict[
            str, Dict[str, Union[numpy.ndarray, List[float], SimulatorAttribute]]
        ],
    ):
        number_of_envs = self.write_data.shape[0]
        self._write_objects, self._write_data = (
            self._get_objects_and_data_from_target_objects(send_objects, number_of_envs)
        )
        assert self.write_data.shape[0] == self.read_data.shape[0]

    @property
    def read_objects(self) -> Dict[str, Dict[str, SimulatorAttribute]]:
        self._update_objects_from_data(self._read_objects, self.read_data)
        return self._read_objects

    @read_objects.setter
    def read_objects(
        self,
        objects: Dict[
            str, Dict[str, Union[numpy.ndarray, List[float], SimulatorAttribute]]
        ],
    ):
        number_of_envs = self.read_data.shape[0]
        self._read_objects, self._read_data = (
            self._get_objects_and_data_from_target_objects(objects, number_of_envs)
        )
        assert self.read_data.shape[0] == self.write_data.shape[0]

    @staticmethod
    def _get_objects_and_data_from_target_objects(
        target_objects: Dict[
            str, Dict[str, Union[numpy.ndarray, List[float], SimulatorAttribute]]
        ],
        number_of_envs: int,
    ) -> Tuple[Dict[str, Dict[str, SimulatorAttribute]], numpy.ndarray]:
        """
        Update object attribute values from the target objects.

        :param target_objects: Dict[str, Dict[str, Union[numpy.ndarray, List[float], SimulatorAttribute]]], target objects
        :param number_of_envs: int, number of environments
        """
        if any(
            isinstance(value, (numpy.ndarray, list))
            for values in target_objects.values()
            for value in values.values()
        ):
            objects = (
                SimulatorViewer.from_array(target_objects)
                if target_objects is not None
                else {}
            )
        else:
            objects = target_objects
        for attrs in objects.values():
            for attr in attrs.values():
                if attr._values is None:
                    attr.initialize_data(number_of_envs)
        data = numpy.array(
            [
                [
                    value
                    for attrs in objects.values()
                    for attr in attrs.values()
                    for value in attr.values[env_id]
                ]
                for env_id in range(number_of_envs)
            ]
        )
        return objects, data

    @staticmethod
    def _update_objects_from_data(
        objects: Dict[str, Dict[str, SimulatorAttribute]], data: numpy.ndarray
    ):
        """
        Update object attribute values from the data array.

        :param objects: Dict[str, List[SimulatorAttribute]], objects with attributes
        """
        for i, values in enumerate(data):
            j = 0
            for obj_name, attrs in objects.items():
                for name, attr in attrs.items():
                    attr.values[i] = values[j : j + len(attr.default_value)]
                    j += len(attr.default_value)

    @property
    def write_data(self) -> numpy.ndarray:
        return self._write_data

    @write_data.setter
    def write_data(self, data: numpy.ndarray):
        if data.shape != self._write_data.shape:
            raise ValueError(
                f"Data length mismatch with write_objects, expected {self._write_data.shape}, got {data.shape}"
            )
        self._write_data[:] = data

    @property
    def read_data(self) -> numpy.ndarray:
        return self._read_data

    @read_data.setter
    def read_data(self, data: numpy.ndarray):
        if data.shape != self._read_data.shape:
            raise ValueError(
                f"Data length mismatch with read_objects, expected {self._read_data.shape}, got {data.shape}"
            )
        self._read_data[:] = data


@dataclass
class SimulatorCallbackResult:
    class OutType(str, Enum):
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


class SimulatorCallback:
    """Base class for Simulator Callback"""

    def __init__(self, callback: Callable):
        """
        Initialize the function with the callback

        :param callback: Callable, callback function, must return SimulatorCallbackResult
        """
        self._call = callback
        self.__name__ = callback.__name__

    def __call__(self, *args, render: bool = True, **kwargs):
        result = self._call(*args, **kwargs)
        if not isinstance(result, SimulatorCallbackResult):
            raise TypeError("Callback function must return SimulatorCallbackResult")
        simulator = args[0]
        if not isinstance(simulator, BaseSimulator):
            raise TypeError("First argument must be of type BaseSimulator")
        if render:
            simulator.renderer.sync()
        return result


class BaseSimulator:
    """Base class for Base Simulator"""

    name: str = "Base Simulation"
    """Name of the simulator"""

    ext: str = ""
    """Extension of the simulator description file"""

    simulation_thread: Thread = None
    """Simulation thread, run step() method in this thread"""

    render_thread: Thread = None
    """Render thread, run render() method in this thread"""

    logger: logging.Logger = logging.getLogger(__name__)
    """Logger for the simulator"""

    class_level_callbacks: List[SimulatorCallback] = []
    """Class level callback functions"""

    instance_level_callbacks: List[SimulatorCallback] = None
    """Instance level callback functions"""

    def __init__(
        self,
        viewer: Optional[SimulatorViewer] = None,
        number_of_envs: int = 1,
        headless: bool = False,
        real_time_factor: float = 1.0,
        step_size: float = 1e-3,
        callbacks: List[SimulatorCallback] = None,
        **kwargs,
    ):
        """
        Initialize the simulator with the viewer and the following keyword arguments:

        :param viewer: SimulatorViewer, viewer for the simulator
        :param number_of_envs: int, number of environments
        :param headless: bool, True to run the simulator in headless mode
        :param real_time_factor: float, real time factor
        :param step_size: float, step size
        :param callbacks: List[SimulatorCallback], list of callback functions
        """
        self._headless = headless
        self._real_time_factor = real_time_factor
        self._step_size = step_size
        self._current_number_of_steps = 0
        self._start_real_time = self.current_real_time
        self._state = SimulatorState.STOPPED
        self._stop_reason = None
        self._viewer = (
            viewer.initialize_data(number_of_envs) if viewer is not None else None
        )
        self._renderer = SimulatorRenderer()
        self._current_render_time = self.current_real_time
        self.instance_level_callbacks = []
        if callbacks is not None:
            for func in callbacks:
                self.add_instance_callback(func)
        self._write_objects = {}
        self._read_objects = {}
        self._write_ids = {}
        self._read_ids = {}
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

        :param constraints: SimulatorConstraints, constraints for stopping the simulator
        :param simulate_in_thread: bool, True to simulate the simulator in a thread
        :param render_in_thread: bool, True to render the simulator in a thread
        :param constraints: SimulatorConstraints, constraints for stopping the simulator
        :param time_out_in_seconds: float, timeout for starting the renderer
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

        :param constraints: SimulatorConstraints, constraints for stopping the simulator
        """
        with self.renderer:
            while self.state != SimulatorState.STOPPED:
                self._stop_reason = self.should_stop(constraints)
                if self.stop_reason is not None:
                    self._state = SimulatorState.STOPPED
                    break
                if self.state == SimulatorState.RUNNING:
                    if self.current_simulation_time == 0.0:
                        self.reset()
                    self.step()
                elif self.state == SimulatorState.PAUSED:
                    self.pause_callback()
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
        If the current simulation time is not consistent with the current number of steps and step size, it resets the simulator.
        """
        self.pre_step_callback()
        last_simulation_time = (
            self.current_simulation_time
            if self.state == SimulatorState.RUNNING
            else None
        )
        if self._viewer is not None:
            self.write_data_to_simulator(write_data=self._viewer.write_data)
            self.step_callback()
            self.read_data_from_simulator(read_data=self._viewer.read_data)
        else:
            self.step_callback()
        if self.state == SimulatorState.RUNNING and not numpy.isclose(
            self.current_simulation_time - last_simulation_time, self.step_size
        ):
            self.log_warning(
                f"Simulation time {self.current_simulation_time:.4f} is inconsistent with "
                f"number of steps {self.current_number_of_steps} and step size {self.step_size}, resetting the simulator"
            )
            self.reset()
        if not numpy.isclose(
            self.current_number_of_steps * self.step_size, self.current_simulation_time
        ):
            if numpy.isclose(
                self.current_simulation_time, self.step_size, self.step_size
            ):
                self._current_number_of_steps = 1
            else:
                self.log_error(
                    f"Simulation time {self.current_simulation_time:.4f} is inconsistent with "
                    f"number of steps {self.current_number_of_steps} and step size {self.step_size}"
                )

    def write_data_to_simulator(self, write_data: numpy.ndarray):
        """
        Write data to the simulator.

        :param write_data: numpy.ndarray, data to write
        """
        raise NotImplementedError("write_data method is not implemented")

    def read_data_from_simulator(self, read_data: numpy.ndarray):
        """
        Read data from the simulator.

        :param read_data: numpy.ndarray, data to read
        """
        raise NotImplementedError("read_data method is not implemented")

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

        :param constraints: SimulatorConstraints, constraints for stopping the simulator

        :return: bool, True if the simulator should stop, False otherwise
        """
        if constraints is not None:
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

    def _process_objects(self, objects, ids_dict):
        """
        Process objects for updating `read_ids` or `write_ids`.

        :param objects: Dictionary of objects and attributes.
        :param ids_dict: Dictionary to store processed IDs.
        """
        pass

    def pre_step_callback(self):
        """
        Update `write_ids` and `read_ids` based on the viewer's `write_objects` and `read_objects`.
        """
        if self._viewer is not None:
            if self.__should_process_objects(
                self._viewer.write_objects, self._write_objects
            ):
                self._process_objects(self._viewer.write_objects, self._write_ids)
                self._write_objects = self._viewer.write_objects
            if self.__should_process_objects(
                self._viewer.read_objects, self._read_objects
            ):
                self._process_objects(self._viewer.read_objects, self._read_ids)
                self._read_objects = self._viewer.read_objects

    @staticmethod
    def __should_process_objects(viewer_objects, cache_objects):
        """
        Check if the objects in the viewer are different from the objects in the cache.
        """
        for name, attrs in cache_objects.items():
            if name not in viewer_objects or any(
                attr_name not in viewer_objects[name] for attr_name in attrs
            ):
                return True
        for name, attrs in viewer_objects.items():
            if name not in cache_objects or any(
                attr_name not in cache_objects[name] for attr_name in attrs
            ):
                return True
        return False

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
