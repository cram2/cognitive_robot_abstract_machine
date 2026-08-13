import os
import time

import mujoco
import numpy
import pytest

from physics_simulators.mujoco_simulator import MujocoSimulator
from physics_simulators.base_simulator import (
    SimulatorConstraints,
    SimulatorState,
    SimulatorCallbackResult,
)

resources_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "semantic_digital_twin",
    "resources",
    "mjcf",
)
headless = os.environ.get("CI", "false").lower() == "true"
# headless = False


class TestMujocoSimulator:
    file_path = os.path.join(resources_path, "floor.xml")
    headless = headless
    step_size = 1e-3

    @pytest.fixture
    def simulator(self):
        sim = MujocoSimulator(
            _headless=self.headless,
            _step_size=self.step_size,
            file_path=os.path.join(resources_path, "mjx_single_cube_no_mesh.xml"),
        )
        yield sim
        try:
            sim.stop()
        except Exception:
            pass

    def test_functions(self, simulator):
        simulator.start(simulate_in_thread=False, render_in_thread=True)

        key_id = None
        save_file_path = None

        for step in range(4000):
            if step < 1000:
                result = simulator.callbacks["get_all_body_names"]()
                assert isinstance(result, SimulatorCallbackResult)
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert result.result == [
                    "world",
                    "link0",
                    "link1",
                    "link2",
                    "link3",
                    "link4",
                    "link5",
                    "link6",
                    "link7",
                    "hand",
                    "left_finger",
                    "right_finger",
                    "floor",
                    "box",
                ]

                result = simulator.callbacks["get_all_joint_names"]()
                assert isinstance(result, SimulatorCallbackResult)
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert result.result == [
                    "joint1",
                    "joint2",
                    "joint3",
                    "joint4",
                    "joint5",
                    "joint6",
                    "joint7",
                    "finger_joint1",
                    "finger_joint2",
                ]

            if step == 1000 or step == 3000:
                result = simulator.callbacks["attach"](body_1_name="abc")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.FAILURE_BEFORE_EXECUTION_ON_MODEL
                )
                assert result.info == "Body 1 abc not found"

                result = simulator.callbacks["attach"](body_1_name="world")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.FAILURE_BEFORE_EXECUTION_ON_MODEL
                )
                assert result.info == "Body 1 and body 2 are the same"

                result = simulator.callbacks["attach"](body_1_name="box")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert result.info == "Body 1 box is already attached to body 2 world"

                result = simulator.callbacks["attach"](
                    body_1_name="box", body_2_name="hand"
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL
                )
                assert "Attached body 1 box to body 2 hand" in result.info

                result = simulator.callbacks["enable_contact"](
                    body_1_name="box", body_2_name="left_finger"
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL
                )

                result = simulator.enable_contact(
                    body_1_name="box", body_2_name="right_finger"
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL
                )

                result = simulator.callbacks["attach"](
                    body_1_name="box", body_2_name="hand"
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert result.info == "Body 1 box is already attached to body 2 hand"

            if step == 1200:
                result = simulator.callbacks["get_joint_value"](joint_name="joint1")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, float)
                joint1_value = result.result

                result = simulator.callbacks["get_joints_values"](
                    joint_names=["joint1", "joint2"]
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, dict)
                assert len(result.result) == 2
                assert "joint1" in result.result
                assert isinstance(result.result["joint1"], float)
                assert "joint2" in result.result
                assert isinstance(result.result["joint2"], float)
                assert joint1_value == result.result["joint1"]

                result = simulator.callbacks["get_body_position"](body_name="box")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, numpy.ndarray)
                box_position = result.result

                result = simulator.callbacks["get_body_quaternion"](body_name="box")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, numpy.ndarray)
                box_quaternion = result.result

                result = simulator.callbacks["get_bodies_positions"](
                    body_names=["box", "link0"]
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, dict)
                assert len(result.result) == 2
                assert "box" in result.result
                assert isinstance(result.result["box"], numpy.ndarray)
                assert "link0" in result.result
                assert isinstance(result.result["link0"], numpy.ndarray)
                assert numpy.allclose(box_position, result.result["box"])

                result = simulator.callbacks["get_bodies_quaternions"](
                    body_names=["box", "link0"]
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, dict)
                assert len(result.result) == 2
                assert "box" in result.result
                assert isinstance(result.result["box"], numpy.ndarray)
                assert "link0" in result.result
                assert isinstance(result.result["link0"], numpy.ndarray)
                assert numpy.allclose(box_quaternion, result.result["box"])

            if step == 800:
                box_position = numpy.array([0.7, 0.0, 1.0])
                result = simulator.callbacks["set_body_position"](
                    body_name="box", position=box_position
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA
                )

                result = simulator.callbacks["get_body_position"](body_name="box")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, numpy.ndarray)
                assert numpy.allclose(box_position, result.result)

                box_position = numpy.array([0.7, 0.0, 2.0])
                result = simulator.callbacks["set_bodies_positions"](
                    bodies_positions={"box": box_position}
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA
                )

                result = simulator.callbacks["get_body_position"](body_name="box")
                assert numpy.allclose(box_position, result.result)

                box_quaternion = numpy.array([0.707, 0.707, 0.0, 0.0])
                box_quaternion /= numpy.linalg.norm(box_quaternion)
                result = simulator.callbacks["set_body_quaternion"](
                    body_name="box", quaternion=box_quaternion
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA
                )

                result = simulator.callbacks["get_body_quaternion"](body_name="box")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, numpy.ndarray)
                assert numpy.allclose(box_quaternion, result.result)

                box_quaternion = numpy.array([0.707, 0.0, 0.707, 0.0])
                box_quaternion /= numpy.linalg.norm(box_quaternion)
                result = simulator.callbacks["set_bodies_quaternions"](
                    bodies_quaternions={"box": box_quaternion}
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA
                )

                result = simulator.callbacks["get_body_quaternion"](body_name="box")
                assert numpy.allclose(box_quaternion, result.result)

                joint1_value = 0.3
                result = simulator.callbacks["set_joint_value"](
                    joint_name="joint1", value=joint1_value
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA
                )

                result = simulator.callbacks["get_joint_value"](joint_name="joint1")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, float)
                assert result.result == pytest.approx(joint1_value, abs=1e-3)

                joints_values = {"joint1": joint1_value, "joint2": 0.5}
                result = simulator.callbacks["set_joints_values"](
                    joints_values=joints_values
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA
                )

                result = simulator.callbacks["get_joints_values"](
                    joint_names=["joint1", "joint2"]
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, dict)
                assert len(result.result) == 2
                assert "joint1" in result.result
                assert isinstance(result.result["joint1"], float)
                assert "joint2" in result.result
                assert isinstance(result.result["joint2"], float)
                assert result.result["joint1"] == pytest.approx(joint1_value, abs=1e-3)

            if step == 1550:
                result = simulator.callbacks["save"](key_name="step_1550")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                key_id = result.result

                result = simulator.callbacks["load"](key_id=key_id)
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA
                )
                assert result.result == key_id

            if step == 1570:
                save_file_path = os.path.join(resources_path, "../output/step_1570.xml")
                os.makedirs(os.path.dirname(save_file_path), exist_ok=True)

                result = simulator.callbacks["save"](
                    file_path=save_file_path, key_name="step_1570"
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                key_id = result.result

                result = simulator.callbacks["load"](
                    file_path=save_file_path, key_id=key_id
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA
                )
                assert result.result == key_id

            if step == 2000 or step == 4000:
                result = simulator.callbacks["detach"](body_name="abc")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.FAILURE_BEFORE_EXECUTION_ON_MODEL
                )
                assert result.info == "Body abc not found"

                result = simulator.callbacks["detach"](body_name="world")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert result.info == "Body world is already detached"

                result = simulator.callbacks["detach"](body_name="box")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL
                )
                assert result.info == "Detached body box from body hand"

                result = simulator.callbacks["detach"](body_name="box")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert result.info == "Body box is already detached"

            if step == 8000:
                result = simulator.callbacks["get_contact_bodies"](body_name="abc")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION
                )
                assert result.info == "Body abc not found"

                result = simulator.callbacks["get_contact_bodies"](body_name="hand")
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, set)

            if step == 100:
                result = simulator.callbacks["get_contact_points"](body_names=["abc"])
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION
                )
                assert result.info == "Body abc not found"

                result = simulator.callbacks["get_contact_points"](
                    body_names=["box", "hand"]
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, list)
                assert len(result.result) == 0

                result = simulator.callbacks["get_contact_points"](body_names=["world"])
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert isinstance(result.result, list)
                assert len(result.result) == 4

            if step == 500 and mujoco.mj_version() < 3005000:
                result = simulator.callbacks["ray_test"](
                    ray_from_position=[0.7, 0.0, 1.0],
                    ray_to_position=[0.7, 0.0, 0.0],
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )

                result = simulator.callbacks["ray_test"](
                    ray_from_position=[0.7, 0.0, 0.2],
                    ray_to_position=[0.7, 0.0, 0.0],
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )

                result = simulator.callbacks["ray_test_batch"](
                    ray_from_position=[0.7, 0.0, 0.2],
                    ray_to_positions=[[0.7, 0.0, 1.0], [0.7, 0.0, 0.0]],
                )
                assert (
                    result.type
                    is SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                )
                assert result.result[1]["hit_position"][2] == pytest.approx(
                    0.0599, abs=1e-3
                )

            simulator.step()
            time.sleep(0.001)

        simulator.stop()


class TestMujocoSimulatorComplex:
    file_path = os.path.join(resources_path, "mjx_single_cube_no_mesh.xml")
    Simulator = MujocoSimulator
    headless = headless
    step_size = 5e-4

    def make_simulator(self, headless=None):
        if headless is None:
            headless = self.headless
        return MujocoSimulator(
            _headless=headless,
            _step_size=self.step_size,
            file_path=self.file_path,
        )

    def test_running_in_10s_in_1(self):
        simulator = self.make_simulator()
        try:
            constraints = SimulatorConstraints(max_real_time=10.0)
            simulator.start(
                constraints=constraints,
                simulate_in_thread=True,
                render_in_thread=True,
            )
            while simulator.state != SimulatorState.STOPPED:
                time.sleep(1)
            assert simulator.state is SimulatorState.STOPPED
        finally:
            try:
                simulator.stop()
            except Exception:
                pass

    def test_running_in_10s_2(self):
        simulator = self.make_simulator()
        try:
            constraints = SimulatorConstraints(max_real_time=10.0)
            simulator.start(
                constraints=constraints,
                simulate_in_thread=True,
                render_in_thread=False,
            )
            while simulator.state != SimulatorState.STOPPED:
                time.sleep(1)
            assert simulator.state is SimulatorState.STOPPED
        finally:
            try:
                simulator.stop()
            except Exception:
                pass

    def test_running_in_10s_3(self):
        simulator = self.make_simulator()
        try:
            constraints = SimulatorConstraints(max_real_time=10.0)
            simulator.start(
                constraints=constraints,
                simulate_in_thread=False,
                render_in_thread=True,
            )
            while simulator.state != SimulatorState.STOPPED:
                simulator.step()
                time.sleep(0.001)
                if simulator.current_number_of_steps == 10000:
                    simulator.stop()
            assert simulator.state is SimulatorState.STOPPED
        finally:
            try:
                simulator.stop()
            except Exception:
                pass

    def test_running_in_10s_4(self):
        simulator = self.make_simulator()
        try:
            constraints = SimulatorConstraints(max_real_time=10.0)
            simulator.start(
                constraints=constraints,
                simulate_in_thread=False,
                render_in_thread=False,
            )
            while simulator.state != SimulatorState.STOPPED:
                simulator.step()
                simulator.render()
                time.sleep(0.001)
                if simulator.current_number_of_steps == 10000:
                    simulator.stop()
            assert simulator.state is SimulatorState.STOPPED
        finally:
            try:
                simulator.stop()
            except Exception:
                pass

    def test_running_2_simulators(self):
        simulator1 = self.make_simulator(headless=self.headless)
        simulator2 = self.make_simulator(headless=True)
        simulator3 = self.make_simulator(headless=True)

        try:
            simulator1.start(simulate_in_thread=False, render_in_thread=True)
            simulator2.start(simulate_in_thread=False)
            simulator3.start(simulate_in_thread=False)

            for _ in range(10000):
                simulator1.step()
                simulator2.step()
                simulator3.step()

            simulator1.stop()
            simulator2.stop()
            simulator3.stop()

            assert simulator1.state is SimulatorState.STOPPED
            assert simulator2.state is SimulatorState.STOPPED
            assert simulator3.state is SimulatorState.STOPPED
        finally:
            for sim in (simulator1, simulator2, simulator3):
                try:
                    sim.stop()
                except Exception:
                    pass