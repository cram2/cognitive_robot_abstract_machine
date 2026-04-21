#!/usr/bin/env python3

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, InitVar
from typing import Optional, List, Set, Dict, Union, Any

import mujoco
import mujoco.viewer
import numpy

from physics_simulators.base_simulator import (
    BaseSimulator,
    SimulatorRenderer,
    SimulatorCallbackResult,
    SimulatorState,
)


@dataclass
class MujocoRenderer(SimulatorRenderer):
    """MuJoCo Renderer class"""

    def __init__(self, mj_viewer: mujoco.viewer):
        self._mj_viewer = mj_viewer
        super().__init__()

    def is_running(self) -> bool:
        return self.mj_viewer.is_running()

    def close(self):
        self.mj_viewer.close()

    @property
    def mj_viewer(self) -> mujoco.viewer:
        return self._mj_viewer


@dataclass(unsafe_hash=True)
class MujocoSimulator(BaseSimulator):
    """Mujoco Simulator class"""

    _name: str = field(init=False, repr=False)
    """
    Name of the scene
    """

    _file_path: str = field(init=False, repr=False)
    """
    Path to the XML file of the scene
    """

    file_path: InitVar[str] = ""
    """
    Path to the XML file of the scene (for initialization)
    """


    def __post_init__(self, file_path: str = ""):
        super().__post_init__()
        self._file_path = file_path
        root = ET.parse(file_path).getroot()
        self._name = root.attrib.get("model", self.name)
        self._mj_spec = mujoco.MjSpec.from_file(filename=self._file_path)
        self._mj_spec.option.integrator = getattr(
            mujoco.mjtIntegrator, f"mjINT_{self.config.get('integrator', 'RK4')}"
        )
        self._mj_spec.option.noslip_iterations = int(self.config.get("noslip_iterations", 0))
        self._mj_spec.option.noslip_tolerance = float(
            self.config.get("noslip_tolerance", 1e-6)
        )
        self._mj_spec.option.cone = getattr(
            mujoco.mjtCone, f"mjCONE_{self.config.get('cone', 'PYRAMIDAL')}"
        )
        self._mj_spec.option.impratio = float(self.config.get("impratio", 1))
        self._mj_spec.option.timestep = self.step_size
        if self.config.get("multiccd", False):
            self._mj_spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD
        if self.config.get("energy", True):
            self._mj_spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
        if not self.config.get("contact", True):
            self._mj_spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        if not self.config.get("gravity", True):
            self._mj_spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_GRAVITY
        if mujoco.mj_version() >= 330:
            if not self.config.get("nativeccd", True):
                self._mj_spec.option.disableflags |= (
                    mujoco.mjtDisableBit.mjDSBL_NATIVECCD
                )
        else:
            if self.config.get("nativeccd", False):
                self._mj_spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_NATIVECCD
        self._mj_model = self._mj_spec.compile()
        assert self._mj_model is not None
        self._mj_data = mujoco.MjData(self._mj_model)

        mujoco.mj_resetDataKeyframe(self._mj_model, self._mj_data, 0)

    def start_callback(self):
        if not self.headless:
            self._renderer = mujoco.viewer.launch_passive(self._mj_model, self._mj_data)
        else:
            self._renderer = SimulatorRenderer()

    def step_callback(self):
        def _do_step():
            if self.state == SimulatorState.RUNNING:
                self._current_number_of_steps += 1
                mujoco.mj_step(self._mj_model, self._mj_data)
            elif self.state == SimulatorState.PAUSED:
                mujoco.mj_kinematics(self._mj_model, self._mj_data)

        if self.render_thread is not None:
            with self.renderer.lock():
                _do_step()
        else:
            _do_step()

    def reset_callback(self):
        mujoco.mj_resetDataKeyframe(self._mj_model, self._mj_data, 0)

    def _fix_prefix_and_recompile(
        self, body_spec: mujoco.MjsBody, dummy_prefix: str, body_name: str
    ):
        """
        This is a workaround to a bug that happens in MuJoCo older versions when detaching a body that has children:
        the children bodies are not properly detached and keep the same name,
        which causes a conflict when attaching them to another body with the same name.
        The workaround is to add a dummy prefix to the body and its children before recompiling,
        then remove the dummy prefix after recompiling.
        """
        body_spec.name = body_name
        try:
            for body_child in (
                body_spec.bodies + body_spec.joints + body_spec.geoms + body_spec.sites
            ):
                body_child.name = body_child.name.replace(dummy_prefix, "")
        except ValueError:
            self.log_warning(
                f"Failed to resolve body_spec for {body_name}, this is a bug from MuJoCo"
            )
            self._mj_model, self._mj_data = self._mj_spec.recompile(
                self._mj_model, self._mj_data
            )
            for body_child in (
                body_spec.bodies + body_spec.joints + body_spec.geoms + body_spec.sites
            ):
                body_child.name = body_child.name.replace(dummy_prefix, "")
        self._mj_model, self._mj_data = self._mj_spec.recompile(
            self._mj_model, self._mj_data
        )
        for key in self._mj_spec.keys:
            if key.name != "home":
                if mujoco.mj_version() < 335:
                    key.delete()
                else:
                    self._mj_spec.delete(key)
        self._mj_model, self._mj_data = self._mj_spec.recompile(
            self._mj_model, self._mj_data
        )
        if not self.headless:
            self._renderer._sim().load(self._mj_model, self._mj_data, "")
            if self.simulation_thread is None:
                mujoco.mj_step1(self._mj_model, self._mj_data)

    @property
    def file_path(self) -> str:
        return self._file_path

    @property
    def current_simulation_time(self) -> float:
        return self._mj_data.time

    @property
    def renderer(self):
        return self._renderer

    @BaseSimulator.simulator_callback
    def get_all_body_names(self) -> SimulatorCallbackResult:
        """
        Get all body names

        :return: A SimulatorCallbackResult with a list of all body names as the result
        """
        result = [
            self._mj_model.body(body_id).name for body_id in range(self._mj_model.nbody)
        ]
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info="Getting all body names",
            result=result,
        )

    @BaseSimulator.simulator_callback
    def get_body(self, body_name: str) -> SimulatorCallbackResult:
        """
        Get a body by its name

        :param body_name: The name of the body
        :return: A SimulatorCallbackResult with a mujoco.MjsBody as the result
        """
        body_id = mujoco.mj_name2id(
            m=self._mj_model, type=mujoco.mjtObj.mjOBJ_BODY, name=body_name
        )
        if body_id == -1:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                info=f"Body {body_name} not found",
            )
        body = self._mj_data.body(body_id)
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info=f"Getting body {body_name}",
            result=body,
        )

    @BaseSimulator.simulator_callback
    def get_body_position(self, body_name: str) -> SimulatorCallbackResult:
        """
        Get a body position by its name

        :param body_name: The name of the body
        :return: A SimulatorCallbackResult with a numpy array of shape (3,) representing the position as the result
        """
        get_body = self.get_body(body_name)
        if (
            get_body.type
            != SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
        ):
            return get_body
        body = get_body.result
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info=f"Getting body position of {body_name}",
            result=body.xpos,
        )

    @BaseSimulator.simulator_callback
    def get_body_quaternion(self, body_name: str) -> SimulatorCallbackResult:
        """
        Get a body quaternion by its name

        :param body_name: The name of the body
        :return: A SimulatorCallbackResult with a numpy array of shape (4,) representing the quaternion in WXYZ format as the result
        """
        get_body = self.get_body(body_name)
        if (
            get_body.type
            != SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
        ):
            return get_body
        body = get_body.result
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info=f"Getting body quaternion (WXYZ) of {body_name}",
            result=body.xquat,
        )

    @BaseSimulator.simulator_callback
    def get_bodies_positions(self, body_names: List[str]) -> SimulatorCallbackResult:
        """
        Get bodies positions by body names

        :param body_names: The names of the bodies
        :return: A SimulatorCallbackResult with a list of all bodies positions as the result
        """
        result = {}
        for body_name in body_names:
            get_body_position = self.get_body_position(body_name)
            if (
                get_body_position.type
                != SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
            ):
                return get_body_position
            result[body_name] = get_body_position.result
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info=f"Getting bodies positions of {body_names}",
            result=result,
        )

    @BaseSimulator.simulator_callback
    def get_bodies_quaternions(self, body_names: List[str]) -> SimulatorCallbackResult:
        """
        Get all bodies quaternions by body names

        :param body_names: The names of the bodies
        :return: A SimulatorCallbackResult with a list of all bodies quaternions as the result
        """
        result = {}
        for body_name in body_names:
            get_body_quaternion = self.get_body_quaternion(body_name)
            if (
                get_body_quaternion.type
                != SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
            ):
                return get_body_quaternion
            result[body_name] = get_body_quaternion.result
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info=f"Getting bodies quaternions of {body_names}",
            result=result,
        )

    @BaseSimulator.simulator_callback
    def get_body_joints(self, body_name):
        """
        Get all joints of a body by its name

        :param body_name: The name of the body
        :return: A SimulatorCallbackResult with a list of all joints as the result
        """
        get_body = self.get_body(body_name)
        if (
            get_body.type
            != SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
        ):
            return get_body
        body = get_body.result
        body_id = body.id
        jntids = self._mj_model.body(body_id).jntadr
        joints = [self._mj_model.joint(jntid) for jntid in jntids if jntid != -1]
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info=f"Getting body {body_name} joints",
            result=joints,
        )

    @BaseSimulator.simulator_callback
    def set_body_position(
        self, body_name: str, position: numpy.ndarray
    ) -> SimulatorCallbackResult:
        """
        Set the position of a body by its name

        :param body_name: The name of the body
        :param position: The position of the body
        :return: A SimulatorCallbackResult indicating the success or failure of the operation
        """
        get_body_joints = self.get_body_joints(body_name)
        if (
            get_body_joints.type
            != SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
        ):
            return get_body_joints
        joints = get_body_joints.result
        if len(joints) != 1:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                info=f"Body {body_name} doesn't have exactly one joint",
            )
        joint = joints[0]
        if joint.type != mujoco.mjtJoint.mjJNT_FREE:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                info=f"Body {body_name} joint is not free",
            )
        qpos_adr = joint.qposadr[0]
        if numpy.isclose(self._mj_data.qpos[qpos_adr : qpos_adr + 3], position).all():
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
                info=f"Body {body_name} is already at position {position}",
            )
        self._mj_data.qpos[qpos_adr : qpos_adr + 3] = position
        if self.simulation_thread is None:
            mujoco.mj_step1(self._mj_model, self._mj_data)
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA,
            info=f"Set body {body_name} to position {position}",
        )

    @BaseSimulator.simulator_callback
    def set_bodies_positions(
        self, bodies_positions: Dict[str, numpy.ndarray]
    ) -> SimulatorCallbackResult:
        """
        Set the positions of all bodies by its name and positions

        :param bodies_positions: A dictionary of names and positions
        :return: A SimulatorCallbackResult indicating the success or failure of the operation
        """
        for body_name, position in bodies_positions.items():
            set_body_position = self.set_body_position(body_name, position)
            if set_body_position.type not in [
                SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
                SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA,
            ]:
                return set_body_position
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA,
            info=f"Set bodies positions of {bodies_positions}",
        )

    @BaseSimulator.simulator_callback
    def set_body_quaternion(
        self, body_name: str, quaternion: numpy.ndarray
    ) -> SimulatorCallbackResult:
        """
        Set the quaternion of a body by its name and quaternion

        :param body_name: The name of the body
        :param quaternion: The quaternion
        """
        get_body_joints = self.get_body_joints(body_name)
        if (
            get_body_joints.type
            != SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
        ):
            return get_body_joints
        joints = get_body_joints.result
        if len(joints) != 1:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                info=f"Body {body_name} doesn't have exactly one joint",
            )
        joint = joints[0]
        if joint.type != mujoco.mjtJoint.mjJNT_FREE:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                info=f"Body {body_name} joint is not free",
            )
        qpos_adr = joint.qposadr[0]
        if numpy.isclose(
            self._mj_data.qpos[qpos_adr + 3 : qpos_adr + 7], quaternion
        ).all():
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
                info=f"Body {body_name} is already at quaternion {quaternion}",
            )
        self._mj_data.qpos[qpos_adr + 3 : qpos_adr + 7] = quaternion
        if self.simulation_thread is None:
            mujoco.mj_step1(self._mj_model, self._mj_data)
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA,
            info=f"Set body {body_name} to quaternion (WXYZ) {quaternion}",
        )

    @BaseSimulator.simulator_callback
    def set_bodies_quaternions(
        self, bodies_quaternions: Dict[str, numpy.ndarray]
    ) -> SimulatorCallbackResult:
        """
        Set the quaternions of all bodies by its name and quaternions

        :param bodies_quaternions: A dictionary of names and quaternions
        :return: A SimulatorCallbackResult indicating the success or failure of the operation
        """
        for body_name, quaternion in bodies_quaternions.items():
            set_body_quaternion = self.set_body_quaternion(body_name, quaternion)
            if set_body_quaternion.type not in [
                SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
                SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA,
            ]:
                return set_body_quaternion
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA,
            info=f"Set bodies quaternions of {bodies_quaternions}",
        )

    @BaseSimulator.simulator_callback
    def get_all_joint_names(
        self, joint_types: Optional[List[mujoco.mjtJoint]] = None
    ) -> SimulatorCallbackResult:
        """
        Get the names of all joints, filtered by types

        :param joint_types: The types of joints to filter by, if None, all joints are returned
        :return: A SimulatorCallbackResult with a list of all joint names as the result
        """
        if joint_types is None:
            joint_types = [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]
        result = [
            self._mj_model.joint(joint_id).name
            for joint_id in range(self._mj_model.njnt)
            if self._mj_model.joint(joint_id).type in joint_types
        ]
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info="Getting all joint names",
            result=result,
        )

    @BaseSimulator.simulator_callback
    def get_joint(
        self, joint_name: str, joint_types: Optional[mujoco.mjtJoint] = None
    ) -> SimulatorCallbackResult:
        """
        Get a joint by its name and type

        :param joint_name: The name of the joint
        :param joint_types: The types of joints to get, if None, all joints are returned
        :return: A SimulatorCallbackResult with a mujoco.MjsJoint as the result
        """
        if joint_types is None:
            joint_types = [
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ]
        joint_id = mujoco.mj_name2id(
            m=self._mj_model, type=mujoco.mjtObj.mjOBJ_JOINT, name=joint_name
        )
        if joint_id == -1:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                info=f"Joint {joint_name} not found",
            )
        if self._mj_model.joint(joint_id).type not in joint_types:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                info=f"Joint {joint_name} does not have allowed joint types {joint_types}",
            )
        joint = self._mj_data.joint(joint_id)
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info=f"Getting joint {joint_name}",
            result=joint,
        )

    @BaseSimulator.simulator_callback
    def get_joint_value(self, joint_name: str) -> SimulatorCallbackResult:
        """
        Get the joint value by its name

        :param joint_name: The name of the joint
        :return: A SimulatorCallbackResult with the joint value as the result
        """
        get_joint = self.get_joint(joint_name)
        if (
            get_joint.type
            != SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
        ):
            return get_joint
        joint = get_joint.result
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info=f"Getting joint value of {joint_name}",
            result=joint.qpos[0],
        )

    @BaseSimulator.simulator_callback
    def get_joints_values(self, joint_names: List[str]) -> SimulatorCallbackResult:
        """
        Get values of joints by their names

        :param joint_names: The names of the joints
        :return: A SimulatorCallbackResult indicating the success or failure of the operation
        """
        result = {}
        for joint_name in joint_names:
            joint_value = self.get_joint_value(joint_name)
            if (
                joint_value.type
                != SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
            ):
                return joint_value
            result[joint_name] = joint_value.result
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info=f"Getting joints values of {joint_names}",
            result=result,
        )

    @BaseSimulator.simulator_callback
    def set_joint_value(self, joint_name: str, value: float) -> SimulatorCallbackResult:
        """
        Set the value of a joint by its name

        :param joint_name: The name of the joint
        :param value: The new value to set
        :return: A SimulatorCallbackResult indicating the success or failure of the operation
        """
        get_joint = self.get_joint(joint_name)
        if (
            get_joint.type
            != SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
        ):
            return get_joint
        joint = get_joint.result
        if numpy.isclose(joint.qpos[0], value):
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
                info=f"Joint {joint_name} is already at value {value}",
            )
        joint.qpos[0] = value
        if self.simulation_thread is None:
            mujoco.mj_step1(self._mj_model, self._mj_data)
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA,
            info=f"Set joint {joint_name} to value {value}",
        )

    @BaseSimulator.simulator_callback
    def set_joints_values(
        self, joints_values: Dict[str, float]
    ) -> SimulatorCallbackResult:
        """
        Set the values of joints by their names

        :param joints_values: The names of the joints
        :return: A SimulatorCallbackResult indicating the success or failure of the operation
        """
        for joint_name, value in joints_values.items():
            set_joint_value = self.set_joint_value(joint_name, value)
            if set_joint_value.type not in [
                SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
                SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA,
            ]:
                return set_joint_value
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA,
            info=f"Set joints values of {joints_values}",
        )

    @BaseSimulator.simulator_callback
    def get_all_actuator_names(self) -> SimulatorCallbackResult:
        """
        Get all actuator names

        :return: A SimulatorCallbackResult with a list of actuator names as the result
        """
        result = [
            self._mj_model.actuator(actuator_id).name
            for actuator_id in range(self._mj_model.nu)
        ]
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info="Getting all actuator names",
            result=result,
        )

    @BaseSimulator.simulator_callback
    def get_actuator(self, actuator_name: str) -> SimulatorCallbackResult:
        """
        Get an actuator by its name

        :param actuator_name: The name of the actuator
        :return: A SimulatorCallbackResult with a mujoco.MjsActuator as the result
        """
        actuator_id = mujoco.mj_name2id(
            m=self._mj_model, type=mujoco.mjtObj.mjOBJ_ACTUATOR, name=actuator_name
        )
        if actuator_id == -1:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                info=f"Actuator {actuator_name} not found",
            )
        actuator = self._mj_data.actuator(actuator_id)
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info=f"Getting actuator {actuator_name}",
            result=actuator,
        )

    @BaseSimulator.simulator_callback
    def attach(
        self,
        body_1_name: str,
        body_2_name: Optional[str] = None,
        relative_position: Optional[numpy.ndarray] = None,
        relative_quaternion: Optional[numpy.ndarray] = None,
    ) -> SimulatorCallbackResult:
        """
        Attach body 1 to body 2 with the given relative position and quaternion.
        If body 2 is None, body 1 will be attached to the world.
        If relative position or quaternion is None, they will be computed from the current state of the simulation.

        :param body_1_name: The name of the body 1
        :param body_2_name: The name of the body 2
        :param relative_position: The relative position of the body 1
        :param relative_quaternion: The relative quaternion of the body 1
        :return: A SimulatorCallbackResult indicating the success or failure of the operation
        """
        body_1_id = mujoco.mj_name2id(
            m=self._mj_model, type=mujoco.mjtObj.mjOBJ_BODY, name=body_1_name
        )
        if body_1_id == -1:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_BEFORE_EXECUTION_ON_MODEL,
                info=f"Body 1 {body_1_name} not found",
            )
        if body_2_name is not None:
            body_2_id = mujoco.mj_name2id(
                m=self._mj_model, type=mujoco.mjtObj.mjOBJ_BODY, name=body_2_name
            )
            if body_2_id == -1:
                return SimulatorCallbackResult(
                    type=SimulatorCallbackResult.ResultType.FAILURE_BEFORE_EXECUTION_ON_MODEL,
                    info=f"Body 2 {body_2_name} not found",
                )
        else:
            body_2_id = 0
            body_2_name = mujoco.mj_id2name(
                m=self._mj_model, type=mujoco.mjtObj.mjOBJ_BODY, id=body_2_id
            )
        if body_1_id == body_2_id:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_BEFORE_EXECUTION_ON_MODEL,
                info="Body 1 and body 2 are the same",
            )

        body_1_xpos = self._mj_data.body(body_1_id).xpos
        body_1_xquat = self._mj_data.body(body_1_id).xquat
        body_2_xpos = self._mj_data.body(body_2_id).xpos
        body_2_xquat = self._mj_data.body(body_2_id).xquat

        body_1_in_2_pos = numpy.zeros(3)
        body_1_in_2_quat = numpy.zeros(4)

        body_2_neq_quat = numpy.zeros(4)
        mujoco.mju_negQuat(body_2_neq_quat, body_2_xquat)
        mujoco.mju_sub3(body_1_in_2_pos, body_1_xpos, body_2_xpos)
        mujoco.mju_rotVecQuat(body_1_in_2_pos, body_1_in_2_pos, body_2_neq_quat)
        mujoco.mju_mulQuat(body_1_in_2_quat, body_2_neq_quat, body_1_xquat)

        body_1 = self._mj_model.body(body_1_id)
        if relative_position is not None:
            if len(relative_position) != 3 or any(
                not isinstance(x, (int, float)) for x in relative_position
            ):
                return SimulatorCallbackResult(
                    type=SimulatorCallbackResult.ResultType.FAILURE_BEFORE_EXECUTION_ON_MODEL,
                    info=f"Invalid relative position {relative_position}",
                )
        else:
            if body_1.parentid[0] == body_2_id:
                relative_position = body_1.pos
            else:
                relative_position = body_1_in_2_pos

        if relative_quaternion is not None:
            if len(relative_quaternion) != 4 or any(
                not isinstance(x, (int, float)) for x in relative_quaternion
            ):
                return SimulatorCallbackResult(
                    type=SimulatorCallbackResult.ResultType.FAILURE_BEFORE_EXECUTION_ON_MODEL,
                    info=f"Invalid relative quaternion {relative_quaternion}",
                )
        else:
            if body_1.parentid[0] == body_2_id:
                relative_quaternion = body_1.quat
            else:
                relative_quaternion = body_1_in_2_quat

        if (
            body_1.parentid[0] == body_2_id
            and numpy.isclose(body_1.pos, relative_position).all()
            and numpy.isclose(body_1.quat, relative_quaternion).all()
        ):
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
                info=f"Body 1 {body_1_name} is already attached to body 2 {body_2_name}",
            )
        if mujoco.mj_version() >= 330:
            body_1_spec = self._mj_spec.body(body_1_name)
        else:
            body_1_spec = self._mj_spec.find_body(body_1_name)
        if body_1_spec is None:
            self.log_warning(
                f"Body 1 {body_1_name} not found in the model specification, this is a bug from MuJoCo"
            )
            body_1_spec = next(
                body for body in self._mj_spec.bodies if body.name == body_1_name
            )
        if mujoco.mj_version() >= 330:
            body_2_spec = self._mj_spec.body(body_2_name)
        else:
            body_2_spec = self._mj_spec.find_body(body_2_name)
        if body_2_spec is None:
            self.log_warning(
                f"Body 2 {body_2_name} not found in the model specification, this is a bug from MuJoCo"
            )
            body_2_spec = next(
                body for body in self._mj_spec.bodies if body.name == body_2_name
            )
        dummy_prefix = "AVeryDumbassPrefixThatIsUnlikelyToBeUsedBecauseMuJoCoRequiresIt"
        body_1_spec_new = body_2_spec.add_body(
            name=f"{dummy_prefix}{body_1_name}",
            pos=relative_position,
            quat=relative_quaternion,
        )
        # for body_child in body_1_spec_copy.bodies:
        #     body_2_spec.add_body(body_child)
        for geom_child in body_1_spec.geoms:
            body_1_spec_new.add_geom(
                name=f"{dummy_prefix}{geom_child.name}",
                pos=geom_child.pos,
                quat=geom_child.quat,
                type=geom_child.type,
                size=geom_child.size,
                rgba=geom_child.rgba,
                conaffinity=geom_child.conaffinity,
                condim=geom_child.condim,
                contype=geom_child.contype,
                density=geom_child.density,
                friction=geom_child.friction,
                meshname=geom_child.meshname,
            )
        # for site_child in body_1_spec_copy.sites:
        #     body_1_spec.add_site(site_child)

        # body_2_frame = body_2_spec.add_frame()
        # body_1_spec_new = body_2_frame.attach_body(body_1_spec, dummy_prefix, "")
        # body_1_spec_new.pos = relative_position
        # body_1_spec_new.quat = relative_quaternion
        if mujoco.mj_version() < 335:
            self._mj_spec.detach_body(body_1_spec)
        else:
            self._mj_spec.delete(body_1_spec)
        self._fix_prefix_and_recompile(body_1_spec_new, dummy_prefix, body_1_name)
        if self.simulation_thread is None:
            mujoco.mj_step1(self._mj_model, self._mj_data)

        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL,
            info=f"Attached body 1 {body_1_name} to body 2 {body_2_name} "
            f"at relative position {relative_position}, relative quaternion {relative_quaternion}",
        )

    @BaseSimulator.simulator_callback
    def detach(
        self, body_name: str, add_freejoint: bool = True
    ) -> SimulatorCallbackResult:
        """
        Detach a body from its parent body and attach it to the world with the same absolute position and quaternion.
        If add_freejoint is True, a free joint will be added between the body and the world, allowing the body to move freely after detaching.
        If add_freejoint is False, the body will be fixed to the world after detaching.
        Note that due to a bug in MuJoCo, if the detached body has children,
        the children will not be properly detached and will keep the same name,
        which may cause conflicts when attaching them to another body with the same name.
        The workaround is to add a dummy prefix to the body and its children before recompiling,
        then remove the dummy prefix after recompiling.

        :param body_name: The name of the body to detach
        :param add_freejoint: Whether to add a free joint between the body and the world after detaching
        :return: A SimulatorCallbackResult indicating the success or failure of the operation
        """
        body_id = mujoco.mj_name2id(
            m=self._mj_model, type=mujoco.mjtObj.mjOBJ_BODY, name=body_name
        )
        if body_id == -1:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_BEFORE_EXECUTION_ON_MODEL,
                info=f"Body {body_name} not found",
            )

        parent_body_id = self._mj_model.body(body_id).parentid[0]
        if parent_body_id == 0:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
                info=f"Body {body_name} is already detached",
            )

        absolute_position = self._mj_data.body(body_id).xpos
        absolute_quaternion = self._mj_data.body(body_id).xquat
        parent_body_name = self._mj_model.body(parent_body_id).name
        if mujoco.mj_version() >= 330:
            body_spec = self._mj_spec.body(body_name)
        else:
            body_spec = self._mj_spec.find_body(body_name)
        dummy_prefix = "AVeryDumbassPrefixThatIsUnlikelyToBeUsedBecauseMuJoCoRequiresIt"
        body_spec_new = self._mj_spec.worldbody.add_body(
            name=f"{dummy_prefix}{body_name}",
            pos=absolute_position,
            quat=absolute_quaternion,
        )
        # for body_child in body_1_spec_copy.bodies:
        #     body_2_spec.add_body(body_child)
        for geom_child in body_spec.geoms:
            body_spec_new.add_geom(
                name=f"{dummy_prefix}{geom_child.name}",
                pos=geom_child.pos,
                quat=geom_child.quat,
                type=geom_child.type,
                size=geom_child.size,
                rgba=geom_child.rgba,
                conaffinity=geom_child.conaffinity,
                condim=geom_child.condim,
                contype=geom_child.contype,
                density=geom_child.density,
                friction=geom_child.friction,
                meshname=geom_child.meshname,
            )
        # for site_child in body_1_spec_copy.sites:
        #     body_1_spec.add_site(site_child)
        if add_freejoint:
            body_spec_new.add_freejoint()
        if mujoco.mj_version() < 335:
            self._mj_spec.detach_body(body_spec)
        else:
            self._mj_spec.delete(body_spec)
        self._fix_prefix_and_recompile(body_spec_new, dummy_prefix, body_name)
        if self.simulation_thread is None:
            mujoco.mj_step1(self._mj_model, self._mj_data)

        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL,
            info=f"Detached body {body_name} from body {parent_body_name}",
        )

    @BaseSimulator.simulator_callback
    def get_children_ids(self, body_id: int) -> SimulatorCallbackResult:
        """
        Get all children body ids of a body by its id

        :param body_id: id of the body to get the children ids
        :return Set[int]: all children ids
        """
        children_ids = set()
        for child_body_id in range(body_id + 1, self._mj_model.nbody):
            parent_body_id = self._mj_model.body(child_body_id).parentid[0]
            if parent_body_id == body_id or parent_body_id in children_ids:
                children_ids.add(child_body_id)
            else:
                break
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            result=children_ids,
            info=f"Found {len(children_ids)} children of body id {body_id}",
        )

    @BaseSimulator.simulator_callback
    def get_contact_bodies(
        self, body_name: str, including_children: bool = True
    ) -> SimulatorCallbackResult:
        """
        Get the names of all bodies that are in contact with a body by its name.

        :param body_name: name of the body to get the contact bodies
        :param including_children: include all children bodies
        :return A SimulatorCallbackResult with a set of contact body names as the result
        """
        body_id = mujoco.mj_name2id(
            m=self._mj_model, type=mujoco.mjtObj.mjOBJ_BODY, name=body_name
        )
        if body_id == -1:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                info=f"Body {body_name} not found",
            )

        body_ids = {body_id}
        if including_children:
            body_ids.update(self.get_children_ids(body_id).result)

        contact_body_ids = set()
        for contact in self._mj_data.contact:
            geom_1_id = contact.geom1
            geom_2_id = contact.geom2
            body_1_id = self._mj_model.geom_bodyid[geom_1_id]
            body_2_id = self._mj_model.geom_bodyid[geom_2_id]
            if body_1_id in body_ids and body_2_id not in body_ids:
                contact_body_ids.add(body_2_id)
            elif body_2_id in body_ids and body_1_id not in body_ids:
                contact_body_ids.add(body_1_id)

        contact_body_names = {
            self._mj_model.body(contact_body_id).name
            for contact_body_id in contact_body_ids
        }
        including_children_str = (
            f"with its {len(body_ids) - 1} children"
            if including_children
            else "without children"
        )
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info=f"There are {len(contact_body_names)} contact bodies with body {body_name} {including_children_str}",
            result=contact_body_names,
        )

    @BaseSimulator.simulator_callback
    def get_body_root_name(self, body_name: str) -> SimulatorCallbackResult:
        """
        Get the name of a body root by its name

        :param body_name: name of the body to get the body root name
        :return A SimulatorCallbackResult with a string of the body root name as the result
        """
        body_id = mujoco.mj_name2id(
            m=self._mj_model, type=mujoco.mjtObj.mjOBJ_BODY, name=body_name
        )
        if body_id == -1:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                info=f"Body {body_name} not found",
            )
        body_root_id = body_id
        while self._mj_model.body(body_root_id).parentid[0] != 0:
            body_root_id = self._mj_model.body(body_root_id).parentid[0]
        body_root_name = self._mj_model.body(body_root_id).name
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info=f"Getting root body name of {body_name}",
            result=body_root_name,
        )

    @BaseSimulator.simulator_callback
    def get_contact_points(
        self,
        body_names: List[str],
        including_children: bool = True,
        contact_style: Union[
            SimulatorCallbackResult.OutputType, str
        ] = SimulatorCallbackResult.OutputType.PYBULLET,
    ) -> SimulatorCallbackResult:
        """
        Get the contact points between bodies by their names.

        :param body_names: list of names of bodies
        :param including_children: whether to include children or not
        :param contact_style: contact style. Only `SimulatorCallbackResult.OutputType.PYBULLET` is currently supported, which returns contact points in a PyBullet\-compatible format.
        :return: A SimulatorCallbackResult indicating the success or failure of the operation. On success, the result contains a list of contact points in the specified format.
        """
        if isinstance(contact_style, str):
            contact_style = SimulatorCallbackResult.OutputType(contact_style)
        if contact_style != SimulatorCallbackResult.OutputType.PYBULLET:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                info=f"Contact style {contact_style} is not supported",
            )

        if len(body_names) == 0:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                info="Body 1 names are empty",
            )

        body_root_map = {"world": "world"}
        for body_name in body_names:
            get_body_root_name = self.get_body_root_name(body_name)
            if (
                get_body_root_name.type
                != SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
            ):
                return get_body_root_name
            body_root_map[body_name] = get_body_root_name.result

            if including_children:
                body_id = mujoco.mj_name2id(
                    m=self._mj_model, type=mujoco.mjtObj.mjOBJ_BODY, name=body_name
                )
                if body_id != 0:
                    body_root_map.update(
                        {
                            self._mj_model.body(
                                child_body_id
                            ).name: get_body_root_name.result
                            for child_body_id in self.get_children_ids(body_id).result
                        }
                    )
                else:
                    for child_body_id in range(self._mj_model.nbody):
                        child_body = self._mj_model.body(child_body_id)
                        body_root_map.update(
                            {
                                child_body.name: self.get_body_root_name(
                                    child_body.name
                                ).result
                            }
                        )
        body_names = list(body_root_map.keys())

        contact_points = []
        contact_bodies = set()
        for contact_id in range(self._mj_data.ncon):
            contact = self._mj_data.contact[contact_id]
            if contact.exclude != 0 and contact.exclude != 1:
                continue
            geom_A_id = contact.geom[1]
            geom_B_id = contact.geom[0]
            body_A_id = self._mj_model.geom(geom_A_id).bodyid[0]
            body_B_id = self._mj_model.geom(geom_B_id).bodyid[0]
            body_A_name = self._mj_model.body(body_A_id).name
            body_B_name = self._mj_model.body(body_B_id).name
            if body_A_name not in body_names or body_B_name not in body_names:
                continue
            contact_bodies.add(body_A_name)
            contact_bodies.add(body_B_name)

            contact_position = contact.pos
            contact_normal = contact.frame[0:3]
            contact_distance = contact.dist
            contact_effort = numpy.zeros(6)
            mujoco.mj_contactForce(
                self._mj_model, self._mj_data, contact_id, contact_effort
            )

            contact_point = {
                "bodyUniqueNameA": body_root_map[body_A_name],
                "bodyUniqueNameB": body_root_map[body_B_name],
                "linkNameA": body_A_name,
                "linkNameB": body_B_name,
                "positionOnA": contact_position + contact_normal * contact_distance / 2,
                "positionOnB": contact_position - contact_normal * contact_distance / 2,
                "contactNormalOnB": contact_normal,
                "contactDistance": contact_distance,
                "normalForce": contact_effort[0],
                "lateralFriction1": contact_effort[1],
                "lateralFrictionDir1": contact.frame[3:6],
                "lateralFriction2": contact_effort[2],
                "lateralFrictionDir2": contact.frame[6:9],
            }
            contact_points.append(contact_point)

        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info=f"There are {len(contact_points)} contact points of bodies {body_names}.",
            result=contact_points,
        )

    @BaseSimulator.simulator_callback
    def ray_test(
        self,
        ray_from_position: List[float],
        ray_to_position: List[float],
        ray_style: Union[
            SimulatorCallbackResult.OutputType, str
        ] = SimulatorCallbackResult.OutputType.PYBULLET,
    ) -> SimulatorCallbackResult:
        """
        Cast a ray in the scene and return intersections in a PyBullet\-style format.

        :param ray_from_position: Start position of the ray as a list \[x, y, z\].
        :param ray_to_position: End position of the ray as a list \[x, y, z\].
        :param ray_style: Output style. Only `SimulatorCallbackResult.OutputType.PYBULLET` is currently supported.
        :return: A `SimulatorCallbackResult` indicating success or failure. On success, the result contains a list of hit records in PyBullet\-compatible format.
        """
        if isinstance(ray_style, str):
            ray_style = SimulatorCallbackResult.OutputType(ray_style)
        if ray_style != SimulatorCallbackResult.OutputType.PYBULLET:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                info=f"Ray style {ray_style} is not supported",
            )
        if len(ray_from_position) != 3 or len(ray_to_position) != 3:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                info=f"Invalid ray from position {ray_from_position} or ray to position {ray_to_position}",
            )
        pnt = numpy.array(ray_from_position)
        vec = numpy.array(ray_to_position) - pnt
        geomgroup = numpy.ones(6, numpy.uint8)
        vec_len = mujoco.mju_normalize3(vec)
        geom_id = numpy.zeros(1, numpy.int32)

        dist = mujoco.mj_ray(
            m=self._mj_model,
            d=self._mj_data,
            pnt=pnt,
            vec=vec,
            geomgroup=geomgroup,
            flg_static=1,
            bodyexclude=-1,
            geomid=geom_id,
        )
        if geom_id.item() >= 0 and dist <= vec_len:
            geom = self._mj_model.geom(geom_id.item())
            hit_normal = None  # TODO: get hit normal
            body_id = geom.bodyid[0]
            body_name = self._mj_model.body(body_id).name
            get_body_root_name = self.get_body_root_name(body_name)
            if (
                get_body_root_name.type
                != SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
            ):
                return get_body_root_name
            root_body_name = get_body_root_name.result
            result = {
                "objectUniqueName": root_body_name,
                "linkName": body_name,
                "hit_fraction": dist / vec_len,
                "hit_position": pnt + vec * dist,
                "hit_normal": hit_normal,
            }
        else:
            result = None
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info=f"Single raycast from {ray_from_position} to {ray_to_position}",
            result=result,
        )

    @BaseSimulator.simulator_callback
    def ray_test_batch(
        self,
        ray_from_position: List[float],
        ray_to_positions: List[List[float]],
        parent_link_name: Optional[str] = None,
        ray_style: Union[
            SimulatorCallbackResult.OutputType, str
        ] = SimulatorCallbackResult.OutputType.PYBULLET,
    ) -> SimulatorCallbackResult:
        """
        Cast a batch of rays from a single origin to multiple end positions.

        :param ray_from_position: A list \[x, y, z\] representing the ray origin in world coordinates.
        :param ray_to_positions: A list of \[x, y, z\] lists representing ray end points in world coordinates.
        :param parent_link_name: Optional name of a link the rays are conceptually attached to (for compatibility).
        :param ray_style: Output style, must be `SimulatorCallbackResult.OutputType.PYBULLET`.

        :return: A SimulatorCallbackResult whose `result` is a list of hit results in PyBullet\-style dicts.
        """
        if isinstance(ray_style, str):
            ray_style = SimulatorCallbackResult.OutputType(ray_style)
        if ray_style != SimulatorCallbackResult.OutputType.PYBULLET:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                info=f"Ray style {ray_style} is not supported",
            )
        pnt = numpy.array(ray_from_position)
        N = len(ray_to_positions)
        vec = numpy.array(
            [numpy.array(ray_to_position) - pnt for ray_to_position in ray_to_positions]
        )
        geomgroup = numpy.ones(6, numpy.uint8)
        if parent_link_name is not None:
            body_id = mujoco.mj_name2id(
                m=self._mj_model, type=mujoco.mjtObj.mjOBJ_BODY, name=parent_link_name
            )
            if body_id == -1:
                return SimulatorCallbackResult(
                    type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                    info=f"Parent link {parent_link_name} not found",
                )
            body = self._mj_model.body(body_id)
            mujoco.mju_mulMatVec3(res=pnt, mat=body.xmat, vec=pnt)
            for i in range(N):
                mujoco.mju_mulMatVec3(res=vec[i], mat=body.xmat, vec=vec[i])
        geom_id = numpy.zeros(N, numpy.int32)
        dist = numpy.zeros(N, numpy.float64)
        if mujoco.mj_version() < 3005000:
            mujoco.mj_multiRay(
                m=self._mj_model,
                d=self._mj_data,
                pnt=pnt,
                vec=vec.flatten(),
                geomgroup=geomgroup,
                flg_static=1,
                bodyexclude=-1,
                geomid=geom_id,
                dist=dist,
                nray=N,
                cutoff=mujoco.mjMAXVAL,
            )
        else:
            raise NotImplementedError(
                "mj_multiRay implementation for mujoco version >= 3.5.0 is not implemented yet"
            )
        results = []
        for i in range(N):
            if geom_id[i] < 0:
                results.append(None)
            else:
                geom = self._mj_model.geom(geom_id[i])
                body_id = geom.bodyid[0]
                body_name = self._mj_model.body(body_id).name
                get_body_root_name = self.get_body_root_name(body_name)
                if (
                    get_body_root_name.type
                    != SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
                ):
                    return get_body_root_name
                root_body_name = get_body_root_name.result
                hit_normal = None  # TODO: get hit normal
                results.append(
                    {
                        "objectUniqueName": root_body_name,
                        "linkName": body_name,
                        "hit_fraction": dist[i],
                        "hit_position": pnt + vec[i] * dist[i],
                        "hit_normal": hit_normal,
                    }
                )
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info=f"Batch raycast from {ray_from_position} to {ray_to_positions}",
            result=results,
        )

    @BaseSimulator.simulator_callback
    def save(
        self, file_path: Optional[str] = None, key_name: Optional[str] = None
    ) -> SimulatorCallbackResult:
        """
        Save the current simulation state to a file or to a keyframe.

        :param file_path: The path to the file to save the simulation state to. If None, the simulation state will be saved to a keyframe with the given key_name.
        :param key_name: The name of the keyframe to save the simulation state to. If None, the simulation state will be saved to the default keyframe with id 0.
        """
        if key_name is None:
            key_id = 0
        else:
            key_id = mujoco.mj_name2id(
                m=self._mj_model, type=mujoco.mjtObj.mjOBJ_KEY, name=key_name
            )
            if key_id == -1:
                self._mj_spec.add_key(
                    name=key_name,
                    qpos=self._mj_data.qpos,
                    qvel=self._mj_data.qvel,
                    act=self._mj_data.act,
                    ctrl=self._mj_data.ctrl,
                    time=self.current_simulation_time,
                )
                self._mj_model, self._mj_data = self._mj_spec.recompile(
                    self._mj_model, self._mj_data
                )
                if not self.headless:
                    self._renderer._sim().load(self._mj_model, self._mj_data, "")
                    if self.simulation_thread is None:
                        mujoco.mj_step1(self._mj_model, self._mj_data)
                key_id = mujoco.mj_name2id(
                    m=self._mj_model, type=mujoco.mjtObj.mjOBJ_KEY, name=key_name
                )
        mujoco.mj_setKeyframe(self._mj_model, self._mj_data, key_id)
        if file_path is not None:
            if os.path.exists(file_path):
                os.remove(file_path)
            xml_string = self._mj_spec.to_xml()
            with open(file_path, "w") as f:
                f.write(xml_string)
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
                info=f"Saved simulation with key {key_name} to {file_path}",
                result=key_id,
            )
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info=f"Saved simulation with key {key_name}",
            result=key_id,
        )

    @BaseSimulator.simulator_callback
    def load(
        self, file_path: Optional[str] = None, key_id: int = 0
    ) -> SimulatorCallbackResult:
        """
        Load a simulation state from a file or from a keyframe.

        :param file_path: The path to the file to load the simulation state from. If None, the simulation state will be loaded from the keyframe with the given key_id.
        :param key_id: The ID of the keyframe to load the simulation state from.

        :return: SimulatorCallbackResult with the key_id of the loaded simulation state if successful as a result.
        """
        if file_path is not None:
            if not os.path.exists(file_path):
                return SimulatorCallbackResult(
                    type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                    info=f"File {file_path} not found",
                )
            self._mj_model, self._mj_data = self._mj_spec.recompile(
                self._mj_model, self._mj_data
            )
            if not self.headless:
                self._renderer._sim().load(self._mj_model, self._mj_data, "")
                if self.simulation_thread is None:
                    mujoco.mj_step1(self._mj_model, self._mj_data)
            if key_id >= self._mj_model.nkey:
                return SimulatorCallbackResult(
                    type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                    info=f"Key {key_id} not found",
                )
            mujoco.mj_resetDataKeyframe(self._mj_model, self._mj_data, key_id)
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA,
                info=f"Loaded simulation with key_id {key_id} from {file_path}",
                result=key_id,
            )
        if key_id >= self._mj_model.nkey:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                info=f"Key {key_id} not found",
            )
        mujoco.mj_resetDataKeyframe(self._mj_model, self._mj_data, key_id)
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA,
            info=f"Loaded simulation with key_id {key_id}",
            result=key_id,
        )

    @BaseSimulator.simulator_callback
    def capture_rgb(
        self, camera_name: str = None, height=240, width=320
    ) -> SimulatorCallbackResult:
        """
        This method returns a NumPy uint8 array of RGB values.

        :param camera_name: The name of the camera to capture the RGB data from. If None, the default camera is used.
        :param height: The height of the image to capture.
        :param width: The width of the image to capture.

        :return: A SimulatorCallbackResult object with the captured RGB data.
        """
        with mujoco.Renderer(self._mj_model, height, width) as renderer:
            if camera_name is not None:
                renderer.update_scene(self._mj_data, camera_name)
            else:
                renderer.update_scene(self._mj_data)
            rgb = renderer.render()
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info="Captured RGB data",
            result=rgb,
        )

    @BaseSimulator.simulator_callback
    def capture_depth(
        self, camera_name: str = None, height=240, width=320
    ) -> SimulatorCallbackResult:
        """
        This method returns a NumPy float array of depth values (in meters).

        :param camera_name: The name of the camera to capture the depth from. If None, the default camera is used.
        :param height: The height of the image to capture.
        :param width: The width of the image to capture.

        :return: A SimulatorCallbackResult object with the captured depth data.
        """
        with mujoco.Renderer(self._mj_model, height, width) as renderer:
            renderer.enable_depth_rendering()
            if camera_name is not None:
                renderer.update_scene(self._mj_data, camera_name)
            else:
                renderer.update_scene(self._mj_data)
            depth = renderer.render()
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info="Captured depth data",
            result=depth,
        )

    @BaseSimulator.simulator_callback
    def capture_segmentation(
        self, camera_name: str = None, height=240, width=320
    ) -> SimulatorCallbackResult:
        """
        This method returns a 2-channel NumPy int32 array of label values where the pixels of each object are labeled with the pair (mjModel ID, mjtObj enum object type).
        Background pixels are labeled (-1, -1).

        :param camera_name: The name of the camera to capture the segmentation data from. If None, the default camera is used.
        :param height: The height of the rendered image.
        :param width: The width of the rendered image.

        :return: A SimulatorCallbackResult object with the segmentation data as the result.
        """

        with mujoco.Renderer(self._mj_model, height, width) as renderer:
            renderer.enable_segmentation_rendering()
            if camera_name is not None:
                renderer.update_scene(self._mj_data, camera_name)
            else:
                renderer.update_scene(self._mj_data)
            segmentation = renderer.render()
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info="Captured segmentation data",
            result=segmentation,
        )

    @BaseSimulator.simulator_callback
    def enable_contact(
        self, body_1_name: str, body_2_name: str
    ) -> SimulatorCallbackResult:
        """
        This method enables contact between two bodies.

        :param body_1_name: The name of the first body.
        :param body_2_name: The name of the second body.

        :return: A SimulatorCallbackResult object indicating the result of the operation.
        """

        body_1_id = mujoco.mj_name2id(
            m=self._mj_model, type=mujoco.mjtObj.mjOBJ_BODY, name=body_1_name
        )
        if body_1_id == -1:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                info=f"Body 1 {body_1_name} not found",
            )
        body_2_id = mujoco.mj_name2id(
            m=self._mj_model, type=mujoco.mjtObj.mjOBJ_BODY, name=body_2_name
        )
        if body_2_id == -1:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                info=f"Body 2 {body_2_name} not found",
            )

        body_1 = self._mj_model.body(body_1_id)
        body_2 = self._mj_model.body(body_2_id)
        for geom_1_id in range(
            body_1.geomadr[0], body_1.geomadr[0] + body_1.geomnum[0]
        ):
            geom_1 = self._mj_model.geom(geom_1_id)
            if geom_1.contype[0] == 0 and geom_1.conaffinity[0] == 0:
                continue
            geom_1_name = geom_1.name
            for geom_2_id in range(
                body_2.geomadr[0], body_2.geomadr[0] + body_2.geomnum[0]
            ):
                geom_2 = self._mj_model.geom(geom_2_id)
                if geom_2.contype == 0 and geom_2.conaffinity == 0:
                    continue
                geom_2_name = geom_2.name
                pair_name = f"{geom_1_name}-{geom_2_name}"
                self._mj_spec.add_pair(
                    name=pair_name, geomname1=geom_1_name, geomname2=geom_2_name
                )
        self._mj_model, self._mj_data = self._mj_spec.recompile(
            self._mj_model, self._mj_data
        )
        if not self.headless:
            self._renderer._sim().load(self._mj_model, self._mj_data, "")
            if self.simulation_thread is None:
                mujoco.mj_step1(self._mj_model, self._mj_data)
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL,
            info=f"Enabled contact between {body_1_name} and {body_2_name}",
        )

    @BaseSimulator.simulator_callback
    def add_entity(
        self,
        entity_name: str,
        entity_type: str,
        entity_properties: Dict[str, Any],
        parent_name: Optional[str] = None,
        parent_type: str = "body",
    ) -> SimulatorCallbackResult:
        """
        This method adds a new entity to the simulation. The entity can be a body, joint, geom, frame, or site.

        :param entity_name: The name of the new entity.
        :param entity_type: The type of the new entity. Can be "body", "joint", "geom", "actuator", "frame", or "site".
        :param entity_properties: A dictionary of properties for the new entity.
        :param parent_name: The name of the parent body or frame to attach the new entity to. If None, the worldbody is used.
        :param parent_type: The type of the parent entity. Must be "body" for now.

        :return: A SimulatorCallbackResult object indicating the result of the operation.
        """

        if entity_type != "actuator":
            if parent_name is None:
                parent_name = "world"
                parent_type = "body"
            if mujoco.mj_version() >= 330:
                if parent_type == "body":
                    parent_spec = self._mj_spec.body(parent_name)
                elif parent_type == "frame":
                    parent_spec = self._mj_spec.frame(parent_name)
                else:
                    return SimulatorCallbackResult(
                        type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                        info=f"Parent type {parent_type} is not supported",
                    )
            else:
                if parent_type == "body":
                    parent_spec = self._mj_spec.find_body(parent_name)
                elif parent_type == "frame":
                    parent_spec = self._mj_spec.find_frame(parent_name)
                else:
                    return SimulatorCallbackResult(
                        type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                        info=f"Parent type {parent_type} is not supported",
                    )
            if parent_spec is None:
                return SimulatorCallbackResult(
                    type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                    info=f"Parent body {parent_name} not found",
                )
        else:
            parent_spec = self._mj_spec
        if entity_type not in ["body", "joint", "geom", "actuator", "frame", "site"]:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                info=f"Entity type {entity_type} is not supported",
            )
        if mujoco.mj_version() >= 330:
            entity_spec = self._mj_spec.__getattribute__(entity_type)(entity_name)
        else:
            entity_spec = self._mj_spec.__getattribute__(f"find_{entity_type}")(
                entity_name
            )
        if entity_spec:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                info=f"{entity_type} {entity_name} already exists",
            )
        try:
            entity = parent_spec.__getattribute__(f"add_{entity_type}")(
                name=entity_name, **entity_properties
            )
        except Exception as e:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_BEFORE_EXECUTION_ON_MODEL,
                info=f"Failed to create {entity_type} {entity_name} with properties {entity_properties}: {e}",
            )

        def do_spawn():
            self._mj_model, self._mj_data = self._mj_spec.recompile(
                self._mj_model, self._mj_data
            )
            if not self.headless:
                self._renderer._sim().load(self._mj_model, self._mj_data, "")
                if self.simulation_thread is None:
                    mujoco.mj_step1(self._mj_model, self._mj_data)
        if self.state == SimulatorState.RUNNING:
            self.pause()
            do_spawn()
            self.unpause()
        else:
            do_spawn()
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL,
            info=f"Spawned {entity_type} {entity.name} under parent body {parent_name} with properties {entity_properties}",
        )
