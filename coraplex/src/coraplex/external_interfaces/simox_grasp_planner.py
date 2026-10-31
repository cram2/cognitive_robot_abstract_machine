"""Adapt the external Simox service to object-relative grasp candidates."""

from __future__ import annotations

# %% imports

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from xml.etree.ElementTree import Element, ElementTree, SubElement

import numpy as np
from typing_extensions import TYPE_CHECKING

from coraplex.external_interfaces.exceptions import (
    InvalidSimoxResponse,
    SimoxPlanningFailed,
)
from semantic_digital_twin.adapters.ros.ros2_to_semdt_converters import (
    PoseToSemDTConverter,
)
from semantic_digital_twin.adapters.ros.semdt_to_ros2_converters import (
    PoseToRos2Converter,
)
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose

if TYPE_CHECKING:
    from geometry_msgs.msg import Pose as RosPose
    from rclpy.node import Node
    from semantic_digital_twin.robots.robot_parts import Arm
    from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
    from semantic_digital_twin.world_description.world_entity import Body

# %% service data


class SimoxInterface(StrEnum):
    """ROS interface identifiers for the external planner."""

    MODULE = "grasp_planner_msgs.srv"
    TOPIC = "/plan_grasp"


class SimoxObjectFormat(StrEnum):
    """Object document names understood by the external mesh loader."""

    MESH = "object.stl"
    DOCUMENT = "object.xml"
    ROOT = "ManipulationObject"
    VISUALIZATION = "Visualization"
    COLLISION = "CollisionModel"
    FILE = "File"
    STL = "stl"


@dataclass
class SimoxConfiguration:
    """External robot model and its tool calibration, expressed in metres."""

    robot_model_path: Path
    """Robot XML readable by the service process."""
    end_effector_name: str
    """End effector declared in the external robot XML."""
    simox_T_tool: HomogeneousTransformationMatrix
    """CoraPlex tool frame expressed in the external tool frame."""
    preshape_name: str = "open"
    """External model's open-hand configuration."""
    kinematic_chain_name: str = ""
    """Optional external IK chain; empty leaves reachability to CoraPlex."""
    num_grasps_to_plan: int = 15
    """Maximum number of candidate grasps requested."""
    quality_threshold: float = 0.001
    """Minimum quality requested from Simox."""
    timeout_ms: int = 30000
    """Planner time budget in milliseconds."""

    @classmethod
    def for_pr2(cls, robot_model_path: Path) -> SimoxConfiguration:
        """Calibrate the PR2 model whose tool is at 0.13 m with +90 degree pitch."""
        return cls(
            robot_model_path,
            "r_gripper",
            HomogeneousTransformationMatrix.from_xyz_rpy(z=0.05, pitch=-np.pi / 2),
        )


@dataclass
class SimoxRequest:
    """One planning request in the external robot's base frame."""

    configuration: SimoxConfiguration
    """Robot and search parameters."""
    object_model_path: Path
    """Object XML readable by the service process."""
    object_pose: Pose
    """Object placement in the robot base frame, in metres."""


@dataclass
class SimoxResponse:
    """Parallel arrays returned by Simox, with poses already in metres."""

    grasp_poses: list[Pose]
    """External tool goals in the requested robot base frame."""
    qualities: list[float]
    """Planner quality for each pose."""
    are_force_closure: list[bool]
    """Measured force-closure result for each pose."""


@dataclass
class RankedGrasp:
    """A standard grasp candidate together with its external evaluation."""

    candidate: GraspCandidate
    """Grasp expressed relative to the grasped object's root."""
    quality: float
    """Simox quality of the original pose."""
    is_force_closure: bool
    """Whether Simox measured force closure for this pose."""


# %% transport


@dataclass
class SimoxService(ABC):
    """The external planner boundary, independent of ROS lifecycle ownership."""

    @abstractmethod
    def plan(self, request: SimoxRequest) -> SimoxResponse:
        """Return evaluated tool poses, or raise when the request cannot complete."""


@dataclass
class RosSimoxService(SimoxService):
    """Call Simox through a borrowed node with an executor already spinning."""

    node: Node
    """Node owned and spun by the caller."""
    service_name: str = SimoxInterface.TOPIC
    """Service address."""
    availability_timeout: float = 3.0
    """Seconds to wait for discovery."""
    response_grace_period: float = 5.0
    """Additional seconds allowed for serialization and transport."""

    def plan(self, request: SimoxRequest) -> SimoxResponse:
        service_type = import_module(SimoxInterface.MODULE).PlanGrasp
        client = self.node.create_client(service_type, self.service_name)
        if not client.wait_for_service(timeout_sec=self.availability_timeout):
            self.node.destroy_client(client)
            raise SimoxPlanningFailed("Service is unavailable")
        message = service_type.Request()
        configuration = request.configuration
        message.robot_model_path = str(configuration.robot_model_path)
        message.object_model_path = str(request.object_model_path)
        message.end_effector_name = configuration.end_effector_name
        message.kinematic_chain_name = configuration.kinematic_chain_name
        message.preshape_name = configuration.preshape_name
        message.num_grasps_to_plan = configuration.num_grasps_to_plan
        message.quality_threshold = configuration.quality_threshold
        message.timeout_ms = configuration.timeout_ms
        message.object_pose = PoseToRos2Converter.convert(request.object_pose)
        completion = Event()
        future = client.call_async(message)
        future.add_done_callback(lambda completed: completion.set())
        completed = completion.wait(
            configuration.timeout_ms / 1000 + self.response_grace_period
        )
        if not completed:
            future.cancel()
        self.node.destroy_client(client)
        if not completed:
            raise SimoxPlanningFailed("Service response timed out")
        response = future.result()
        if response is None or not response.success:
            raise SimoxPlanningFailed(
                "No response" if response is None else response.error_message
            )
        return SimoxResponse(
            [self.pose_from_message(pose) for pose in response.grasp_poses],
            list(response.qualities),
            list(response.are_force_closure),
        )

    @staticmethod
    def pose_from_message(message: RosPose) -> Pose:
        """Validate and normalize a metre-valued ROS pose without assigning a frame."""
        if not np.isfinite(
            [message.position.x, message.position.y, message.position.z]
        ).all():
            raise InvalidSimoxResponse("Non-finite position")
        quaternion = np.array(
            [
                message.orientation.x,
                message.orientation.y,
                message.orientation.z,
                message.orientation.w,
            ]
        )
        quaternion_norm = np.linalg.norm(quaternion)
        if not np.isfinite(quaternion_norm) or quaternion_norm == 0:
            raise InvalidSimoxResponse("Invalid quaternion")
        normalized_message = deepcopy(message)
        (
            normalized_message.orientation.x,
            normalized_message.orientation.y,
            normalized_message.orientation.z,
            normalized_message.orientation.w,
        ) = map(float, quaternion / quaternion_norm)
        return PoseToSemDTConverter.convert(normalized_message)


# %% object export and candidate conversion


@dataclass
class SimoxGraspPlanner:
    """Generate standard candidates while retaining Simox's ranking metadata."""

    configuration: SimoxConfiguration
    """Model, calibration and search parameters."""
    service: SimoxService
    """Transport used to request grasps."""

    def plan_grasps_for_body(
        self, graspable: HasGraspCandidates, arm: Arm, robot_frame: Body
    ) -> list[RankedGrasp]:
        """Plan for an annotation using an explicit robot base and gripper.

        The service must share the client's filesystem. Geometry is exported in
        metres, in the annotation root's frame. Returned candidates follow that root
        even when the object moves after the request.
        """
        world = graspable.root._world
        base_T_object = world.transform(graspable.root.global_pose, robot_frame)
        with TemporaryDirectory(prefix="coraplex-simox-") as directory:
            object_path = self.export_object(graspable, Path(directory))
            response = self.service.plan(
                SimoxRequest(self.configuration, object_path, base_T_object)
            )
        count = len(response.grasp_poses)
        if len(response.qualities) != count or len(response.are_force_closure) != count:
            raise InvalidSimoxResponse("Pose, quality and force-closure counts differ")
        if count == 0:
            raise SimoxPlanningFailed("No grasp candidates returned")
        if not np.isfinite(response.qualities).all():
            raise InvalidSimoxResponse("Non-finite quality")
        results = []
        gripper = arm.end_effector
        for pose, quality, force_closure in zip(
            response.grasp_poses, response.qualities, response.are_force_closure
        ):
            if not np.isfinite(pose.to_np()).all():
                raise InvalidSimoxResponse("Non-finite pose")
            object_T_tool = (
                base_T_object.homogeneous_matrix.inverse()
                @ pose.homogeneous_matrix
                @ self.configuration.simox_T_tool
            )
            object_T_grasp = HomogeneousTransformationMatrix.from_point_rotation_matrix(
                point=object_T_tool @ gripper.tool_P_grasp_when_closed,
                rotation_matrix=object_T_tool.rotation_matrix @ gripper.tool_R_grasp,
                reference_frame=graspable.root,
            )
            results.append(
                RankedGrasp(
                    GraspCandidate(graspable, object_T_grasp.pose),
                    float(quality),
                    bool(force_closure),
                )
            )
        return sorted(results, key=lambda result: result.quality, reverse=True)

    @staticmethod
    def export_object(graspable: HasGraspCandidates, directory: Path) -> Path:
        """Export the annotation's placed geometry in metres for the Simox STL loader."""
        mesh_path = directory / SimoxObjectFormat.MESH
        graspable.grasp_surface().export(mesh_path)
        root = Element(SimoxObjectFormat.ROOT, name=str(graspable.root.name))
        for tag in (SimoxObjectFormat.VISUALIZATION, SimoxObjectFormat.COLLISION):
            SubElement(
                SubElement(root, tag),
                SimoxObjectFormat.FILE,
                type=SimoxObjectFormat.STL,
            ).text = mesh_path.name
        object_path = directory / SimoxObjectFormat.DOCUMENT
        ElementTree(root).write(object_path, encoding="utf-8", xml_declaration=True)
        return object_path
