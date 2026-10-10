"""
simox_grasp_planner.py
======================
External interface to the Simox Grasp Planner ROS 2 service.

This module bridges the Simox grasp planning service and CoraPlex. It:
  - Auto-generates Simox ManipulationObject XML from a CoraPlex Body's STL mesh
  - Scales meter-scale STLs to millimetres for Simox
  - Calls the /plan_grasp ROS 2 service
  - Converts the geometry_msgs/Pose[] response to CoraPlex Pose objects
  - Wraps each pose as a GraspPose for downstream PickUpAction / SimoxPickUpAction

Interface pattern follows coraplex/external_interfaces/robokudo.py:
  - Module-level globals for is_init and service client
  - Decorator init_simox_interface for lazy ROS 2 initialisation
  - Works with ROS_VERSION=2

Usage:
    from coraplex.external_interfaces.simox_grasp_planner import (
        plan_grasps_for_body,
        DEFAULT_ROBOT_XML,
        DEFAULT_OBJECT_DIR,
    )
    grasp_poses = plan_grasps_for_body(
        body=world.get_body_by_name('breakfast_cereal.stl'),
        arm=Arms.RIGHT,
        end_effector_name='r_gripper',
        kinematic_chain_name='RightArm',
        world=world,
    )
"""

from __future__ import annotations

import logging
import os
import struct
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from geometry_msgs.msg import Pose as RosPose
from geometry_msgs.msg import Quaternion

from dataclasses import dataclass
import numpy as np

from coraplex.datastructures.enums import Arms, SimoxApproachDirection
from coraplex.ros import ServiceProxy
from semantic_digital_twin.spatial_types import Point3, Quaternion as SpatialQuaternion
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)


class GraspPose(Pose):
    """
    A pose from which a grasp can be performed along with Simox metadata.
    """

    def __init__(
        self,
        position: Optional[Point3] = None,
        orientation: Optional[SpatialQuaternion] = None,
        reference_frame: Optional[KinematicStructureEntity] = None,
        arm: Optional[Any] = None,
        approach: str = "",
        quality: float = 0.0,
    ) -> None:
        super().__init__(
            position=position,
            orientation=orientation,
            reference_frame=reference_frame,
        )
        self.arm = arm
        self.approach = approach
        self.quality = quality

    @classmethod
    def from_pose(
        cls,
        pose: Pose,
        arm: Optional[Any] = None,
        grasp_description: Optional[Any] = None,
        approach: str = "",
        quality: float = 0.0,
        **kwargs: Any,
    ) -> GraspPose:
        return cls(
            position=pose.position,
            orientation=pose.orientation,
            reference_frame=pose.reference_frame,
            arm=arm,
            approach=approach,
            quality=quality,
        )


def translate_pose_along_local_axis(
    pose: Pose, axis: Union[List, np.ndarray], distance: float
) -> Pose:
    """
    Translate a pose along a given 3d vector (axis) by a given distance.
    """
    axis_arr = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis_arr)
    if norm < 1e-9:
        return pose
    normalized_translation_vector = axis_arr / norm

    rot_matrix = pose.rotation_matrix.to_np()[:3, :3]
    translation_in_world = rot_matrix @ normalized_translation_vector
    scaled_translation_vector = (
        np.array(pose.position.to_list()[:3]) + translation_in_world * distance
    )

    return Pose(
        Point3.from_iterable(scaled_translation_vector),
        pose.quaternion,
        reference_frame=pose.reference_frame,
    )


if TYPE_CHECKING:
    from semantic_digital_twin.world_description.world import World

logger = logging.getLogger(__name__)

# %% Module-level state (ROS-style lazy init)

_is_init: bool = False
_service_proxy: Optional[ServiceProxy] = None
_init_lock: Lock = Lock()

# %% Constants

_SIMOX_ENV_PATH = os.environ.get("SIMOX_WORKSPACE_DIR")
if _SIMOX_ENV_PATH:
    _BASE_PATH = Path(_SIMOX_ENV_PATH)
else:
    _candidate = Path(__file__).resolve().parents[5]
    if (_candidate / "grasp_test_files").exists():
        _BASE_PATH = _candidate
    else:
        _BASE_PATH = Path.cwd()

# Absolute path to the Simox robot XML wrapper (pr2_for_simox.urdf + EEF definitions)
DEFAULT_ROBOT_XML: str = str(
    _BASE_PATH / "grasp_test_files" / "resources" / "robots" / "pr2.xml"
)

# Directory where scaled mm STLs and ManipulationObject XMLs are cached
DEFAULT_OBJECT_DIR: Path = _BASE_PATH / "grasp_test_files" / "resources" / "objects"

SERVICE_TOPIC: str = "/plan_grasp"
SERVICE_TIMEOUT_SECONDS: float = 60.0


# %% Orientation & Frame Transformation helpers


def _classify_approach(ros_orientation: Quaternion) -> SimoxApproachDirection:
    """
    Classify the grasp approach direction in the robot base frame based on
    Simox's tool-frame forward vector.

    In Simox PR2 tool frame (r_gripper_tool_frame, with pitch = +90° in URDF),
    the local +Z axis (column 2 of R_simox) points along the forward finger
    approach direction into the object.

    In the robot base frame (X forward, Y left, Z up):
    - fwd[2] > 0.5: pointing up from underneath -> BOTTOM grasp -> skip!
    - fwd[2] < -0.4: pointing down from above -> TOP grasp
    - Horizontal approaches:
        - If |fwd[0]| >= |fwd[1]|:
            - fwd[0] >= 0: gripper approaches from FRONT, pointing forward (+X)
            - fwd[0] < 0:  gripper approaches from BACK, pointing backward (-X)
        - Otherwise:
            - fwd[1] >= 0: gripper is on the RIGHT, pointing leftwards (+Y) into object -> RIGHT
            - fwd[1] < 0:  gripper is on the LEFT, pointing rightwards (-Y) into object -> LEFT

    :param ros_orientation: geometry_msgs/Quaternion from Simox.
    :return: SimoxApproachDirection enum member.
    """
    from coraplex.tf_transformations import quaternion_matrix

    q = [
        float(ros_orientation.x),
        float(ros_orientation.y),
        float(ros_orientation.z),
        float(ros_orientation.w),
    ]
    R_simox = quaternion_matrix(q)[:3, :3]
    fwd = R_simox[:, 2]

    # 1. Vertical component
    if fwd[2] > 0.5:
        logger.debug("Skipping BOTTOM approach (fwd_z=%.3f)", fwd[2])
        return SimoxApproachDirection.SKIPPED
    elif fwd[2] < -0.4:
        logger.debug("Detected TOP approach (fwd_z=%.3f)", fwd[2])
        return SimoxApproachDirection.TOP

    # 2. Horizontal component
    if abs(fwd[0]) >= abs(fwd[1]):
        if fwd[0] >= 0:
            logger.debug("Detected FRONT approach (fwd_x=%.3f)", fwd[0])
            return SimoxApproachDirection.FRONT
        else:
            logger.debug("Detected BACK approach (fwd_x=%.3f)", fwd[0])
            return SimoxApproachDirection.BACK
    else:
        if fwd[1] >= 0:
            logger.debug("Detected RIGHT approach (fwd_y=%.3f)", fwd[1])
            return SimoxApproachDirection.RIGHT
        else:
            logger.debug("Detected LEFT approach (fwd_y=%.3f)", fwd[1])
            return SimoxApproachDirection.LEFT


def _simox_pose_to_coraplex_tool_pose(
    ros_pose: RosPose, reference_frame: Optional[KinematicStructureEntity] = None
) -> Pose:
    """
    Transform a Simox TCP pose to CoraPlex's r_gripper_tool_frame.

    Frame Definitions:
    - Simox (pr2_for_simox.urdf):
        r_gripper_tool_joint has origin xyz="0.13 0 0" rpy="0 1.5707963 0"
        - Local +Z is the forward approach axis into the object
        - Local +Y is the finger opening axis
        - Local +X is pointing down

    - CoraPlex (standard PR2 model, pr2.py):
        r_gripper_tool_frame has origin xyz="0.18 0 0" rpy="0 0 0"
        - Local +X is the forward approach axis into the object
        - Local +Y is the finger opening axis
        - Local +Z is orthogonal up

    The relative transformation from Simox TCP to CoraPlex tool frame is:
        T_simox_to_coraplex = inv(T_palm_to_simox) @ T_palm_to_coraplex
            R = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]  (pitch = -90° around Y)
            P = [0, 0, 0.05]                          (+5 cm along Simox +Z forward axis)

    This ensures:
    1. CoraPlex +X aligns with Simox forward approach vector.
    2. CoraPlex +Y aligns with Simox finger opening axis (so finger roll matches object geometry).
    3. The 5 cm offset between 0.13 m and 0.18 m is accounted for, eliminating mesh penetration.

    :param ros_pose: geometry_msgs/Pose from Simox (in robot frame).
    :param reference_frame: CoraPlex reference entity (robot base frame).
    :return: CoraPlex Pose for r_gripper_tool_frame (in robot frame).
    """
    from coraplex.tf_transformations import quaternion_matrix, quaternion_from_matrix

    q_simox = [
        float(ros_pose.orientation.x),
        float(ros_pose.orientation.y),
        float(ros_pose.orientation.z),
        float(ros_pose.orientation.w),
    ]
    R_simox = quaternion_matrix(q_simox)[:3, :3]
    P_simox = np.array(
        [
            float(ros_pose.position.x),
            float(ros_pose.position.y),
            float(ros_pose.position.z),
        ]
    )

    R_simox_to_coraplex = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    R_coraplex = R_simox @ R_simox_to_coraplex
    P_coraplex = P_simox + 0.05 * R_simox[:, 2]

    T_coraplex = np.eye(4)
    T_coraplex[:3, :3] = R_coraplex
    q_coraplex = quaternion_from_matrix(T_coraplex)

    return Pose.from_xyz_quaternion(
        pos_x=float(P_coraplex[0]),
        pos_y=float(P_coraplex[1]),
        pos_z=float(P_coraplex[2]),
        quat_x=float(q_coraplex[0]),
        quat_y=float(q_coraplex[1]),
        quat_z=float(q_coraplex[2]),
        quat_w=float(q_coraplex[3]),
        reference_frame=reference_frame,
    )


# %% STL scale helpers


def _detect_stl_unit(path: Path) -> str:
    """
    Heuristic: if max absolute vertex coordinate < 5, assume meters; else mm.
    PR2-graspable objects are 5-30 cm = 0.05-0.30 m or 50-300 mm.

    :param path: Path to a binary STL file.
    :return: 'meters' or 'millimeters'
    """
    max_coord = 0.0
    with open(path, "rb") as f:
        f.read(80)
        count = struct.unpack("<I", f.read(4))[0]
        sample = min(count, 200)
        for _ in range(sample):
            f.read(12)  # normal
            for _ in range(3):
                x, y, z = struct.unpack("<3f", f.read(12))
                max_coord = max(max_coord, abs(x), abs(y), abs(z))
            f.read(2)  # attr
    return "meters" if max_coord < 5.0 else "millimeters"


def _scale_stl(source: Path, dest: Path, scale: float = 1000.0) -> None:
    """
    Read a binary STL, multiply all vertex coordinates by scale, and write a new binary
    STL.

    :param source: Input binary STL path.
    :param dest: Output binary STL path.
    :param scale: Multiplier for vertex coordinates.
    """
    with open(source, "rb") as f:
        header = f.read(80)
        count = struct.unpack("<I", f.read(4))[0]
        triangles = []
        for _ in range(count):
            normal = struct.unpack("<3f", f.read(12))
            v1 = struct.unpack("<3f", f.read(12))
            v2 = struct.unpack("<3f", f.read(12))
            v3 = struct.unpack("<3f", f.read(12))
            attr = struct.unpack("<H", f.read(2))[0]
            triangles.append((normal, v1, v2, v3, attr))

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(header[:80].ljust(80, b"\x00"))
        f.write(struct.pack("<I", len(triangles)))
        for normal, v1, v2, v3, attr in triangles:
            f.write(struct.pack("<3f", *normal))
            f.write(struct.pack("<3f", v1[0] * scale, v1[1] * scale, v1[2] * scale))
            f.write(struct.pack("<3f", v2[0] * scale, v2[1] * scale, v2[2] * scale))
            f.write(struct.pack("<3f", v3[0] * scale, v3[1] * scale, v3[2] * scale))
            f.write(struct.pack("<H", attr))


def _scale_stl_to_mm(source: Path, dest: Path) -> None:
    """
    Convenience wrapper to scale meters to millimetres.
    """
    _scale_stl(source, dest, scale=1000.0)


# %% Object XML helpers


def _get_object_stl_path(body: Body) -> Optional[Path]:
    """
    Extract the STL collision mesh path from a CoraPlex Body.

    Tries body.description.urdf_object.collision_mesh_path first, then falls back to
    searching resources/objects/ by body name.

    :param body: A CoraPlex Body instance.
    :return: Path to the STL file, or None if not found.
    """
    # Method 1: body may expose its mesh path directly
    direct = getattr(body, "mesh_path", None)
    if direct and Path(str(direct)).suffix.lower() == ".stl":
        return Path(str(direct))

    for attr in ("urdf_object", "description"):
        obj = getattr(body, attr, None)
        if obj is None:
            continue
        for sub_attr in ("collision_mesh_path", "visual_mesh_path", "mesh_path"):
            candidate = getattr(obj, sub_attr, None)
            if candidate and Path(candidate).suffix.lower() == ".stl":
                return Path(candidate)

    # Method 2: search resources/objects by body name
    resources_dir = Path(__file__).resolve().parents[2] / "resources" / "objects"
    candidates = [
        resources_dir / f"{str(body.name)}.stl",
        resources_dir / str(body.name),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    logger.warning("Could not find STL mesh for body '%s'", body.name)
    return None


def _ensure_simox_object_xml(
    body: Body,
    end_effector_name: str,
    object_dir: Path,
    robot_type: str = "PR2",
) -> Optional[Path]:
    """
    Ensure a Simox ManipulationObject XML exists for this body in object_dir.

    Creates it (and a scaled mm STL) if not already present.

    :param body: CoraPlex Body to generate XML for.
    :param end_effector_name: EEF name for the GraspSet element.
    :param object_dir: Directory where XMLs and scaled STLs are cached.
    :param robot_type: Simox robot type string (default: 'PR2').
    :return: Path to the XML file, or None on failure.
    """
    clean_name = Path(str(body.name)).stem
    xml_path = object_dir / f"{clean_name}.xml"
    if xml_path.exists():
        logger.debug("Using cached Simox XML: %s", xml_path)
        return xml_path

    source_stl = _get_object_stl_path(body)
    if source_stl is None:
        return None

    # Simox's mesh loader (CoinVisualizationNode) assumes mesh coordinates
    # are in METERS and multiplies by 1000 internally to convert to mm.
    # CoraPlex STLs are already in meters, so we use the STL directly.
    unit = _detect_stl_unit(source_stl)
    target_stl_name = f"{clean_name}.stl"
    target_stl_path = object_dir / target_stl_name

    if not target_stl_path.exists():
        if unit == "millimeters":
            logger.info(
                "STL '%s' is in mm (>5.0). Scaling to meters for Simox → %s",
                source_stl.name,
                target_stl_path,
            )
            _scale_stl(source_stl, target_stl_path, scale=0.001)
        else:
            import shutil

            shutil.copy2(source_stl, target_stl_path)

    # Write ManipulationObject XML
    xml_content = (
        f'<?xml version="1.0" encoding="UTF-8" ?>\n'
        f"<!-- Auto-generated by simox_grasp_planner.py for body: {body.name} -->\n"
        f'<ManipulationObject name="{body.name}">\n'
        f"    <Visualization>\n"
        f'        <File type="stl">{target_stl_name}</File>\n'
        f"    </Visualization>\n"
        f"    <CollisionModel>\n"
        f'        <File type="stl">{target_stl_name}</File>\n'
        f"    </CollisionModel>\n"
        f'    <GraspSet name="Simox_{end_effector_name}" RobotType="{robot_type}" '
        f'EndEffector="{end_effector_name}"/>\n'
        f"</ManipulationObject>\n"
    )
    object_dir.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(xml_content, encoding="utf-8")
    logger.info("Written Simox ManipulationObject XML: %s", xml_path)
    return xml_path


# %% ROS 2 client


class _SimoxClient:
    def __init__(self):
        import rclpy

        if not rclpy.ok():
            rclpy.init()
        from grasp_planner_msgs.srv import PlanGrasp

        self.node = rclpy.create_node("simox_grasp_client")
        self.client = self.node.create_client(PlanGrasp, SERVICE_TOPIC)

    def is_available(self, timeout_sec: float = 3.0) -> bool:
        return self.client.wait_for_service(timeout_sec=timeout_sec)

    def call(self, request, timeout_sec: float = 60.0):
        import rclpy

        if not self.is_available():
            logger.error("Simox service '%s' is not available.", SERVICE_TOPIC)
            return None
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout_sec)
        if future.done():
            return future.result()
        logger.error(
            "Simox service '%s' timed out after %.1fs", SERVICE_TOPIC, timeout_sec
        )
        return None


_client: Optional[_SimoxClient] = None
_init_lock: Lock = Lock()


def get_simox_client() -> Optional[_SimoxClient]:
    global _client
    with _init_lock:
        if _client is None:
            try:
                import rclpy  # noqa: F401
                from grasp_planner_msgs.srv import PlanGrasp  # noqa: F401
            except ImportError:
                logger.error(
                    "grasp_planner_msgs not importable. "
                    "Did you source the install/setup.bash?"
                )
                return None
            _client = _SimoxClient()
        return _client


# %% Pose conversion


def _ros_pose_to_coraplex_pose(
    ros_pose: RosPose, reference_frame: Optional[KinematicStructureEntity] = None
) -> Pose:
    """
    Convert a geometry_msgs/Pose to a CoraPlex Pose (world frame, meters).

    :param ros_pose: ROS 2 geometry_msgs/Pose in meters.
    :param reference_frame: Optional reference frame.
    :return: Equivalent CoraPlex Pose.
    """
    return Pose.from_xyz_quaternion(
        pos_x=float(ros_pose.position.x),
        pos_y=float(ros_pose.position.y),
        pos_z=float(ros_pose.position.z),
        quat_x=float(ros_pose.orientation.x),
        quat_y=float(ros_pose.orientation.y),
        quat_z=float(ros_pose.orientation.z),
        quat_w=float(ros_pose.orientation.w),
        reference_frame=reference_frame,
    )


# %% Public API


def plan_grasps_for_body(
    body: Body,
    arm: Arms,
    end_effector_name: str,
    kinematic_chain_name: str,
    world: World,
    robot_xml: Optional[str] = None,
    object_dir: Optional[Path] = None,
    preshape_name: str = "Power Preshape",
    num_grasps_to_plan: int = 50,
    quality_threshold: float = 0.001,
    timeout_ms: int = 30000,
) -> Dict[Union[SimoxApproachDirection, str], List[GraspPose]]:
    """
    Call the Simox grasp planner service for the given body and return a dictionary of
    GraspPose lists grouped by approach direction.

    Structure::

        {
            'top':   [best_top_pose, second_top_pose, ...],   # best quality first
            'right': [best_right_pose, ...],
            'front': [...],
            'back':  [...],
            'left':  [...],
        }

    Only directions for which Simox found at least one valid grasp are present
    as keys. Bottom grasps are always filtered out (table blocks from below).

    :param body: The CoraPlex Body object to grasp.
    :param arm: Which arm (Arms.RIGHT or Arms.LEFT).
    :param end_effector_name: Simox EEF name, e.g. 'r_gripper' or 'l_gripper'.
    :param kinematic_chain_name: Simox RobotNodeSet, e.g. 'RightArm' or 'LeftArm'.
    :param world: The active CoraPlex World (needed for body pose and frame resolution).
    :param robot_xml: Absolute path to the Simox robot XML wrapper (pr2.xml).
    :param object_dir: Directory for cached mm STLs and ManipulationObject XMLs.
    :param preshape_name: Gripper preshape to use, e.g. 'Open' or 'Power Preshape'.
    :param num_grasps_to_plan: How many grasp candidates to request.
    :param quality_threshold: Minimum grasp quality (0.0–1.0).
    :param timeout_ms: Planning timeout in milliseconds.
    :return: Dict mapping approach direction → list of GraspPose (best-quality-first).
             Returns {} if the service fails or returns no valid grasps.
    """
    if robot_xml is None:
        robot_xml = DEFAULT_ROBOT_XML
    if object_dir is None:
        object_dir = DEFAULT_OBJECT_DIR

    client = get_simox_client()
    if client is None or not client.is_available():
        logger.error(
            "Simox grasp_planner_service node not available on %s", SERVICE_TOPIC
        )
        return {}

    # 1. Ensure ManipulationObject XML exists
    xml_path = _ensure_simox_object_xml(body, end_effector_name, object_dir)
    if xml_path is None:
        logger.error("Could not create Simox XML for body '%s'", body.name)
        return {}

    # 2. Get object pose relative to robot base (since Simox robot is loaded at origin)
    robot_frame = None
    for candidate in world.bodies:
        c_str = str(candidate.name).lower()
        if "base_footprint" in c_str or "base_link" in c_str:
            robot_frame = candidate
            break

    if robot_frame is not None:
        object_pose_in_robot = world.transform(body.global_pose, robot_frame)
    else:
        object_pose_in_robot = body.global_pose
        robot_frame = world.root

    ros_object_pose = RosPose()
    _fill_ros_pose(ros_object_pose, object_pose_in_robot)

    # 3. Build service request
    try:
        from grasp_planner_msgs.srv import PlanGrasp
    except ImportError:
        logger.error(
            "grasp_planner_msgs not importable. Did you source install/setup.bash?"
        )
        return {}
    request = PlanGrasp.Request()
    request.robot_model_path = robot_xml
    request.object_model_path = str(xml_path)
    request.object_pose = ros_object_pose
    request.end_effector_name = end_effector_name
    request.preshape_name = preshape_name
    request.kinematic_chain_name = kinematic_chain_name
    request.num_grasps_to_plan = num_grasps_to_plan
    request.quality_threshold = quality_threshold
    request.timeout_ms = timeout_ms

    logger.info(
        "Calling Simox /plan_grasp for '%s' with %s (object in robot frame: x=%.3f y=%.3f z=%.3f)",
        str(body.name),
        end_effector_name,
        ros_object_pose.position.x,
        ros_object_pose.position.y,
        ros_object_pose.position.z,
    )

    # 4. Call service
    response = client.call(request, timeout_sec=(timeout_ms / 1000.0) + 5.0)

    if response is None:
        logger.error("Simox /plan_grasp returned None (service timeout?)")
        return {}

    if not response.success:
        logger.error("Simox /plan_grasp failed: %s", response.error_message)
        return {}

    logger.info(
        "Simox returned %d grasp poses for '%s'",
        len(response.grasp_poses),
        str(body.name),
    )

    # 5. Convert geometry_msgs/Pose[] → GraspPose grouped by approach direction.
    # We transform each Simox TCP pose into CoraPlex's r_gripper_tool_frame
    # using T_simox_to_coraplex. This preserves the exact physical finger roll
    # and collision-free clearance that Simox verified against the object mesh,
    # while classifying approach directions relative to the robot's base frame.

    # Read quality scores — Simox sends a parallel float[] alongside grasp_poses[].
    # Fall back to zeros if the field is absent (older service versions).
    qualities = list(response.qualities) if hasattr(response, "qualities") else []
    if len(qualities) != len(response.grasp_poses):
        qualities = [0.0] * len(response.grasp_poses)

    counts: Dict[SimoxApproachDirection, int] = {
        SimoxApproachDirection.FRONT: 0,
        SimoxApproachDirection.BACK: 0,
        SimoxApproachDirection.LEFT: 0,
        SimoxApproachDirection.RIGHT: 0,
        SimoxApproachDirection.TOP: 0,
        SimoxApproachDirection.SKIPPED: 0,
    }

    # Accumulate into groups: direction → list of (quality, GraspPose)
    grouped: Dict[Union[SimoxApproachDirection, str], List[tuple]] = {}

    for ros_pose, quality in zip(response.grasp_poses, qualities):
        # Classify approach direction in robot frame (bottom grasps skipped)
        label = _classify_approach(ros_pose.orientation)
        if label == SimoxApproachDirection.SKIPPED:
            counts[SimoxApproachDirection.SKIPPED] += 1
            continue
        counts[label] += 1

        # Transform Simox TCP pose (rpy=[0, pi/2, 0], xyz=[0.13, 0, 0])
        # to CoraPlex tool frame (rpy=[0, 0, 0], xyz=[0.18, 0, 0]) in robot frame.
        local_coraplex_pose = _simox_pose_to_coraplex_tool_pose(
            ros_pose, reference_frame=robot_frame
        )

        # Convert TCP pose from robot frame to world frame
        if robot_frame != world.root:
            world_coraplex_pose = world.transform(local_coraplex_pose, world.root)
        else:
            world_coraplex_pose = local_coraplex_pose

        grasp_pose = GraspPose.from_pose(
            world_coraplex_pose,
            arm,
            grasp_description=None,
            approach=str(label),
            quality=float(quality),
        )

        if label not in grouped:
            grouped[label] = []
        grouped[label].append((float(quality), grasp_pose))

    # Sort each direction group by quality descending and strip the sort key
    grasp_dict: Dict[Union[SimoxApproachDirection, str], List[GraspPose]] = {
        direction: [
            gp for _, gp in sorted(candidates, key=lambda t: t[0], reverse=True)
        ]
        for direction, candidates in grouped.items()
    }

    total = sum(len(v) for v in grasp_dict.values())
    logger.info(
        "Pose classification: front=%d back=%d left=%d right=%d top=%d skipped(bottom)=%d → %d total",
        counts[SimoxApproachDirection.FRONT],
        counts[SimoxApproachDirection.BACK],
        counts[SimoxApproachDirection.LEFT],
        counts[SimoxApproachDirection.RIGHT],
        counts[SimoxApproachDirection.TOP],
        counts[SimoxApproachDirection.SKIPPED],
        total,
    )
    for direction, poses in grasp_dict.items():
        logger.info(
            "  [%s] %d candidate(s) — best quality=%.4f",
            direction,
            len(poses),
            poses[0].quality if poses else 0.0,
        )

    return grasp_dict


def _fill_ros_pose(ros_pose: RosPose, coraplex_pose: Pose) -> None:
    """
    Fill a geometry_msgs/Pose from a CoraPlex Pose (in-place).

    Assumes CoraPlex Pose is already in world frame, meters.

    :param ros_pose: ROS Pose message to fill.
    :param coraplex_pose: CoraPlex Pose source.
    """
    ros_pose.position.x = float(coraplex_pose.position.x)
    ros_pose.position.y = float(coraplex_pose.position.y)
    ros_pose.position.z = float(coraplex_pose.position.z)
    ros_pose.orientation.x = float(coraplex_pose.orientation.x)
    ros_pose.orientation.y = float(coraplex_pose.orientation.y)
    ros_pose.orientation.z = float(coraplex_pose.orientation.z)
    ros_pose.orientation.w = float(coraplex_pose.orientation.w)
