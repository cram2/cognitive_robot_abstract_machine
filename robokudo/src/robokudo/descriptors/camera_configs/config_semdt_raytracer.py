from dataclasses import dataclass

from typing_extensions import ClassVar, Optional, Tuple

from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig


@dataclass(slots=True)
class SemDTRayTracerCameraConfig(BaseCameraConfig):
    """Configuration for a simulated RGB-D camera backed by SemDT's RayTracer."""

    registry_name: ClassVar[str] = "semdt_raytracer"

    interface_type: str = "SemDTRayTracer"

    world_descriptor_ros_package: str = "robokudo"
    """ROS package containing the world descriptor module."""

    world_descriptor_name: str = "world_semdt_raytracer_tabletop"
    """Module name in descriptors/worlds that defines class WorldDescriptor."""

    world_frame: Optional[str] = None
    """World frame to use as camera pose reference. If None, descriptor root is used."""

    camera_frame: str = "semdt_camera_optical_frame"
    """Name of the camera optical frame in the SemDT world."""

    camera_x: float = -1.20
    camera_y: float = 0.40
    camera_z: float = 1.05
    camera_roll: float = -2.2689280275926285
    camera_pitch: float = 0.0
    camera_yaw: float = -0.2707963267948965
    """Camera optical-frame pose (x right, y down, z forward) in world frame (xyz + rpy)."""

    resolution: int = 512
    """Square output resolution in pixels (width == height)."""

    fov_deg: float = 90.0
    """Symmetric horizontal/vertical field of view in degrees."""

    min_distance: float = 0.05
    max_distance: float = 8.0
    """RayTracer hit distance interval in meters."""

    color2depth_ratio: Tuple[float, float] = (1.0, 1.0)
    """Scale factor from RGB image to depth image resolution."""

    rgb_mode: str = "semantic"
    """RGB rendering mode: 'semantic' or 'trimesh' (falls back to semantic on failure)."""
