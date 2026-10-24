from __future__ import annotations

from krrood.ormatic.utils import classproperty
from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING

from robokudo.types.cv import ColorToDepthRatio
from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig
from semantic_digital_twin.datastructures.camera_rendering import ColorRenderingMode

if TYPE_CHECKING:
    from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
    from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


# %% World sources


@dataclass(slots=True)
class WorldDescriptorSource:
    """
    Load a standalone semantic world and camera from a world descriptor.
    """

    ros_package: str = "robokudo"
    """
    ROS package containing the world descriptor module.
    """

    descriptor_name: str = "world_semdt_raytracer_tabletop"
    """
    Module in ``descriptors/worlds`` that defines ``WorldDescriptor``.
    """

    camera_name: PrefixedName | None = None
    """
    Exact semantic camera name, or ``None`` for the descriptor's sole camera.
    """

    camera_pose: HomogeneousTransformationMatrix | None = None
    """
    Optional initial pose for the selected camera root.
    """


@dataclass(slots=True)
class RuntimeRobotWorldSource:
    """
    Select a robot camera from RoboKudo's runtime semantic world.

    .. todo::
        Replace the current global runtime-world lookup with the synchronized world
        reference once ``WorldSynchronizer`` provides that API.
    """

    robot_name: PrefixedName | None = None
    """
    Exact robot name, or ``None`` when the runtime world contains one robot.
    """


# %% Semantic Digital Twin ray-tracer configuration


@dataclass(slots=True)
class RayTracingCameraConfiguration(BaseCameraConfig):
    """
    Configuration for a simulated RGB-D camera backed by SemDT's RayTracer.
    """

    @classproperty
    def registry_name(cls) -> str:
        """
        Identifier selecting this camera configuration.
        """
        return "ray_tracing"

    """
    Name under which the camera configuration is registered.
    """

    interface_type: str = "RayTracingCameraInterface"
    """
    Camera interface selected by the collection-reader factory.
    """

    source: WorldDescriptorSource | RuntimeRobotWorldSource = field(
        default_factory=WorldDescriptorSource
    )
    """
    World source and camera-selection policy used for rendering.
    """

    color_to_depth_ratio: ColorToDepthRatio = field(default_factory=ColorToDepthRatio)
    """
    Scale factor from RGB image to depth image resolution.
    """

    rgb_mode: ColorRenderingMode = ColorRenderingMode.SEMANTIC
    """
    Method used to create the rendered color image.
    """
