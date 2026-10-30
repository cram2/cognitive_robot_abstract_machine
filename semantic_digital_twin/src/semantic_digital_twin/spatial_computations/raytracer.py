from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from typing_extensions import TYPE_CHECKING

import numpy as np
import trimesh
from trimesh import Scene
from krrood.ormatic.utils import classproperty
from semantic_digital_twin.spatial_computations.exceptions import (
    RayPointShapeMismatch,
    UnsupportedCameraProjection,
)

from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.camera_model import (
    PinholeCameraModel,
    NoCameraDistortion,
)
from semantic_digital_twin.datastructures.camera_rendering import (
    CameraRays,
    RayIntersections,
    DepthMode,
    OpticalAxisDepthMode,
    RayDistanceDepthMode,
    RenderedRGBDFrame,
    SegmentationDepthFrame,
    BodyColorRenderingMode,
)
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.spatial_types.spatial_types import (
    GenericSpatialType,
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from semantic_digital_twin.datastructures.camera_rendering import ColorRenderingMode

    from semantic_digital_twin.world import World
    from semantic_digital_twin.robots.robot_parts import Camera


# %% World ray tracing
@dataclass(eq=False)
class RayTracer:
    """
    Trace world geometry and render camera segmentation, depth, and color images.
    """

    world: World
    """
    The world to use for ray tracing.
    """

    _last_world_model: int = field(default=-1, init=False)
    """
    Last model version of the world to which the ray tracer was updated.
    """

    _last_world_state: int = field(default=-1, init=False)
    """
    Last state version of the world to which the ray tracer was updated.
    """

    index_to_body: dict[int, Body] = field(default_factory=dict, init=False)
    """
    Maps the index of a body to the body itself.
    """

    scene_to_index: dict[str, int] = field(default_factory=dict, init=False)
    """
    Maps the index in the trimesh scene to the index of the body in the world.
    """

    scene: Scene = field(default_factory=Scene, init=False)
    """
    The trimesh scene used for ray tracing which mirrors the world.
    """

    def __post_init__(self) -> None:
        """
        Populate the scene from the current world geometry and state.
        """
        self.update_scene()

    def update_scene(self) -> None:
        """
        Keep collision geometry and poses consistent with the current world.
        """
        if self._last_world_model != self.world.get_world_model_manager().version:
            self.add_missing_bodies()
            self._last_world_model = self.world.get_world_model_manager().version
        if self._last_world_state != self.world.state.version:
            self.update_transforms()
            self._last_world_state = self.world.state.version

    def add_missing_bodies(self) -> None:
        """
        Add collision geometry for bodies not yet represented in the scene.
        """
        bodies_to_add = [
            body
            for body in self.world.bodies
            if body.name.name not in "\t".join(self.scene.graph.nodes)
        ]
        # Each collision geometry uses the body name with a geometry-index suffix.
        for body in bodies_to_add:
            world_T_body = self.world.compute_forward_kinematics_np(
                self.world.root, body
            )
            for geometry_index, collision in enumerate(body.collision):
                node_name = self.collision_node_name(body, geometry_index)
                self.scene.add_geometry(
                    collision.mesh,
                    node_name=node_name,
                    parent_node_name="world",
                    transform=world_T_body @ collision.origin.to_np(),
                )
                self.scene_to_index[node_name] = body.index
                self.index_to_body[body.index] = body

    def update_transforms(self) -> None:
        """
        Update scene geometry to the current world-body poses.
        """
        for body in self.world.bodies:
            world_T_body = self.world.compute_forward_kinematics_np(
                self.world.root, body
            )
            for geometry_index, collision in enumerate(body.collision):
                world_T_collision = world_T_body @ collision.origin.to_np()
                self.scene.graph[self.collision_node_name(body, geometry_index)] = (
                    world_T_collision
                )

    @staticmethod
    def collision_node_name(body: Body, geometry_index: int) -> str:
        """
        Return the scene node identifying a body's collision geometry.

        :param body: Body owning the collision geometry.
        :param geometry_index: Position of the geometry in the body's collision list.
        :return: Scene node name identifying the collision geometry.
        """
        return body.name.name + f"_collision_{geometry_index}"

    # %% Camera depth and segmentation
    def create_segmentation_mask(
        self,
        camera_pose: GenericSpatialType,
        resolution: CameraResolution | None = None,
        min_distance: float = 0,
        max_distance: float = np.inf,
        field_of_view: FieldOfView | None = None,
    ) -> np.ndarray:
        """
        Return world-body indices per pixel, or negative values for missed rays.

        Image dimensions follow the ray tracer's width-first pixel convention.

        :param camera_pose: Camera pose in world coordinates, with its viewing direction
            along x.
        :param resolution: Pixel dimensions of the rendered image.
        :param min_distance: Inclusive minimum ray-hit distance in meters.
        :param max_distance: Inclusive maximum ray-hit distance in meters.
        :param field_of_view: Horizontal and vertical viewing angles.
        :return: Width-first body indices, with negative values for missed rays.
        """
        return self.create_segmentation_and_depth(
            camera_pose,
            resolution=resolution,
            min_distance=min_distance,
            max_distance=max_distance,
            field_of_view=field_of_view,
        ).segmentation

    def create_depth_map(
        self,
        camera_pose: GenericSpatialType,
        resolution: CameraResolution | None = None,
        min_distance: float = 0,
        max_distance: float = np.inf,
        field_of_view: FieldOfView | None = None,
        depth_mode: DepthMode = RayDistanceDepthMode(),
    ) -> np.ndarray:
        """
        Return meter depths per pixel, or negative values for missed rays.

        Depth defaults to distance along the viewing ray. Optical-axis depth is
        available through :class:`~semantic_digital_twin.datastructures.camera_rendering.DepthMode`.
        Image dimensions follow the ray tracer's width-first pixel convention.

        :param camera_pose: Camera pose in world coordinates, with its viewing direction along x.
        :param resolution: Pixel dimensions of the rendered image.
        :param min_distance: Inclusive minimum ray-hit distance in meters.
        :param max_distance: Inclusive maximum ray-hit distance in meters.
        :param field_of_view: Horizontal and vertical viewing angles.
        :param depth_mode: Distance convention represented by the depth values.
        :return: Width-first meter depths, with negative values for missed rays.
        """
        return self.create_segmentation_and_depth(
            camera_pose,
            resolution=resolution,
            min_distance=min_distance,
            max_distance=max_distance,
            field_of_view=field_of_view,
            depth_mode=depth_mode,
        ).depth

    def create_segmentation_and_depth(
        self,
        camera_pose: GenericSpatialType,
        resolution: CameraResolution | None = None,
        min_distance: float = 0,
        max_distance: float = np.inf,
        field_of_view: FieldOfView | None = None,
        depth_mode: DepthMode = RayDistanceDepthMode(),
        camera_model: PinholeCameraModel | None = None,
    ) -> SegmentationDepthFrame:
        """
        Return aligned body indices and meter depths from one intersection query.

        Missed rays have negative segmentation and depth values. Distance limits apply
        along the viewing ray for both depth conventions. Arrays use the width-first
        pixel convention of :meth:`create_camera_rays`.

        :param camera_pose: Camera pose in world coordinates, with its viewing direction
            along x.
        :param resolution: Pixel dimensions of the rendered image.
        :param min_distance: Inclusive minimum ray-hit distance in meters.
        :param max_distance: Inclusive maximum ray-hit distance in meters.
        :param field_of_view: Horizontal and vertical viewing angles.
        :param camera_model: Full calibration, overriding resolution and angular extent.
        :param depth_mode: Distance convention represented by the depth values.
        :return: Aligned body indices and meter depths from the same ray query.
        """
        resolution = (
            camera_model.resolution
            if camera_model is not None
            else (resolution or CameraResolution())
        )
        segmentation = np.full(
            (resolution.width, resolution.height), -1, dtype=np.int32
        )
        depth = np.full(segmentation.shape, -1.0, dtype=np.float32)
        camera_rays = self.create_camera_rays(
            camera_pose,
            resolution=resolution,
            field_of_view=field_of_view,
            camera_model=camera_model,
        )
        ray_origins = camera_rays.origin_points
        ray_directions = camera_rays.direction_vectors
        pixels = camera_rays.pixel_coordinates
        ray_intersections = self.ray_test(
            ray_origins,
            ray_origins + ray_directions,
            multiple_hits=True,
            min_distance=min_distance,
            max_distance=max_distance,
        )
        points = ray_intersections.hit_points
        ray_indices = ray_intersections.ray_indices
        bodies = ray_intersections.bodies
        if len(ray_indices) == 0:
            return SegmentationDepthFrame(segmentation=segmentation, depth=depth)

        selected_hits = np.unique(ray_indices, return_index=True)[1]
        ray_indices = ray_indices[selected_hits]
        points = points[selected_hits]
        body_indices = np.asarray([body.index for body in bodies], dtype=np.int32)[
            selected_hits
        ]
        hit_pixels = pixels[ray_indices]
        segmentation[hit_pixels[:, 0], hit_pixels[:, 1]] = body_indices

        hit_depths = depth_mode.hit_depth(
            hit_points=points,
            ray_origins=ray_origins[ray_indices],
            ray_directions=ray_directions[ray_indices],
            world_T_camera=HomogeneousTransformationMatrix(self.scene.camera_transform),
        )
        valid_depths = hit_depths >= 0.0
        hit_pixels = hit_pixels[valid_depths]
        hit_depths = hit_depths[valid_depths]
        depth[hit_pixels[:, 0], hit_pixels[:, 1]] = hit_depths.astype(np.float32)
        return SegmentationDepthFrame(segmentation=segmentation, depth=depth)

    def create_camera_rays(
        self,
        camera_pose: GenericSpatialType,
        resolution: CameraResolution | None = None,
        field_of_view: FieldOfView | None = None,
        camera_model: PinholeCameraModel | None = None,
    ) -> CameraRays:
        """
        Return world-space rays and width-first, top-left-origin pixels for an x-forward
        camera pose.

        :param camera_pose: Camera pose in world coordinates, with its viewing direction
            along x.
        :param resolution: Pixel dimensions of the rendered image.
        :param field_of_view: Horizontal and vertical viewing angles.
        :param camera_model: Full calibration, overriding resolution and angular extent.
        :raises UnsupportedCameraProjection: If lens distortion is requested.
        :return: World-space ray origins and directions with corresponding width-first
            pixels.
        """
        if camera_model is None:
            camera_model = PinholeCameraModel.from_field_of_view(
                resolution=resolution or CameraResolution(),
                field_of_view=field_of_view or FieldOfView(),
            )
        if not isinstance(camera_model.distortion, NoCameraDistortion):
            raise UnsupportedCameraProjection(
                reason="distorted rendering is not supported"
            )
        resolution = camera_model.resolution
        world_T_forward_view = camera_pose.to_np()
        self.update_scene()
        self.scene.camera.resolution = (resolution.width, resolution.height)
        self.scene.camera.focal = (
            camera_model.focal_length_x,
            camera_model.focal_length_y,
        )
        self.scene.graph[self.scene.camera.name] = (
            world_T_forward_view
            @ self.forward_view_R_image_axes
            @ self.image_axes_R_optical
        )

        pixels = np.stack(
            np.meshgrid(
                np.arange(resolution.width), np.arange(resolution.height), indexing="ij"
            ),
            axis=-1,
        ).reshape(-1, 2)
        optical_V_rays = np.column_stack(
            (
                (pixels[:, 0] + 0.5 - float(camera_model.principal_point.x))
                / camera_model.focal_length_x,
                (pixels[:, 1] + 0.5 - float(camera_model.principal_point.y))
                / camera_model.focal_length_y,
                np.ones(len(pixels)),
            )
        )
        optical_V_rays /= np.linalg.norm(optical_V_rays, axis=1, keepdims=True)
        image_axes_V_rays = optical_V_rays * np.array([1.0, -1.0, -1.0])
        world_T_image_axes = self.scene.camera_transform
        directions = image_axes_V_rays @ world_T_image_axes[:3, :3].T
        origins = np.broadcast_to(world_T_image_axes[:3, 3], directions.shape).copy()
        return CameraRays(
            origin_points=origins,
            direction_vectors=directions,
            pixel_coordinates=pixels,
        )

    def ray_test(
        self,
        origin_points: np.ndarray,
        target_points: np.ndarray,
        multiple_hits: bool = False,
        min_distance: float = 0,
        max_distance: float = np.inf,
    ) -> RayIntersections:
        """
        Return ray intersections within the inclusive distance interval.

        :param origin_points: World-space starting points of the rays.
        :param target_points: World-space points defining the ray directions.
        :param multiple_hits: Whether to return every intersection instead of the first
            hit.
        :param min_distance: Inclusive minimum ray-hit distance in meters.
        :param max_distance: Inclusive maximum ray-hit distance in meters.
        :return: Hit points, source ray indices, and intersected world bodies.
        :raises RayPointShapeMismatch: If origins and targets contain different point
            counts.
        """
        origin_points = np.array(origin_points)
        target_points = np.array(target_points)
        self.update_scene()
        origin_points = origin_points.reshape((-1, 3))
        target_points = target_points.reshape((-1, 3))
        if origin_points.shape != target_points.shape:
            raise RayPointShapeMismatch(
                origin_shape=origin_points.shape, target_shape=target_points.shape
            )

        ray_directions = target_points - origin_points
        scene_geometry = trimesh.Trimesh(
            **trimesh.triangles.to_kwargs(self.scene.triangles), process=False
        )
        hit_points, ray_indices, triangle_indices = (
            scene_geometry.ray.intersects_location(
                origin_points, ray_directions, multiple_hits=multiple_hits
            )
        )
        hit_distances = np.linalg.norm(hit_points - origin_points[ray_indices], axis=1)

        valid_indices = np.where(
            (hit_distances >= min_distance) & (hit_distances <= max_distance)
        )[0]
        hit_points = hit_points[valid_indices]
        ray_indices = ray_indices[valid_indices]
        triangle_indices = triangle_indices[valid_indices]

        bodies = self.scene.triangles_node[triangle_indices]

        bodies = [self.index_to_body[self.scene_to_index[body]] for body in bodies]

        return RayIntersections(
            hit_points=hit_points, ray_indices=ray_indices, bodies=bodies
        )

    # %% Aligned RGB-D rendering
    def render(
        self,
        camera: Camera,
        rgb_mode: ColorRenderingMode = BodyColorRenderingMode(),
        depth_mode: DepthMode = OpticalAxisDepthMode(),
    ) -> RenderedRGBDFrame:
        """
        Render camera colors, body labels, and unsigned millimeter depths.

        Depth defaults to the optical-axis convention for pinhole back-projection.
        Textured rendering errors propagate to the caller.

        :param camera: Camera supplying pose, calibration, resolution, and usable range.
        :param rgb_mode: Source of the rendered color values.
        :param depth_mode: Distance convention represented by the depth values.
        :raises UnsupportedCameraProjection: If distortion is requested, or textured
            rendering cannot represent the calibration.
        :return: Aligned BGR colors, millimeter depths, segmentation, and effective
            calibration.
        """
        camera_model = camera.camera_model
        if not isinstance(camera_model, PinholeCameraModel):
            camera_model = PinholeCameraModel.from_field_of_view(
                resolution=camera_model.resolution,
                field_of_view=camera_model.field_of_view,
            )
        if not isinstance(camera_model.distortion, NoCameraDistortion):
            raise UnsupportedCameraProjection(
                reason="distorted rendering is not supported"
            )
        resolution = camera_model.resolution
        rgb_mode.validate_camera_model(camera_model)
        world_T_optical = camera.root.global_transform
        world_T_forward_view = HomogeneousTransformationMatrix(
            data=world_T_optical.to_np() @ camera.optical_R_forward_view.to_np(),
            reference_frame=world_T_optical.reference_frame,
        )
        frame = self.create_segmentation_and_depth(
            world_T_forward_view,
            resolution=resolution,
            camera_model=camera_model,
            min_distance=camera.camera_range.minimum_distance,
            max_distance=camera.camera_range.maximum_distance,
            depth_mode=depth_mode,
        )
        color_frame = rgb_mode.render(
            segmentation=frame.segmentation,
            camera_model=camera_model,
            scene=self.scene,
            world=self.world,
        )
        return RenderedRGBDFrame(
            color_bgr=color_frame.color_image,
            depth_in_millimeters=self.convert_depth_to_millimeters(frame.depth),
            segmentation=frame.segmentation,
            body_colors=color_frame.body_colors,
            camera_model=camera_model,
        )

    @staticmethod
    def convert_depth_to_millimeters(depth_in_meters: np.ndarray) -> np.ndarray:
        """
        Round and saturate meter depths to unsigned millimeters, encoding misses as
        zero.

        :param depth_in_meters: Meter depths, with negative values marking missed rays.
        :return: Rounded unsigned millimeter depths, saturated to the supported range,
            with zero for misses.
        """
        depth_in_millimeters = np.zeros(depth_in_meters.shape, dtype=np.uint16)
        valid_mask = depth_in_meters >= 0.0
        depth_in_millimeters[valid_mask] = np.clip(
            np.round(depth_in_meters[valid_mask] * RayTracer.millimeters_per_meter),
            0,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16)
        return depth_in_millimeters

    @classproperty
    def millimeters_per_meter(cls: type[RayTracer]) -> float:
        """
        Number of millimeters in one meter.

        :return: Metric conversion factor from meters to millimeters.
        """
        return 1000.0

    @classproperty
    def image_axes_R_optical(cls: type[RayTracer]) -> np.ndarray:
        """
        Map the optical negative-z viewing axis onto the intermediate x-forward axes.

        :return: Rotation from optical axes to the intermediate image axes.
        """
        return trimesh.transformations.rotation_matrix(
            angle=-np.pi / 2, direction=[0, 1, 0]
        )

    @classproperty
    def forward_view_R_image_axes(cls: type[RayTracer]) -> np.ndarray:
        """
        Map image axes onto the camera's forward-view convention.

        :return: Rotation from intermediate image axes to the forward-view axes.
        """
        return trimesh.transformations.rotation_matrix(angle=np.pi, direction=[1, 0, 0])
