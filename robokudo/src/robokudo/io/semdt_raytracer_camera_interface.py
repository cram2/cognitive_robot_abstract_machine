"""Simulated RGB-D camera interface backed by SemDT RayTracer."""

from __future__ import annotations

import io
import time
from typing import Dict, Tuple

import numpy as np
import open3d as o3d
from PIL import Image as PILImage
from sensor_msgs.msg import CameraInfo

import robokudo.world as rk_world
from robokudo.cas import CASViews, CAS
from robokudo.io.camera_interface import CameraInterface, ROSCameraInterface
from robokudo.utils.module_loader import ModuleLoader
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body


class SemDTRayTracerCameraInterface(CameraInterface):
    """CameraInterface implementation that renders data from a SemDT world."""

    def __init__(self, camera_config):
        super().__init__(camera_config)
        self.module_loader = ModuleLoader()

    def has_new_data(self) -> bool:
        # Simulated camera can always render a fresh frame.
        return True

    def set_data(self, cas: CAS) -> None:
        world = self._load_runtime_world()
        world_frame_body = self._ensure_world_frame(world)
        camera_body = self._ensure_camera_body(world, world_frame_body)
        (
            render_camera_to_world,
            optical_cam_to_world,
        ) = self._set_camera_pose(world, world_frame_body, camera_body)

        resolution = int(self.camera_config.resolution)
        fov_deg = float(self.camera_config.fov_deg)
        min_distance = float(self.camera_config.min_distance)
        max_distance = float(self.camera_config.max_distance)

        ray_tracer = world.ray_tracer
        segmentation, depth_m = self._render_segmentation_and_depth(
            ray_tracer=ray_tracer,
            cam_to_world=render_camera_to_world,
            resolution=resolution,
            fov_deg=fov_deg,
            min_distance=min_distance,
            max_distance=max_distance,
        )

        color_bgr, object_color_map = self._render_color_image(
            world=world,
            ray_tracer=ray_tracer,
            cam_to_world=render_camera_to_world,
            segmentation=segmentation,
            resolution=resolution,
            fov_deg=fov_deg,
        )
        depth_mm = self._depth_m_to_mm(depth_m)

        cam_info, cam_intrinsic = self._build_camera_models(
            frame_id=self.camera_config.camera_frame,
            resolution=resolution,
            fov_deg=fov_deg,
        )

        timestamp_ns = time.time_ns()
        cam_info.header.stamp.sec = int(timestamp_ns // 1_000_000_000)
        cam_info.header.stamp.nanosec = int(timestamp_ns % 1_000_000_000)

        cas.set(CASViews.COLOR_IMAGE, color_bgr)
        cas.set(CASViews.DEPTH_IMAGE, depth_mm)
        cas.set(CASViews.CAM_INFO, cam_info)
        cas.set(CASViews.CAM_INTRINSIC, cam_intrinsic)
        cas.set(CASViews.COLOR2DEPTH_RATIO, self.camera_config.color2depth_ratio)
        cas.set(CASViews.OBJECT_IMAGE, segmentation)
        cas.set(CASViews.OBJECT_COLOR_MAP, object_color_map)
        # Shared SemDT ground-truth world for this frame. Consumers must treat as read-only.
        cas.set_ref(CASViews.GROUND_TRUTH_WORLD_REF, world)

        cas.cam_to_world_transform = optical_cam_to_world
        cas.data_timestamp = timestamp_ns
        ROSCameraInterface.store_legacy_cam_to_world_transform_from_cas(cas)

    def _load_runtime_world(self) -> World:
        world_descriptor = self.module_loader.load_world_descriptor(
            ros_pkg_name=self.camera_config.world_descriptor_ros_package,
            module_name=self.camera_config.world_descriptor_name,
        )
        rk_world.set_world(world_descriptor.world)
        return rk_world.world_instance()

    def _ensure_world_frame(self, world: World) -> Body:
        world_frame_name = self.camera_config.world_frame
        if world_frame_name is None:
            return world.root

        bodies = world.get_bodies_by_name(world_frame_name)
        if len(bodies) > 0:
            return bodies[0]

        with world.modify_world():
            world_frame_body = Body(name=PrefixedName(name=world_frame_name))
            root = world.root
            if root is None:
                world.add_body(world_frame_body)
            else:
                root_c_world_frame = Connection6DoF.create_with_dofs(
                    parent=root,
                    child=world_frame_body,
                    world=world,
                )
                world.add_connection(root_c_world_frame)

        return world.get_body_by_name(world_frame_name)

    def _ensure_camera_body(self, world: World, world_frame_body: Body) -> Body:
        camera_frame = self.camera_config.camera_frame
        bodies = world.get_bodies_by_name(camera_frame)
        if len(bodies) > 0:
            return bodies[0]

        with world.modify_world():
            camera_body = Body(name=PrefixedName(name=camera_frame))
            world_c_camera = Connection6DoF.create_with_dofs(
                parent=world_frame_body,
                child=camera_body,
                world=world,
            )
            world.add_connection(world_c_camera)

        return world.get_body_by_name(camera_frame)

    def _set_camera_pose(
        self, world: World, world_frame_body: Body, camera_body: Body
    ) -> Tuple[HomogeneousTransformationMatrix, HomogeneousTransformationMatrix]:
        # Config pose is given in ROS optical-frame convention:
        # x right, y down, z forward.
        camera_optical_to_world = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=float(self.camera_config.camera_x),
            y=float(self.camera_config.camera_y),
            z=float(self.camera_config.camera_z),
            roll=float(self.camera_config.camera_roll),
            pitch=float(self.camera_config.camera_pitch),
            yaw=float(self.camera_config.camera_yaw),
            reference_frame=world_frame_body,
            child_frame=camera_body,
        )
        # SemDT ray tracer expects a camera_link-like frame:
        # x forward, y left, z up.
        camera_link_to_world = HomogeneousTransformationMatrix(
            data=(
                camera_optical_to_world.to_np()
                @ SemDTRayTracerCameraInterface._camera_optical_to_link_np()
            ),
            reference_frame=world_frame_body,
        )

        with world.modify_world():
            if camera_body.parent_connection is not None:
                camera_body.parent_connection.origin = camera_optical_to_world

        return camera_link_to_world, camera_optical_to_world

    @staticmethod
    def _camera_link_to_optical_np() -> np.ndarray:
        return np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _camera_optical_to_link_np() -> np.ndarray:
        return np.linalg.inv(SemDTRayTracerCameraInterface._camera_link_to_optical_np())

    @staticmethod
    def _render_segmentation_and_depth(
        ray_tracer,
        cam_to_world: HomogeneousTransformationMatrix,
        resolution: int,
        fov_deg: float,
        min_distance: float,
        max_distance: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        segmentation = np.zeros((resolution, resolution), dtype=np.int32) - 1
        depth_m = np.zeros((resolution, resolution), dtype=np.float32) - 1.0

        ray_origins, ray_directions, pixels = ray_tracer.create_camera_rays(
            cam_to_world, resolution=resolution, fov=fov_deg
        )
        target_points = ray_origins + ray_directions * 10.0
        points, index_ray, bodies = ray_tracer.ray_test(
            ray_origins,
            target_points,
            multiple_hits=True,
            min_distance=min_distance,
            max_distance=max_distance,
        )

        if len(index_ray) == 0:
            return segmentation, depth_m

        unique_index = np.unique(index_ray, return_index=True)[1]
        index_ray = index_ray[unique_index]
        points = points[unique_index]
        body_indices = np.asarray([body.index for body in bodies], dtype=np.int32)[
            unique_index
        ]
        pixel_ray = pixels[index_ray]
        segmentation[pixel_ray[:, 0], pixel_ray[:, 1]] = body_indices

        # Trimesh camera looks along -z in its local frame. Convert hit points to
        # that camera frame and use projective z-depth (not range) for RGB-D.
        scene_camera_transform = ray_tracer.scene.graph[ray_tracer.scene.camera.name]
        if isinstance(scene_camera_transform, tuple):
            scene_camera_transform = scene_camera_transform[0]
        world_to_camera = np.linalg.inv(np.asarray(scene_camera_transform))
        points_h = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1
        )
        points_cam = (world_to_camera @ points_h.T).T
        z_depth = -points_cam[:, 2]
        valid_depth = z_depth > 0.0
        pixel_ray = pixel_ray[valid_depth]
        z_depth = z_depth[valid_depth]
        depth_m[pixel_ray[:, 0], pixel_ray[:, 1]] = z_depth.astype(np.float32)

        return segmentation, depth_m

    def _render_color_image(
        self,
        world: World,
        ray_tracer,
        cam_to_world: HomogeneousTransformationMatrix,
        segmentation: np.ndarray,
        resolution: int,
        fov_deg: float,
    ) -> Tuple[np.ndarray, Dict[str, str]]:
        rgb_mode = str(self.camera_config.rgb_mode).strip().lower()
        if rgb_mode == "trimesh":
            trimesh_image = self._try_render_trimesh_rgb(
                ray_tracer=ray_tracer,
                cam_to_world=cam_to_world,
                resolution=resolution,
                fov_deg=fov_deg,
            )
            if trimesh_image is not None:
                return trimesh_image, {}
            self.rk_logger.warning(
                "SemDTRayTracerCameraInterface: trimesh RGB rendering failed. Falling back to semantic colors."
            )

        semantic_rgb, object_color_map = self._render_semantic_rgb(
            world=world, segmentation=segmentation
        )
        return semantic_rgb[:, :, ::-1].copy(), object_color_map

    def _try_render_trimesh_rgb(
        self,
        ray_tracer,
        cam_to_world: HomogeneousTransformationMatrix,
        resolution: int,
        fov_deg: float,
    ) -> np.ndarray | None:
        try:
            # Keep RayTracer camera pose/FOV/resolution in sync with this frame.
            ray_tracer.create_camera_rays(
                cam_to_world, resolution=resolution, fov=fov_deg
            )
            png_data = ray_tracer.scene.save_image(
                resolution=(resolution, resolution), visible=False
            )
            image = np.array(PILImage.open(io.BytesIO(png_data)).convert("RGB"))
            return image[:, :, ::-1].copy()
        except Exception:
            return None

    def _render_semantic_rgb(
        self, world: World, segmentation: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, str]]:
        rgb = np.zeros(
            (segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8
        )
        object_color_map: Dict[str, str] = {}
        unique_indices = np.unique(segmentation)
        for body_index in unique_indices:
            if body_index < 0:
                continue

            body = world.kinematic_structure[int(body_index)]
            body_rgb = self._rgb_for_body(body)
            rgb[segmentation == body_index] = body_rgb
            object_color_map[f"{body_rgb[0]},{body_rgb[1]},{body_rgb[2]}"] = (
                body.name.name
            )

        return rgb, object_color_map

    @staticmethod
    def _rgb_for_body(body: Body) -> np.ndarray:
        if len(body.collision) > 0:
            color = body.collision[0].color
        elif len(body.visual) > 0:
            color = body.visual[0].color
        else:
            return np.array([128, 128, 128], dtype=np.uint8)

        return np.array(
            [
                int(np.clip(round(color.R * 255.0), 0, 255)),
                int(np.clip(round(color.G * 255.0), 0, 255)),
                int(np.clip(round(color.B * 255.0), 0, 255)),
            ],
            dtype=np.uint8,
        )

    @staticmethod
    def _depth_m_to_mm(depth_m: np.ndarray) -> np.ndarray:
        depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
        valid_mask = depth_m >= 0.0
        depth_mm[valid_mask] = np.clip(
            np.round(depth_m[valid_mask] * 1000.0), 0, np.iinfo(np.uint16).max
        ).astype(np.uint16)
        return depth_mm

    @staticmethod
    def _build_camera_models(
        frame_id: str, resolution: int, fov_deg: float
    ) -> Tuple[CameraInfo, o3d.camera.PinholeCameraIntrinsic]:
        width = int(resolution)
        height = int(resolution)
        fov_rad = np.deg2rad(float(fov_deg))
        focal = (0.5 * width) / np.tan(0.5 * fov_rad)
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0

        cam_info = CameraInfo()
        cam_info.header.frame_id = frame_id
        cam_info.width = width
        cam_info.height = height
        cam_info.distortion_model = "plumb_bob"
        cam_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        cam_info.k = [
            float(focal),
            0.0,
            float(cx),
            0.0,
            float(focal),
            float(cy),
            0.0,
            0.0,
            1.0,
        ]
        cam_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        cam_info.p = [
            float(focal),
            0.0,
            float(cx),
            0.0,
            0.0,
            float(focal),
            float(cy),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]

        cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, float(focal), float(focal), float(cx), float(cy)
        )

        return cam_info, cam_intrinsic
