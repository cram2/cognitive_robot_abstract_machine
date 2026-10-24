# %% Imports

import numpy as np
import open3d
import pytest

from robokudo.types.cv import ColorToDepthRatio
import robokudo.cas
from robokudo.cas import CAS, CASViews
from robokudo.exceptions import PointCloudCameraModelMissing
from robokudo.utils.o3d_helper import (
    place_bounding_box_on_target,
    position_bounding_box_relative_to_target,
    create_oriented_bounding_box,
    get_2d_corner_points_from_3d_bb,
    get_2d_bounding_rect_from_3d_bb,
    draw_bounding_box_wireframe,
    get_mask_from_pointcloud,
    concatenate_clouds,
    get_cloud_from_rgb_depth_and_mask,
    create_line_for_visualization,
    create_sphere_from_translation,
)
from semantic_digital_twin.spatial_types import Point2
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution


# %% Point cloud geometry
class TestUtilsO3DHelper(object):
    @pytest.fixture
    def cas(self, kinect_intrinsics: open3d.camera.PinholeCameraIntrinsic) -> CAS:
        cas = robokudo.cas.CAS()
        intrinsic_matrix = kinect_intrinsics.intrinsic_matrix
        cas.pointcloud_camera_model = PinholeCameraModel(
            image_resolution=CameraResolution(
                width=kinect_intrinsics.width,
                height=kinect_intrinsics.height,
            ),
            focal_length_x=intrinsic_matrix[0, 0],
            focal_length_y=intrinsic_matrix[1, 1],
            principal_point=Point2(x=intrinsic_matrix[0, 2], y=intrinsic_matrix[1, 2]),
        )
        cas.set(CASViews.COLOR_TO_DEPTH_RATIO, ColorToDepthRatio())
        return cas

    @pytest.fixture
    def kinect_intrinsics(self) -> open3d.camera.PinholeCameraIntrinsic:
        camera_intrinsics = open3d.camera.PinholeCameraIntrinsic()
        camera_intrinsics.set_intrinsics(
            width=1024, height=1280, fx=1050.0, fy=1050.0, cx=639.5, cy=479.5
        )
        return camera_intrinsics

    @pytest.fixture
    def camera_model(self, cas: CAS) -> PinholeCameraModel:
        """
        Provide the projection model associated with the test point cloud.
        """
        return cas.pointcloud_camera_model

    @pytest.fixture
    def pointcloud(self) -> open3d.geometry.PointCloud:
        points = np.array([[0, 0, 1], [1, 1, 1], [2, 2, 1]], dtype=np.float64)
        input_cloud = open3d.geometry.PointCloud()
        input_cloud.points = open3d.utility.Vector3dVector(points)
        return input_cloud

    @pytest.mark.parametrize(
        ["target_bounding_box", "input_bounding_box", "expected_z"],
        [
            # Zero center target_bounding_box, off center input_bounding_box, same extents
            (
                open3d.geometry.OrientedBoundingBox(
                    center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                    R=np.eye(3, dtype=np.float64),
                    extent=np.array([1.0, 1.0, 1.0], dtype=np.float64),
                ),
                open3d.geometry.OrientedBoundingBox(
                    center=np.array([1.0, 1.0, 0.0], dtype=np.float64),
                    R=np.eye(3, dtype=np.float64),
                    extent=np.array([1.0, 1.0, 1.0], dtype=np.float64),
                ),
                1.0,
            ),
            # Off center target_bounding_box and input_bounding_box, varying extents
            (
                open3d.geometry.OrientedBoundingBox(
                    center=np.array([1.0, 1.0, 1.0], dtype=np.float64),
                    R=np.eye(3, dtype=np.float64),
                    extent=np.array([2.0, 2.0, 2.0], dtype=np.float64),
                ),
                open3d.geometry.OrientedBoundingBox(
                    center=np.array([1.5, 1.5, 0.5], dtype=np.float64),
                    R=np.eye(3, dtype=np.float64),
                    extent=np.array([3.0, 3.0, 3.0], dtype=np.float64),
                ),
                3.5,
            ),
        ],
    )
    def test_place_bounding_box_on_target(
        self,
        target_bounding_box: open3d.geometry.OrientedBoundingBox,
        input_bounding_box: open3d.geometry.OrientedBoundingBox,
        expected_z: float,
    ):
        adjusted_bounding_box = place_bounding_box_on_target(
            input_bounding_box, target_bounding_box
        )

        assert np.all(
            target_bounding_box.center[:2] == adjusted_bounding_box.center[:2]
        ), "x and y were not adjusted correctly"
        assert (
            adjusted_bounding_box.center[2] == expected_z
        ), "z was not adjusted correctly"

    def test_position_bounding_box_relative_to_target_rotation(self):
        input_bounding_box = open3d.geometry.OrientedBoundingBox(
            center=[0, 0, 0], R=np.eye(3), extent=[1, 1, 1]
        )
        target_bounding_box = open3d.geometry.OrientedBoundingBox(
            center=[0, 0, 0], R=np.eye(3), extent=[1, 1, 1]
        )

        angle = np.pi / 2
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0, 0],
                [np.sin(angle), np.cos(angle), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        transformed_bounding_box = position_bounding_box_relative_to_target(
            input_bounding_box, target_bounding_box, rotation
        )

        expected_R = rotation[:3, :3]
        assert np.allclose(transformed_bounding_box.R, expected_R)
        assert np.allclose(transformed_bounding_box.extent, [1, 1, 1])

    def test_position_bounding_box_relative_to_target_identity_transform(self):
        input_bounding_box = open3d.geometry.OrientedBoundingBox(
            center=[1, 2, 3], R=np.eye(3), extent=[1, 1, 1]
        )
        target_bounding_box = open3d.geometry.OrientedBoundingBox(
            center=[0, 0, 0], R=np.eye(3), extent=[1, 1, 1]
        )

        transform = np.eye(4)

        transformed_bounding_box = position_bounding_box_relative_to_target(
            input_bounding_box, target_bounding_box, transform
        )

        assert np.allclose(transformed_bounding_box.center, target_bounding_box.center)
        assert np.allclose(transformed_bounding_box.R, target_bounding_box.R)
        assert np.allclose(transformed_bounding_box.extent, target_bounding_box.extent)

    def test_position_bounding_box_relative_to_target_non_identity_target_rotation(
        self,
    ):
        input_bounding_box = open3d.geometry.OrientedBoundingBox(
            center=[0, 0, 0], R=np.eye(3), extent=[1, 1, 1]
        )
        target_bounding_box = open3d.geometry.OrientedBoundingBox(
            center=[0, 0, 0],
            R=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            extent=[1, 1, 1],
        )

        transform = np.eye(4)

        transformed_bounding_box = position_bounding_box_relative_to_target(
            input_bounding_box, target_bounding_box, transform
        )

        expected_R = target_bounding_box.R
        assert np.allclose(transformed_bounding_box.R, expected_R)
        assert np.allclose(transformed_bounding_box.extent, [1, 1, 1])

    def test_position_bounding_box_relative_to_target_invalid_transform_matrix(self):
        input_bounding_box = open3d.geometry.OrientedBoundingBox(
            center=[0, 0, 0], R=np.eye(3), extent=[1, 1, 1]
        )
        target_bounding_box = open3d.geometry.OrientedBoundingBox(
            center=[0, 0, 0], R=np.eye(3), extent=[1, 1, 1]
        )

        transform = np.eye(3)

        with pytest.raises(ValueError):
            position_bounding_box_relative_to_target(
                input_bounding_box, target_bounding_box, transform
            )

    def test_create_oriented_bounding_box_basic(self):
        bb_size = np.array([1.0, 1.0, 1.0])
        transform = np.eye(4)
        oriented_bounding_box = create_oriented_bounding_box(bb_size, transform)

        assert isinstance(oriented_bounding_box, open3d.geometry.OrientedBoundingBox)
        assert np.allclose(oriented_bounding_box.center, [0.0, 0.0, 0.0])
        assert np.allclose(oriented_bounding_box.R, np.eye(3))
        assert np.allclose(oriented_bounding_box.extent, bb_size)

    def test_create_oriented_bounding_box_translated(self):
        bb_size = np.array([2.0, 3.0, 4.0])
        transform = np.eye(4)
        transform[:3, 3] = [1.0, 2.0, 3.0]
        oriented_bounding_box = create_oriented_bounding_box(bb_size, transform)

        assert isinstance(oriented_bounding_box, open3d.geometry.OrientedBoundingBox)
        assert np.allclose(oriented_bounding_box.center, [1.0, 2.0, 3.0])
        assert np.allclose(oriented_bounding_box.R, np.eye(3))
        assert np.allclose(oriented_bounding_box.extent, bb_size)

    def test_create_oriented_bounding_box_rotated(self):
        bb_size = np.array([1.0, 1.0, 1.0])
        # Rotated 90 degrees around Z-axis
        rotation = np.array(
            [
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        oriented_bounding_box = create_oriented_bounding_box(bb_size, rotation)

        assert isinstance(oriented_bounding_box, open3d.geometry.OrientedBoundingBox)
        assert np.allclose(oriented_bounding_box.center, [0.0, 0.0, 0.0])
        assert np.allclose(oriented_bounding_box.R, rotation[:3, :3])
        assert np.allclose(oriented_bounding_box.extent, bb_size)

    def test_create_oriented_bounding_box_invalid_size(self):
        bb_size = np.array([1.0, 1.0])
        transform = np.eye(4)
        with pytest.raises(TypeError):
            create_oriented_bounding_box(bb_size, transform)

    def test_create_oriented_bounding_box_invalid_transform(self):
        bb_size = np.array([1.0, 1.0, 1.0])
        transform = np.eye(3)
        with pytest.raises(IndexError):
            create_oriented_bounding_box(bb_size, transform)

    def test_get_2d_corner_points_from_3d_bb_basic_case(self, cas: CAS):
        oriented_bounding_box = open3d.geometry.OrientedBoundingBox(
            center=[0, 0, 1], R=np.eye(3), extent=[1, 1, 1]
        )

        result = get_2d_corner_points_from_3d_bb(cas, oriented_bounding_box)

        assert isinstance(result, np.ndarray)
        assert result.shape == (8, 2)
        assert np.all(
            result
            == [
                [-410, -570],
                [1689, -570],
                [-410, 1529],
                [289, 129],
                [989, 829],
                [289, 829],
                [989, 129],
                [1689, 1529],
            ]
        ), f"unexpected corner points, did the camera intrinsics change? {result}"

    def test_get_2d_corner_points_from_3d_bb_with_scaling(self, cas: CAS):
        cas.set(
            robokudo.cas.CASViews.COLOR_TO_DEPTH_RATIO,
            ColorToDepthRatio(horizontal=2.0, vertical=2.0),
        )

        oriented_bounding_box = open3d.geometry.OrientedBoundingBox(
            center=[0, 0, 1], R=np.eye(3), extent=[1, 1, 1]
        )

        result = get_2d_corner_points_from_3d_bb(cas, oriented_bounding_box)

        assert isinstance(result, np.ndarray)
        assert result.shape == (8, 2)
        assert np.all(
            result
            == [
                [-205, -285],
                [844, -285],
                [-205, 764],
                [144, 64],
                [494, 414],
                [144, 414],
                [494, 64],
                [844, 764],
            ]
        ), f"unexpected corner points, did the camera intrinsics change? {result}"

    def test_get_2d_corner_points_from_3d_bb_far_from_camera(self, cas: CAS):
        oriented_bounding_box = open3d.geometry.OrientedBoundingBox(
            center=[0, 0, 10], R=np.eye(3), extent=[1, 1, 1]
        )

        result = get_2d_corner_points_from_3d_bb(cas, oriented_bounding_box)

        assert isinstance(result, np.ndarray)
        assert result.shape == (8, 2)
        assert np.all(
            result
            == [
                [584, 424],
                [694, 424],
                [584, 534],
                [589, 429],
                [689, 529],
                [589, 529],
                [689, 429],
                [694, 534],
            ]
        ), f"unexpected corner points, did the camera intrinsics change? {result}"

    def test_get_2d_corner_points_from_3d_bb_requires_camera_model(
        self, cas: CAS
    ) -> None:
        """
        Projection requires calibration for the current point cloud.
        """
        cas.views.pop(CASViews.POINTCLOUD_CAMERA_MODEL)

        oriented_bounding_box = open3d.geometry.OrientedBoundingBox(
            center=[0, 0, 1], R=np.eye(3), extent=[1, 1, 1]
        )

        with pytest.raises(PointCloudCameraModelMissing):
            get_2d_corner_points_from_3d_bb(cas, oriented_bounding_box)

    def test_get_2d_corner_points_from_3d_bb_zero_depth(self, cas: CAS):
        oriented_bounding_box = open3d.geometry.OrientedBoundingBox(
            center=[0, 0, 0], R=np.eye(3), extent=[0, 0, 0]
        )

        assert pytest.raises(
            ValueError, get_2d_corner_points_from_3d_bb, cas, oriented_bounding_box
        )

    def test_get_2d_bounding_rect_from_3d_bb(self, cas: CAS):
        oriented_bounding_box = open3d.geometry.OrientedBoundingBox(
            center=[0, 0, 1], R=np.eye(3), extent=[1, 1, 1]
        )

        rect = get_2d_bounding_rect_from_3d_bb(cas, oriented_bounding_box)

        assert isinstance(rect, tuple)
        assert len(rect) == 4  # (x, y, w, h)
        assert np.all(rect == (-410, -570, 2100, 2100))
        assert all(isinstance(val, int) for val in rect)

    def test_draw_bounding_box_wireframe(self, cas: CAS):
        image = np.ones((300, 300, 3), dtype=np.uint8) * 255
        oriented_bounding_box = open3d.geometry.OrientedBoundingBox(
            center=[0, 0, 1], R=np.eye(3), extent=[1, 1, 1]
        )

        draw_bounding_box_wireframe(cas, image, oriented_bounding_box)

        non_white_pixels = np.any(image != 255, axis=2)
        assert np.any(non_white_pixels), "No lines were drawn on the image."
        assert image.shape == (300, 300, 3), "Image dimensions changed unexpectedly."

    def test_get_mask_from_pointcloud_basic(
        self,
        kinect_intrinsics: open3d.camera.PinholeCameraIntrinsic,
        pointcloud: open3d.geometry.PointCloud,
    ):
        ref_image = np.zeros(
            (kinect_intrinsics.height, kinect_intrinsics.width, 3), dtype=np.uint8
        )  # Kinect color image example

        mask = get_mask_from_pointcloud(
            pointcloud, ref_image, kinect_intrinsics, mask_scale_factor=1.0
        )

        assert mask.shape == ref_image.shape
        assert np.any(mask == 255)

    def test_get_mask_from_pointcloud_with_scale_factor(
        self,
        kinect_intrinsics: open3d.camera.PinholeCameraIntrinsic,
        pointcloud: open3d.geometry.PointCloud,
    ):
        ref_image = np.zeros(
            (640, 480, 3), dtype=np.uint8
        )  # Kinect depth image example

        mask = get_mask_from_pointcloud(
            pointcloud, ref_image, kinect_intrinsics, mask_scale_factor=2.0
        )

        assert mask.shape == (1280, 960, 3)

    def test_get_mask_from_pointcloud_with_crop(
        self,
        kinect_intrinsics: open3d.camera.PinholeCameraIntrinsic,
        pointcloud: open3d.geometry.PointCloud,
    ):
        ref_image = np.zeros(
            (640, 480, 3), dtype=np.uint8
        )  # Kinect depth image example

        mask = get_mask_from_pointcloud(
            pointcloud,
            ref_image,
            kinect_intrinsics,
            mask_scale_factor=2.0,
            crop_to_ref=True,
        )

        assert mask.shape == ref_image.shape

    def test_get_mask_from_pointcloud_empty_cloud(
        self, kinect_intrinsics: open3d.camera.PinholeCameraIntrinsic
    ):
        ref_image = np.zeros(
            (640, 480, 3), dtype=np.uint8
        )  # Kinect depth image example

        mask = get_mask_from_pointcloud(
            open3d.geometry.PointCloud(), ref_image, kinect_intrinsics
        )

        assert np.all(mask == 0)

    def test_concatenate_clouds_single_cloud(
        self, pointcloud: open3d.geometry.PointCloud
    ):
        output_cloud = concatenate_clouds([pointcloud])

        assert len(output_cloud.points) == 3
        assert np.all(output_cloud.points == pointcloud.points)

    def test_concatenate_clouds_multiple_clouds(
        self, pointcloud: open3d.geometry.PointCloud
    ):
        points1 = np.array(
            [[0.5, 0.5, 1.5], [1.5, 1.5, 1.5], [2.5, 2.5, 1.5]], dtype=np.float64
        )
        cloud1 = open3d.geometry.PointCloud()
        cloud1.points = open3d.utility.Vector3dVector(points1)

        points2 = np.array(
            [[1.5, 1.5, 2.5], [2.5, 2.5, 2.5], [3.5, 3.5, 3.5]], dtype=np.float64
        )
        cloud2 = open3d.geometry.PointCloud()
        cloud2.points = open3d.utility.Vector3dVector(points2)

        output_cloud = concatenate_clouds(
            [
                cloud1,
                cloud2,
                pointcloud,
            ]
        )

        all_input_points = np.concatenate([points1, points2, pointcloud.points], axis=0)

        assert len(output_cloud.points) == 9
        assert np.all(output_cloud.points == all_input_points)

    def test_get_cloud_from_rgb_depth_and_mask(self, camera_model: PinholeCameraModel):
        rgb_image = (
            np.ones(
                (camera_model.resolution.height, camera_model.resolution.width, 3),
                dtype=np.uint8,
            )
            * 255
        )  # White image
        depth_image = (
            np.ones(
                (camera_model.resolution.height, camera_model.resolution.width),
                dtype=np.uint16,
            )
            * 125
        )  # Grey image

        mask = np.zeros(
            (camera_model.resolution.height, camera_model.resolution.width),
            dtype=np.uint8,
        )
        mask[100:200, 100:200] = 255

        pc = get_cloud_from_rgb_depth_and_mask(
            rgb_image, depth_image, mask, camera_model
        )

        assert len(pc.points) == 10000
        assert np.all(np.asarray(pc.colors) == (1.0, 1.0, 1.0))

    def test_get_cloud_from_rgb_depth_and_mask_depth_truncate(
        self, camera_model: PinholeCameraModel
    ):
        rgb_image = (
            np.ones(
                (camera_model.resolution.height, camera_model.resolution.width, 3),
                dtype=np.uint8,
            )
            * 255
        )  # White image
        depth_image = (
            np.ones(
                (camera_model.resolution.height, camera_model.resolution.width),
                dtype=np.uint16,
            )
            * 201
            * 1000
        )  # 201mm depth

        mask = np.zeros(
            (camera_model.resolution.height, camera_model.resolution.width),
            dtype=np.uint8,
        )
        mask[100:200, 100:200] = 255

        pc = get_cloud_from_rgb_depth_and_mask(
            rgb_image, depth_image, mask, camera_model, depth_truncate=0.2
        )  # truncate from 200 mm

        assert len(pc.points) == 0

    def test_get_cloud_from_rgb_depth_and_mask_keeps_pixels_of_mask_true_value(
        self, camera_model: PinholeCameraModel
    ):
        shape = (camera_model.resolution.height, camera_model.resolution.width)
        rgb_image = np.zeros((*shape, 3), dtype=np.uint8)
        depth_image = np.full(shape, 1000, dtype=np.uint16)
        mask = np.zeros(shape, dtype=np.uint8)
        mask[100:110, 100:120] = 7
        mask[300:310, 300:310] = 255

        pc = get_cloud_from_rgb_depth_and_mask(
            rgb_image, depth_image, mask, camera_model, mask_true_value=7
        )

        assert len(pc.points) == 10 * 20

    def test_get_cloud_from_rgb_depth_and_mask_leaves_depth_image_unchanged(
        self, camera_model: PinholeCameraModel
    ):
        shape = (camera_model.resolution.height, camera_model.resolution.width)
        depth_image = np.full(shape, 1000, dtype=np.uint16)
        mask = np.zeros(shape, dtype=np.uint8)
        mask[100:200, 100:200] = 255

        get_cloud_from_rgb_depth_and_mask(
            np.zeros((*shape, 3), dtype=np.uint8), depth_image, mask, camera_model
        )

        np.testing.assert_array_equal(
            depth_image, np.full(shape, 1000, dtype=np.uint16)
        )

    @pytest.mark.parametrize(
        ["channel", "expected_color"],
        [
            (0, (1.0, 0.0, 0.0)),  # Red
            (1, (0.0, 1.0, 0.0)),  # Green
            (2, (0.0, 0.0, 1.0)),  # Blue
        ],
    )
    def test_get_cloud_from_rgb_depth_and_mask_pc_color(
        self,
        camera_model: PinholeCameraModel,
        channel: int,
        expected_color: tuple[float, float, float],
    ):
        rgb_image = np.zeros(
            (camera_model.resolution.height, camera_model.resolution.width, 3),
            dtype=np.uint8,
        )
        rgb_image[:, :, channel] = 255  # Red image

        depth_image = (
            np.ones(
                (camera_model.resolution.height, camera_model.resolution.width),
                dtype=np.uint16,
            )
            * 125
        )  # Grey image

        mask = np.zeros(
            (camera_model.resolution.height, camera_model.resolution.width),
            dtype=np.uint8,
        )
        mask[100:200, 100:200] = 255

        pc = get_cloud_from_rgb_depth_and_mask(
            rgb_image, depth_image, mask, camera_model
        )

        assert len(pc.points) == 10000
        assert np.all(np.asarray(pc.colors) == expected_color)

    def test_get_cloud_from_rgb_depth_and_mask_empty_depth(
        self, camera_model: PinholeCameraModel
    ):
        rgb_image = (
            np.ones(
                (camera_model.resolution.height, camera_model.resolution.width, 3),
                dtype=np.uint8,
            )
            * 255
        )  # White image
        depth_image = np.zeros(
            (camera_model.resolution.height, camera_model.resolution.width),
            dtype=np.uint16,
        )  # Invalid depth

        mask = np.zeros(
            (camera_model.resolution.height, camera_model.resolution.width),
            dtype=np.uint8,
        )

        pc = get_cloud_from_rgb_depth_and_mask(
            rgb_image, depth_image, mask, camera_model
        )

        assert len(pc.points) == 0

    def test_get_cloud_from_rgb_depth_and_mask_empty_mask(
        self, camera_model: PinholeCameraModel
    ):
        rgb_image = (
            np.ones(
                (camera_model.resolution.height, camera_model.resolution.width, 3),
                dtype=np.uint8,
            )
            * 255
        )  # White image
        depth_image = (
            np.ones(
                (camera_model.resolution.height, camera_model.resolution.width),
                dtype=np.uint16,
            )
            * 200
            * 1000
        )  # 200mm depth

        mask = np.zeros(
            (camera_model.resolution.height, camera_model.resolution.width),
            dtype=np.uint8,
        )

        pc = get_cloud_from_rgb_depth_and_mask(
            rgb_image, depth_image, mask, camera_model
        )

        assert len(pc.points) == 0

    def test_get_cloud_from_rgb_depth_and_mask_invalid_mask(
        self, camera_model: PinholeCameraModel
    ):
        rgb_image = (
            np.ones(
                (camera_model.resolution.height, camera_model.resolution.width, 3),
                dtype=np.uint8,
            )
            * 255
        )  # White image
        depth_image = (
            np.ones(
                (camera_model.resolution.height, camera_model.resolution.width),
                dtype=np.uint16,
            )
            * 200
            * 1000
        )  # 200mm depth

        mask = np.ones(
            (camera_model.resolution.height, camera_model.resolution.width),
            dtype=np.uint8,
        )

        pc = get_cloud_from_rgb_depth_and_mask(
            rgb_image, depth_image, mask, camera_model
        )

        assert len(pc.points) == 0

    def test_get_cloud_from_rgb_depth_and_mask_empty_mask_and_depth(
        self, camera_model: PinholeCameraModel
    ):
        rgb_image = (
            np.ones(
                (camera_model.resolution.height, camera_model.resolution.width, 3),
                dtype=np.uint8,
            )
            * 255
        )  # White image
        depth_image = np.zeros(
            (camera_model.resolution.height, camera_model.resolution.width),
            dtype=np.uint16,
        )  # Invalid depth

        mask = np.zeros(
            (camera_model.resolution.height, camera_model.resolution.width),
            dtype=np.uint8,
        )

        pc = get_cloud_from_rgb_depth_and_mask(
            rgb_image, depth_image, mask, camera_model
        )

        assert len(pc.points) == 0

    @pytest.mark.parametrize(
        ["origin", "target", "color"],
        [
            ([0, 0, 0], [1, 1, 1], [1, 0, 0]),  # Red
            ([-5, 2, 3], [10, -4, 7], [0, 1, 0]),  # Green
            ([0, 0, 0], [0, 0, 0], [0, 0, 1]),  # Blue
        ],
    )
    def test_create_line_for_visualization(
        self, origin: list[int], target: list[int], color: tuple[float, float, float]
    ):
        line_set = create_line_for_visualization(origin, target, color)

        assert np.array_equal(line_set.points, np.array([origin, target]))
        assert np.array_equal(line_set.lines, np.array([[0, 1]]))
        assert np.array_equal(line_set.colors, np.array([color]))

    @pytest.mark.parametrize(
        ["origin", "target", "color"],
        [
            ([0, 0, 0], [1, 1, 1], [1, 0]),  # Invalid color
            ([0, 0, 0], [1, 1], [1, 1, 1]),  # Invalid target
            ([0, 0], [1, 1, 1], [1, 1, 1]),  # Invalid origin
        ],
    )
    def test_create_line_for_visualization_invalid_inputs(
        self, origin: list[int], target: list[int], color: tuple[float, float, float]
    ):
        assert pytest.raises(
            Exception, create_line_for_visualization, origin, target, color
        )

    def test_create_sphere_from_translation(self):
        origin = np.array([1.0, 2.0, 3.0])
        color = [0.5, 0.5, 0.5]
        radius = 1.0

        sphere = create_sphere_from_translation(origin, color, radius)

        assert isinstance(sphere, open3d.geometry.TriangleMesh)
        assert np.allclose(sphere.get_center(), origin)
        assert np.allclose(
            np.asarray(sphere.vertices).max(axis=0)
            - np.asarray(sphere.vertices).min(axis=0),
            2 * radius,
        )
        assert np.allclose(sphere.vertex_colors[0], color)

    @pytest.mark.parametrize(
        ["origin", "color", "radius", "error"],
        [
            (
                np.array([1.0, 2.0, 3.0]),
                [0.5, 0.5, 0.5],
                0.0,
                RuntimeError,
            ),  # Invalid radius
            (
                np.array([1.0, 2.0, 3.0]),
                [0.5, 0.5, 0.5],
                -1.0,
                RuntimeError,
            ),  # Invalid radius
            (
                np.array([1.0, 2.0, 3.0]),
                [0.5, 0.5],
                1.0,
                TypeError,
            ),  # Invalid color -> only 2 elements
            (
                np.array([1.0, 2.0]),
                [0.5, 0.5, 0.5],
                1.0,
                TypeError,
            ),  # Invalid origin only 2 elements
        ],
    )
    def test_create_sphere_from_translation_invalid_inputs(
        self,
        origin: np.ndarray,
        color: tuple[float, float, float],
        radius: float,
        error: type[Exception],
    ):
        assert pytest.raises(
            error, create_sphere_from_translation, origin, color, radius
        )


# %% Semantic camera calibration


def test_masked_cloud_projects_depth_with_semantic_camera_model() -> None:
    """
    Masked depth uses the focal lengths and principal point of its camera model.
    """
    camera_model = PinholeCameraModel(
        image_resolution=CameraResolution(width=3, height=2),
        focal_length_x=2.0,
        focal_length_y=4.0,
        principal_point=Point2(x=0.5, y=0.25),
    )
    depth_image = np.zeros((2, 3), dtype=np.uint16)
    depth_image[1, 2] = 1000
    mask = np.zeros_like(depth_image, dtype=np.uint8)
    mask[1, 2] = 255

    cloud = get_cloud_from_rgb_depth_and_mask(
        rgb_image=np.zeros((2, 3, 3), dtype=np.uint8),
        depth_image=depth_image,
        mask=mask,
        camera_model=camera_model,
    )

    np.testing.assert_array_equal(
        np.asarray(cloud.points),
        [
            [
                (2 - float(camera_model.principal_point.x))
                / camera_model.focal_length_x,
                (1 - float(camera_model.principal_point.y))
                / camera_model.focal_length_y,
                1.0,
            ]
        ],
    )
