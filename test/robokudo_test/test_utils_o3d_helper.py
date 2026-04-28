import numpy as np
import open3d as o3d
import pytest
import sensor_msgs.msg

import robokudo.cas
from robokudo.cas import CAS, CASViews
from robokudo.utils.o3d_helper import (
    put_obb_on_target_obb,
    transform_obb_relative_to_obb_center,
    get_obb_from_size_and_transform,
    get_2d_corner_points_from_3d_bb,
    get_2d_bounding_rect_from_3d_bb,
    draw_wireframe_of_obb_into_image,
    get_mask_from_pointcloud,
    scale_o3d_cam_intrinsics,
    concatenate_clouds,
    get_cloud_from_rgb_depth_and_mask,
    create_line_for_visualization,
    create_sphere_from_translation,
)


class TestUtilsO3DHelper(object):
    @pytest.fixture
    def cas(self, kinect_intrinsics: o3d.camera.PinholeCameraIntrinsic) -> CAS:
        cas = robokudo.cas.CAS()
        cas.set(CASViews.PC_CAM_INTRINSIC, kinect_intrinsics)
        cas.set(CASViews.COLOR2DEPTH_RATIO, (1.0, 1.0))
        return cas

    @pytest.fixture
    def kinect_intrinsics(self) -> o3d.camera.PinholeCameraIntrinsic:
        cam_intrinsics = o3d.camera.PinholeCameraIntrinsic()
        cam_intrinsics.set_intrinsics(
            width=1024, height=1280, fx=1050.0, fy=1050.0, cx=639.5, cy=479.5
        )
        return cam_intrinsics

    @pytest.fixture
    def pointcloud(self) -> o3d.geometry.PointCloud:
        points = np.array([[0, 0, 1], [1, 1, 1], [2, 2, 1]], dtype=np.float64)
        input_cloud = o3d.geometry.PointCloud()
        input_cloud.points = o3d.utility.Vector3dVector(points)
        return input_cloud

    @pytest.mark.parametrize(
        ["obb1", "obb2", "expected_z"],
        [
            # Zero center obb1, off center obb2, same extents
            (
                o3d.geometry.OrientedBoundingBox(
                    center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                    R=np.eye(3, dtype=np.float64),
                    extent=np.array([1.0, 1.0, 1.0], dtype=np.float64),
                ),
                o3d.geometry.OrientedBoundingBox(
                    center=np.array([1.0, 1.0, 0.0], dtype=np.float64),
                    R=np.eye(3, dtype=np.float64),
                    extent=np.array([1.0, 1.0, 1.0], dtype=np.float64),
                ),
                1.0,
            ),
            # Off center obb1 and obb2, varying extents
            (
                o3d.geometry.OrientedBoundingBox(
                    center=np.array([1.0, 1.0, 1.0], dtype=np.float64),
                    R=np.eye(3, dtype=np.float64),
                    extent=np.array([2.0, 2.0, 2.0], dtype=np.float64),
                ),
                o3d.geometry.OrientedBoundingBox(
                    center=np.array([1.5, 1.5, 0.5], dtype=np.float64),
                    R=np.eye(3, dtype=np.float64),
                    extent=np.array([3.0, 3.0, 3.0], dtype=np.float64),
                ),
                3.5,
            ),
        ],
    )
    def test_put_obb_on_target_obb(
        self,
        obb1: o3d.geometry.OrientedBoundingBox,
        obb2: o3d.geometry.OrientedBoundingBox,
        expected_z: float,
    ):
        adjusted_obb = put_obb_on_target_obb(obb2, obb1)

        assert np.all(
            obb1.center[:2] == adjusted_obb.center[:2]
        ), "x and y were not adjusted correctly"
        assert adjusted_obb.center[2] == expected_z, "z was not adjusted correctly"

    def test_transform_obb_relative_to_obb_center_rotation(self):
        input_obb = o3d.geometry.OrientedBoundingBox(
            center=[0, 0, 0], R=np.eye(3), extent=[1, 1, 1]
        )
        target_obb = o3d.geometry.OrientedBoundingBox(
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

        transformed_obb = transform_obb_relative_to_obb_center(
            input_obb, target_obb, rotation
        )

        expected_R = rotation[:3, :3]
        assert np.allclose(transformed_obb.R, expected_R)
        assert np.allclose(transformed_obb.extent, [1, 1, 1])

    def test_transform_obb_relative_to_obb_identity_transform(self):
        input_obb = o3d.geometry.OrientedBoundingBox(
            center=[1, 2, 3], R=np.eye(3), extent=[1, 1, 1]
        )
        target_obb = o3d.geometry.OrientedBoundingBox(
            center=[0, 0, 0], R=np.eye(3), extent=[1, 1, 1]
        )

        transform = np.eye(4)

        transformed_obb = transform_obb_relative_to_obb_center(
            input_obb, target_obb, transform
        )

        assert np.allclose(transformed_obb.center, target_obb.center)
        assert np.allclose(transformed_obb.R, target_obb.R)
        assert np.allclose(transformed_obb.extent, target_obb.extent)

    def test_transform_obb_relative_to_obb_non_identity_target_rotation(self):
        input_obb = o3d.geometry.OrientedBoundingBox(
            center=[0, 0, 0], R=np.eye(3), extent=[1, 1, 1]
        )
        target_obb = o3d.geometry.OrientedBoundingBox(
            center=[0, 0, 0],
            R=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            extent=[1, 1, 1],
        )

        transform = np.eye(4)

        transformed_obb = transform_obb_relative_to_obb_center(
            input_obb, target_obb, transform
        )

        expected_R = target_obb.R
        assert np.allclose(transformed_obb.R, expected_R)
        assert np.allclose(transformed_obb.extent, [1, 1, 1])

    def test_transform_obb_relative_to_obb_invalid_transform_matrix(self):
        input_obb = o3d.geometry.OrientedBoundingBox(
            center=[0, 0, 0], R=np.eye(3), extent=[1, 1, 1]
        )
        target_obb = o3d.geometry.OrientedBoundingBox(
            center=[0, 0, 0], R=np.eye(3), extent=[1, 1, 1]
        )

        transform = np.eye(3)

        with pytest.raises(ValueError):
            transform_obb_relative_to_obb_center(input_obb, target_obb, transform)

    def test_get_obb_from_size_and_transform_basic(self):
        bb_size = np.array([1.0, 1.0, 1.0])
        transform = np.eye(4)
        obb = get_obb_from_size_and_transform(bb_size, transform)

        assert isinstance(obb, o3d.geometry.OrientedBoundingBox)
        assert np.allclose(obb.center, [0.0, 0.0, 0.0])
        assert np.allclose(obb.R, np.eye(3))
        assert np.allclose(obb.extent, bb_size)

    def test_get_obb_from_size_and_transform_translated(self):
        bb_size = np.array([2.0, 3.0, 4.0])
        transform = np.eye(4)
        transform[:3, 3] = [1.0, 2.0, 3.0]
        obb = get_obb_from_size_and_transform(bb_size, transform)

        assert isinstance(obb, o3d.geometry.OrientedBoundingBox)
        assert np.allclose(obb.center, [1.0, 2.0, 3.0])
        assert np.allclose(obb.R, np.eye(3))
        assert np.allclose(obb.extent, bb_size)

    def test_get_obb_from_size_and_transform_rotated(self):
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
        obb = get_obb_from_size_and_transform(bb_size, rotation)

        assert isinstance(obb, o3d.geometry.OrientedBoundingBox)
        assert np.allclose(obb.center, [0.0, 0.0, 0.0])
        assert np.allclose(obb.R, rotation[:3, :3])
        assert np.allclose(obb.extent, bb_size)

    def test_get_obb_from_size_and_transform_invalid_size(self):
        bb_size = np.array([1.0, 1.0])
        transform = np.eye(4)
        with pytest.raises(TypeError):
            get_obb_from_size_and_transform(bb_size, transform)

    def test_get_obb_from_size_and_transform_invalid_transform(self):
        bb_size = np.array([1.0, 1.0, 1.0])
        transform = np.eye(3)
        with pytest.raises(IndexError):
            get_obb_from_size_and_transform(bb_size, transform)

    def test_get_2d_corner_points_from_3d_bb_basic_case(self, cas: CAS):
        obb = o3d.geometry.OrientedBoundingBox(
            center=[0, 0, 1], R=np.eye(3), extent=[1, 1, 1]
        )

        result = get_2d_corner_points_from_3d_bb(cas, obb)

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
        cas.set(robokudo.cas.CASViews.COLOR2DEPTH_RATIO, (2.0, 2.0))

        obb = o3d.geometry.OrientedBoundingBox(
            center=[0, 0, 1], R=np.eye(3), extent=[1, 1, 1]
        )

        result = get_2d_corner_points_from_3d_bb(cas, obb)

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

    def test_get_2d_corner_points_from_3d_bb_far_from_cam(self, cas: CAS):
        obb = o3d.geometry.OrientedBoundingBox(
            center=[0, 0, 10], R=np.eye(3), extent=[1, 1, 1]
        )

        result = get_2d_corner_points_from_3d_bb(cas, obb)

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

    def test_get_2d_corner_points_from_3d_bb_invalid_intrinsics(self, cas: CAS):
        cas.set(robokudo.cas.CASViews.PC_CAM_INTRINSIC, "anything but CameraIntrinsics")

        obb = o3d.geometry.OrientedBoundingBox(
            center=[0, 0, 1], R=np.eye(3), extent=[1, 1, 1]
        )

        assert pytest.raises(AssertionError, get_2d_corner_points_from_3d_bb, cas, obb)

    def test_get_2d_corner_points_from_3d_bb_zero_depth(self, cas: CAS):
        obb = o3d.geometry.OrientedBoundingBox(
            center=[0, 0, 0], R=np.eye(3), extent=[0, 0, 0]
        )

        assert pytest.raises(ValueError, get_2d_corner_points_from_3d_bb, cas, obb)

    def test_get_2d_bounding_rect_from_3d_bb(self, cas: CAS):
        obb = o3d.geometry.OrientedBoundingBox(
            center=[0, 0, 1], R=np.eye(3), extent=[1, 1, 1]
        )

        rect = get_2d_bounding_rect_from_3d_bb(cas, obb)

        assert isinstance(rect, tuple)
        assert len(rect) == 4  # (x, y, w, h)
        assert np.all(rect == (-410, -570, 2100, 2100))
        assert all(isinstance(val, int) for val in rect)

    def test_draw_wireframe_of_obb_into_image(self, cas: CAS):
        image = np.ones((300, 300, 3), dtype=np.uint8) * 255
        obb = o3d.geometry.OrientedBoundingBox(
            center=[0, 0, 1], R=np.eye(3), extent=[1, 1, 1]
        )

        draw_wireframe_of_obb_into_image(cas, image, obb)

        non_white_pixels = np.any(image != 255, axis=2)
        assert np.any(non_white_pixels), "No lines were drawn on the image."
        assert image.shape == (300, 300, 3), "Image dimensions changed unexpectedly."

    def test_get_mask_from_pointcloud_basic(
        self,
        kinect_intrinsics: o3d.camera.PinholeCameraIntrinsic,
        pointcloud: o3d.geometry.PointCloud,
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
        kinect_intrinsics: o3d.camera.PinholeCameraIntrinsic,
        pointcloud: o3d.geometry.PointCloud,
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
        kinect_intrinsics: o3d.camera.PinholeCameraIntrinsic,
        pointcloud: o3d.geometry.PointCloud,
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
        self, kinect_intrinsics: o3d.camera.PinholeCameraIntrinsic
    ):
        ref_image = np.zeros(
            (640, 480, 3), dtype=np.uint8
        )  # Kinect depth image example

        mask = get_mask_from_pointcloud(
            o3d.geometry.PointCloud(), ref_image, kinect_intrinsics
        )

        assert np.all(mask == 0)

    def test_scale_o3d_cam_intrinsics(
        self, kinect_intrinsics: o3d.camera.PinholeCameraIntrinsic
    ):
        new_intrinsics = scale_o3d_cam_intrinsics(kinect_intrinsics, 1.0, 1.0)

        assert np.all(
            kinect_intrinsics.intrinsic_matrix == new_intrinsics.intrinsic_matrix
        )
        assert (
            kinect_intrinsics.width == new_intrinsics.width
        ), "Image width was changed unexpectedly."
        assert (
            kinect_intrinsics.height == new_intrinsics.height
        ), "Image height was changed unexpectedly."

    def test_scale_o3d_cam_intrinsics_scale_x(
        self, kinect_intrinsics: o3d.camera.PinholeCameraIntrinsic
    ):
        new_intrinsics = scale_o3d_cam_intrinsics(kinect_intrinsics, 2.0, 1.0)

        assert np.all(
            kinect_intrinsics.intrinsic_matrix[1, :2]
            == new_intrinsics.intrinsic_matrix[1, :2]
        ), "fy and cy changed unexpectedly."
        assert (
            kinect_intrinsics.height == new_intrinsics.height
        ), "Image height was changed unexpectedly."

        assert np.allclose(
            new_intrinsics.intrinsic_matrix[0, :2],
            kinect_intrinsics.intrinsic_matrix[0, :2] * 2.0,
        ), "fx and cx were not scaled correctly."
        assert (
            kinect_intrinsics.width * 2.0 == new_intrinsics.width
        ), "Image width was not scaled correctly."

    def test_scale_o3d_cam_intrinsics_scale_y(
        self, kinect_intrinsics: o3d.camera.PinholeCameraIntrinsic
    ):
        new_intrinsics = scale_o3d_cam_intrinsics(kinect_intrinsics, 1.0, 2.0)

        assert np.all(
            kinect_intrinsics.intrinsic_matrix[0, :2]
            == new_intrinsics.intrinsic_matrix[0, :2]
        ), "fx and cx changed unexpectedly."
        assert (
            kinect_intrinsics.width == new_intrinsics.width
        ), "Image width was changed unexpectedly."

        assert np.allclose(
            new_intrinsics.intrinsic_matrix[1, :2],
            kinect_intrinsics.intrinsic_matrix[1, :2] * 2.0,
        ), "fy and cy were not scaled correctly."
        assert (
            kinect_intrinsics.height * 2.0 == new_intrinsics.height
        ), "Image height was not scaled correctly."

    def test_scale_o3d_cam_intrinsics_scale_xy(
        self, kinect_intrinsics: o3d.camera.PinholeCameraIntrinsic
    ):
        new_intrinsics = scale_o3d_cam_intrinsics(kinect_intrinsics, 2.0, 2.0)

        assert np.allclose(
            new_intrinsics.intrinsic_matrix[1, :2],
            kinect_intrinsics.intrinsic_matrix[1, :2] * 2.0,
        ), "fy and cy were not scaled correctly."
        assert np.allclose(
            new_intrinsics.intrinsic_matrix[0, :2],
            kinect_intrinsics.intrinsic_matrix[0, :2] * 2.0,
        ), "fx and cx were not scaled correctly."
        assert (
            kinect_intrinsics.width * 2.0 == new_intrinsics.width
        ), "Image width was not scaled correctly."
        assert (
            kinect_intrinsics.height * 2.0 == new_intrinsics.height
        ), "Image height was not scaled correctly."

    def test_concatenate_clouds_single_cloud(self, pointcloud: o3d.geometry.PointCloud):
        output_cloud = concatenate_clouds([pointcloud])

        assert len(output_cloud.points) == 3
        assert np.all(output_cloud.points == pointcloud.points)

    def test_concatenate_clouds_multiple_clouds(
        self, pointcloud: o3d.geometry.PointCloud
    ):
        points1 = np.array(
            [[0.5, 0.5, 1.5], [1.5, 1.5, 1.5], [2.5, 2.5, 1.5]], dtype=np.float64
        )
        cloud1 = o3d.geometry.PointCloud()
        cloud1.points = o3d.utility.Vector3dVector(points1)

        points2 = np.array(
            [[1.5, 1.5, 2.5], [2.5, 2.5, 2.5], [3.5, 3.5, 3.5]], dtype=np.float64
        )
        cloud2 = o3d.geometry.PointCloud()
        cloud2.points = o3d.utility.Vector3dVector(points2)

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

    def test_get_cloud_from_rgb_depth_and_mask(
        self, kinect_intrinsics: o3d.camera.PinholeCameraIntrinsic
    ):
        rgb_image = (
            np.ones(
                (kinect_intrinsics.height, kinect_intrinsics.width, 3), dtype=np.uint8
            )
            * 255
        )  # White image
        depth_image = (
            np.ones(
                (kinect_intrinsics.height, kinect_intrinsics.width), dtype=np.uint16
            )
            * 125
        )  # Grey image

        mask = np.zeros(
            (kinect_intrinsics.height, kinect_intrinsics.width), dtype=np.uint8
        )
        mask[100:200, 100:200] = 255

        pc = get_cloud_from_rgb_depth_and_mask(
            rgb_image, depth_image, mask, kinect_intrinsics
        )

        assert len(pc.points) == 10000
        assert np.all(np.asarray(pc.colors) == (1.0, 1.0, 1.0))

    def test_get_cloud_from_rgb_depth_and_mask_depth_truncate(
        self, kinect_intrinsics: o3d.camera.PinholeCameraIntrinsic
    ):
        rgb_image = (
            np.ones(
                (kinect_intrinsics.height, kinect_intrinsics.width, 3), dtype=np.uint8
            )
            * 255
        )  # White image
        depth_image = (
            np.ones(
                (kinect_intrinsics.height, kinect_intrinsics.width), dtype=np.uint16
            )
            * 201
            * 1000
        )  # 201mm depth

        mask = np.zeros(
            (kinect_intrinsics.height, kinect_intrinsics.width), dtype=np.uint8
        )
        mask[100:200, 100:200] = 255

        pc = get_cloud_from_rgb_depth_and_mask(
            rgb_image, depth_image, mask, kinect_intrinsics, depth_truncate=0.2
        )  # truncate from 200 mm

        assert len(pc.points) == 0

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
        kinect_intrinsics: o3d.camera.PinholeCameraIntrinsic,
        channel: int,
        expected_color: tuple[float, float, float],
    ):
        rgb_image = np.zeros(
            (kinect_intrinsics.height, kinect_intrinsics.width, 3), dtype=np.uint8
        )
        rgb_image[:, :, channel] = 255  # Red image

        depth_image = (
            np.ones(
                (kinect_intrinsics.height, kinect_intrinsics.width), dtype=np.uint16
            )
            * 125
        )  # Grey image

        mask = np.zeros(
            (kinect_intrinsics.height, kinect_intrinsics.width), dtype=np.uint8
        )
        mask[100:200, 100:200] = 255

        pc = get_cloud_from_rgb_depth_and_mask(
            rgb_image, depth_image, mask, kinect_intrinsics
        )

        assert len(pc.points) == 10000
        assert np.all(np.asarray(pc.colors) == expected_color)

    def test_get_cloud_from_rgb_depth_and_mask_empty_depth(
        self, kinect_intrinsics: o3d.camera.PinholeCameraIntrinsic
    ):
        rgb_image = (
            np.ones(
                (kinect_intrinsics.height, kinect_intrinsics.width, 3), dtype=np.uint8
            )
            * 255
        )  # White image
        depth_image = np.zeros(
            (kinect_intrinsics.height, kinect_intrinsics.width), dtype=np.uint16
        )  # Invalid depth

        mask = np.zeros(
            (kinect_intrinsics.height, kinect_intrinsics.width), dtype=np.uint8
        )

        pc = get_cloud_from_rgb_depth_and_mask(
            rgb_image, depth_image, mask, kinect_intrinsics
        )

        assert len(pc.points) == 0

    def test_get_cloud_from_rgb_depth_and_mask_empty_mask(
        self, kinect_intrinsics: o3d.camera.PinholeCameraIntrinsic
    ):
        rgb_image = (
            np.ones(
                (kinect_intrinsics.height, kinect_intrinsics.width, 3), dtype=np.uint8
            )
            * 255
        )  # White image
        depth_image = (
            np.ones(
                (kinect_intrinsics.height, kinect_intrinsics.width), dtype=np.uint16
            )
            * 200
            * 1000
        )  # 200mm depth

        mask = np.zeros(
            (kinect_intrinsics.height, kinect_intrinsics.width), dtype=np.uint8
        )

        pc = get_cloud_from_rgb_depth_and_mask(
            rgb_image, depth_image, mask, kinect_intrinsics
        )

        assert len(pc.points) == 0

    def test_get_cloud_from_rgb_depth_and_mask_invalid_mask(
        self, kinect_intrinsics: o3d.camera.PinholeCameraIntrinsic
    ):
        rgb_image = (
            np.ones(
                (kinect_intrinsics.height, kinect_intrinsics.width, 3), dtype=np.uint8
            )
            * 255
        )  # White image
        depth_image = (
            np.ones(
                (kinect_intrinsics.height, kinect_intrinsics.width), dtype=np.uint16
            )
            * 200
            * 1000
        )  # 200mm depth

        mask = np.ones(
            (kinect_intrinsics.height, kinect_intrinsics.width), dtype=np.uint8
        )

        pc = get_cloud_from_rgb_depth_and_mask(
            rgb_image, depth_image, mask, kinect_intrinsics
        )

        assert len(pc.points) == 0

    def test_get_cloud_from_rgb_depth_and_mask_empty_mask_and_depth(
        self, kinect_intrinsics: o3d.camera.PinholeCameraIntrinsic
    ):
        rgb_image = (
            np.ones(
                (kinect_intrinsics.height, kinect_intrinsics.width, 3), dtype=np.uint8
            )
            * 255
        )  # White image
        depth_image = np.zeros(
            (kinect_intrinsics.height, kinect_intrinsics.width), dtype=np.uint16
        )  # Invalid depth

        mask = np.zeros(
            (kinect_intrinsics.height, kinect_intrinsics.width), dtype=np.uint8
        )

        pc = get_cloud_from_rgb_depth_and_mask(
            rgb_image, depth_image, mask, kinect_intrinsics
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

        assert isinstance(sphere, o3d.geometry.TriangleMesh)
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
