import copy

import numpy as np
import open3d as o3d
import pytest

import robokudo.types.tf
from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.cas import CAS, CASViews
from robokudo.pipeline import Pipeline
from robokudo.types.annotation import PoseAnnotation
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.annotator_helper import (
    transform_pose_from_cam_to_world,
    transform_pose_from_world_to_cam,
    transform_cloud_from_world_to_cam,
    transform_cloud_from_cam_to_world,
    get_cam_to_world_transform_matrix,
    get_world_to_cam_transform_matrix,
    draw_bounding_boxes_from_object_hypotheses,
    scale_cam_intrinsics,
    get_color_image,
    resize_mask,
    generate_source_name,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


class TestUtilsAnnotatorHelper(object):
    @pytest.fixture()
    def annotator_in_pipeline(self) -> BaseAnnotator:
        root = Pipeline("Sequence")
        annotator = ImagePreprocessorAnnotator("1")
        root.add_child(annotator)
        return annotator

    @pytest.fixture()
    def kinect_intrinsics(self) -> o3d.camera.PinholeCameraIntrinsic:
        cam_intrinsics = o3d.camera.PinholeCameraIntrinsic()
        cam_intrinsics.set_intrinsics(
            width=1024, height=1280, fx=1050.0, fy=1050.0, cx=639.5, cy=479.5
        )
        return cam_intrinsics

    @pytest.fixture()
    def identity_cas(self) -> CAS:
        """Creates a CAS containing an identify cam to world transform."""
        cas = robokudo.cas.CAS()

        # cam_to_world = robokudo.types.tf.StampedTransform()
        # cam_to_world.child_frame = 'map'
        # cam_to_world.frame = 'head_rgbd_sensor_rgb_frame'
        # cam_to_world.rotation = [0.0, 0.0, 0.0, 1.0]
        # cam_to_world.translation = [0.0, 0.0, 0.0]
        # cam_to_world.timestamp = None  # rospy.Time[1741809308248347990] would be an example value if needed

        # TODO Set frame names?
        cas.cam_to_world_transform = (
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                pos_x=0.0,
                pos_y=0.0,
                pos_z=0.0,
                quat_x=0.0,
                quat_y=0.0,
                quat_z=0.0,
                quat_w=1.0,
            )
        )
        # cas.set(CASViews.VIEWPOINT_CAM_TO_WORLD, cam_to_world)
        return cas

    @pytest.fixture()
    def cam_to_world_cas(self) -> CAS:
        """Creates a CAS containing a valid non-identity cam to world transform."""
        cas = CAS()

        # Create a fake cam to world transform
        # cam_to_world = robokudo.types.tf.StampedTransform()
        # cam_to_world.child_frame = 'map'
        # cam_to_world.frame = 'head_rgbd_sensor_rgb_frame'
        # cam_to_world.rotation = [0.6586514783471038, -0.009324217076086938, 0.006825388323024484, -0.7523594241593126]
        # cam_to_world.translation = [2.6818742474793744, 1.9778799779168073, 0.9607137539544703]
        # cam_to_world.timestamp = None  # rospy.Time[1741809308248347990] would be an example value if needed
        #
        # cas.set(CASViews.VIEWPOINT_CAM_TO_WORLD, cam_to_world)

        cas.cam_to_world_transform = (
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                pos_x=2.6818742474793744,
                pos_y=1.9778799779168073,
                pos_z=0.9607137539544703,
                quat_x=0.6586514783471038,
                quat_y=-0.009324217076086938,
                quat_z=0.006825388323024484,
                quat_w=-0.7523594241593126,
            )
        )
        return cas

    ######################
    # Pose Transforms #
    ######################

    def test_pose_transform_helper(self, cam_to_world_cas: CAS):
        """Test pose transform round trip."""
        pos = PoseAnnotation()
        pos.source = "test"
        pos.translation.insert(0, 1)
        pos.translation.insert(1, 2)
        pos.translation.insert(2, 3)

        pos.rotation.insert(0, 0)
        pos.rotation.insert(1, 0)
        pos.rotation.insert(2, 0)
        pos.rotation.insert(3, 1)

        pose_in_world = transform_pose_from_cam_to_world(cam_to_world_cas, pos)
        pose_back = transform_pose_from_world_to_cam(cam_to_world_cas, pose_in_world)

        np.testing.assert_array_almost_equal(pos.translation, pose_back.translation)
        np.testing.assert_array_almost_equal(pos.rotation, pose_back.rotation)

    def test_pose_transform_helper_identy_tf(self, identity_cas: CAS):
        """Test pose transform with identity transform."""
        pos = PoseAnnotation()
        pos.source = "test"
        pos.translation.insert(0, 1)
        pos.translation.insert(1, 2)
        pos.translation.insert(2, 3)

        pos.rotation.insert(0, 0)
        pos.rotation.insert(1, 0)
        pos.rotation.insert(2, 0)
        pos.rotation.insert(3, 1)

        pose_in_world = transform_pose_from_cam_to_world(identity_cas, pos)

        np.testing.assert_array_almost_equal(
            pos.translation,
            pose_in_world.translation,
            err_msg="pose translation was modified during identity transform",
        )
        np.testing.assert_array_almost_equal(
            pos.rotation,
            pose_in_world.rotation,
            err_msg="pose rotation was modified during identity transform",
        )

        pose_in_cam = transform_pose_from_world_to_cam(identity_cas, pos)

        np.testing.assert_array_almost_equal(
            pos.translation,
            pose_in_cam.translation,
            err_msg="pose translation was modified during identity transform",
        )
        np.testing.assert_array_almost_equal(
            pos.rotation,
            pose_in_cam.rotation,
            err_msg="pose rotation was modified during identity transform",
        )

    ######################
    # Cloud Transforms #
    ######################

    def test_cloud_transform_helper(self, cam_to_world_cas: CAS):
        """Test cloud transform round trip."""
        x = np.linspace(-10, 10, 20)
        y = np.linspace(-10, 10, 20)
        z = np.linspace(-10, 10, 20)
        xx, yy, zz = np.meshgrid(x, y, z)
        points = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)

        cloud_in_world = transform_cloud_from_cam_to_world(cam_to_world_cas, cloud)
        cloud_back = transform_cloud_from_world_to_cam(cam_to_world_cas, cloud_in_world)

        dists = cloud.compute_point_cloud_distance(cloud_back)
        assert np.all(np.asarray(dists) < 1.0e-6)

    def test_cloud_transform_helper_in_place(self, cam_to_world_cas: CAS):
        """Test cloud transform round trip."""
        x = np.linspace(-10, 10, 20)
        y = np.linspace(-10, 10, 20)
        z = np.linspace(-10, 10, 20)
        xx, yy, zz = np.meshgrid(x, y, z)
        points = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)

        cloud_in_world = transform_cloud_from_cam_to_world(
            cam_to_world_cas, cloud, transform_inplace=True
        )
        assert (
            cloud_in_world == cloud
        ), f"different object instance was returned on in-place transform"

        cloud_back = transform_cloud_from_world_to_cam(
            cam_to_world_cas, cloud_in_world, transform_inplace=True
        )

        assert (
            cloud_back == cloud
        ), f"different object instance was returned on in-place transform"

        dists = cloud.compute_point_cloud_distance(cloud_back)
        assert np.all(np.asarray(dists) < 1.0e-6)

    def test_cloud_transform_helper_identy_tf(self, identity_cas: CAS):
        """Test cloud transform with identity transform."""
        points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)

        cloud_in_world = transform_cloud_from_cam_to_world(identity_cas, cloud)

        dists_in_world = cloud.compute_point_cloud_distance(cloud_in_world)
        assert np.all(
            np.asarray(dists_in_world) < 1.0e-6
        ), "transform_cloud_from_cam_to_world changed points in identify transform"

        cloud_in_cam = transform_cloud_from_world_to_cam(identity_cas, cloud)

        dists_in_cam = cloud.compute_point_cloud_distance(cloud_in_cam)
        assert np.all(
            np.asarray(dists_in_cam) < 1.0e-6
        ), "transform_cloud_from_world_to_cam changed points in identify transform"

    ######################
    # Transform Matrices #
    ######################

    def test_get_cam_to_world_transform_matrix(self, cam_to_world_cas: CAS):
        tf_matrix = get_cam_to_world_transform_matrix(cam_to_world_cas)
        np.testing.assert_allclose(
            tf_matrix,
            np.array(
                [
                    [
                        0.999732946100316,
                        -0.0020125282664391515,
                        0.023021429398708673,
                        2.6818742474793744,
                    ],
                    [
                        -0.022553109179936744,
                        0.132263288290829,
                        0.9909580111371092,
                        1.9778799779168073,
                    ],
                    [
                        -0.005039220961696124,
                        -0.9912125767465192,
                        0.1321825780941852,
                        0.9607137539544703,
                    ],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )

    def test_get_cam_to_world_transform_matrix_identity_tf(self, identity_cas: CAS):
        tf_matrix = get_cam_to_world_transform_matrix(identity_cas)
        np.testing.assert_allclose(tf_matrix, np.eye(4))

    def test_get_world_to_cam_transform_matrix(self, cam_to_world_cas: CAS):
        tf_matrix = get_world_to_cam_transform_matrix(cam_to_world_cas)
        np.testing.assert_allclose(
            tf_matrix,
            np.linalg.inv(
                np.array(
                    [
                        [
                            0.999732946100316,
                            -0.0020125282664391515,
                            0.023021429398708673,
                            2.6818742474793744,
                        ],
                        [
                            -0.022553109179936744,
                            0.132263288290829,
                            0.9909580111371092,
                            1.9778799779168073,
                        ],
                        [
                            -0.005039220961696124,
                            -0.9912125767465192,
                            0.1321825780941852,
                            0.9607137539544703,
                        ],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            ),
        )

    def test_get_world_to_cam_transform_matrix_identity_tf(self, identity_cas: CAS):
        tf_matrix = get_world_to_cam_transform_matrix(identity_cas)
        np.testing.assert_allclose(tf_matrix, np.eye(4))

    def test_draw_bounding_boxes(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        oh = ObjectHypothesis()
        oh.roi.roi.pos.x = 10
        oh.roi.roi.pos.y = 10
        oh.roi.roi.width = 20
        oh.roi.roi.height = 20

        def text_function(oh):
            return "Test Label"

        draw_bounding_boxes_from_object_hypotheses(img, [oh], text_function)

        assert np.any(img[10, 10] != 0)  # Top-left corner of the box
        assert np.any(img[30, 10] != 0)  # Bottom-right corner of the box
        assert np.any(img[10, 30] != 0)  # Top-right corner of the box
        assert np.any(img[30, 30] != 0)  # Bottom-right corner of the box

        text_roi = img[0:9, 10:30]  # Text area
        assert np.any(text_roi != 0)

    def test_draw_bounding_boxes_empty_list(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img_copy = img.copy()
        draw_bounding_boxes_from_object_hypotheses(img, [], lambda oh: "Test")
        assert np.array_equal(img, img_copy)

    def test_draw_bounding_boxes_invalid_objects(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img_copy = img.copy()
        invalid_oh = "Not an ObjectHypothesis"
        draw_bounding_boxes_from_object_hypotheses(img, [invalid_oh], lambda oh: "Test")
        assert np.array_equal(img, img_copy)

    @pytest.mark.parametrize(
        "scale_factor",
        [
            (1.0, 1.0),  # No scaling
            (2.0, 2.0),  # Upscaling
            (0.5, 0.5),  # Downscaling
            (3.0, 2.0),  # Non-uniform scaling
        ],
    )
    def test_scale_cam_intrinsics(
        self,
        scale_factor: tuple[float, float],
        annotator_in_pipeline: BaseAnnotator,
        kinect_intrinsics: o3d.camera.PinholeCameraIntrinsic,
    ):
        annotator_in_pipeline.descriptor.parameters.global_with_depth = True
        width, height = kinect_intrinsics.width, kinect_intrinsics.height
        scalex, scaley = scale_factor

        cas = annotator_in_pipeline.get_cas()
        cas.set(robokudo.cas.CASViews.COLOR2DEPTH_RATIO, (scalex, scaley))
        annotator_in_pipeline.cam_intrinsics = copy.deepcopy(kinect_intrinsics)

        scale_cam_intrinsics(annotator_in_pipeline)
        assert annotator_in_pipeline.cam_intrinsics.width == int(
            width * scalex
        ), f"Width should be scaled to {scalex}"
        assert annotator_in_pipeline.cam_intrinsics.height == int(
            height * scaley
        ), f"Height should be scaled to {scaley}"

        scaled_matrix = copy.deepcopy(kinect_intrinsics.intrinsic_matrix)
        scaled_matrix[0, [0, 2]] *= scalex
        scaled_matrix[1, [1, 2]] *= scaley
        assert np.array_equal(
            annotator_in_pipeline.cam_intrinsics.intrinsic_matrix, scaled_matrix
        ), f"Intrinsics should be scaled to {scalex}, {scaley}"

    @pytest.mark.parametrize(
        "scale_factor",
        [
            (1.0, 1.0),  # No scaling
            (2.0, 2.0),  # Upscaling
            (0.5, 0.5),  # Downscaling
            (3.0, 2.0),  # Non-uniform scaling
            (2.0, 3.0),  # Non-uniform scaling
        ],
    )
    def test_get_color_image(
        self,
        scale_factor: tuple[float, float],
        annotator_in_pipeline: BaseAnnotator,
        kinect_intrinsics: o3d.camera.PinholeCameraIntrinsic,
    ):
        annotator_in_pipeline.descriptor.parameters.global_with_depth = True
        cas = annotator_in_pipeline.get_cas()
        width, height = kinect_intrinsics.width, kinect_intrinsics.height
        scalex, scaley = scale_factor

        cas.set(
            robokudo.cas.CASViews.COLOR_IMAGE,
            np.zeros((height, width, 3), dtype=np.uint8),
        )
        cas.set(robokudo.cas.CASViews.COLOR2DEPTH_RATIO, (scalex, scaley))

        annotator_in_pipeline.cam_intrinsics = copy.deepcopy(kinect_intrinsics)

        image = get_color_image(annotator_in_pipeline)
        assert image.shape[:2] == (
            height * scaley,
            width * scalex,
        ), f"Image should be scaled to {scalex}, {scaley}"

    def test_get_color_image_no_depth(
        self,
        annotator_in_pipeline: BaseAnnotator,
        kinect_intrinsics: o3d.camera.PinholeCameraIntrinsic,
    ):
        annotator_in_pipeline.descriptor.parameters.global_with_depth = False
        cas = annotator_in_pipeline.get_cas()
        width, height = kinect_intrinsics.width, kinect_intrinsics.height
        image = np.zeros((height, width, 3), dtype=np.uint8)

        cas.set(robokudo.cas.CASViews.COLOR_IMAGE, image)

        resized_image = get_color_image(annotator_in_pipeline)

        assert np.array_equal(image, resized_image)

    def test_get_color_image_invalid_depth(
        self,
        annotator_in_pipeline: BaseAnnotator,
        kinect_intrinsics: o3d.camera.PinholeCameraIntrinsic,
    ):
        annotator_in_pipeline.descriptor.parameters.global_with_depth = True
        cas = annotator_in_pipeline.get_cas()
        width, height = kinect_intrinsics.width, kinect_intrinsics.height
        image = np.zeros((height, width, 3), dtype=np.uint8)

        cas.set(robokudo.cas.CASViews.COLOR_IMAGE, image)
        cas.set(robokudo.cas.CASViews.COLOR2DEPTH_RATIO, None)

        assert get_color_image(annotator_in_pipeline) == None

    @pytest.mark.parametrize(
        "scale_factor",
        [
            (1.0, 1.0),  # No scaling
            (2.0, 2.0),  # Upscaling
            (0.5, 0.5),  # Downscaling
            (3.0, 2.0),  # Non-uniform scaling
            (2.0, 3.0),  # Non-uniform scaling
        ],
    )
    def test_resize_mask(
        self,
        scale_factor: tuple[float, float],
        annotator_in_pipeline: BaseAnnotator,
        kinect_intrinsics: o3d.camera.PinholeCameraIntrinsic,
    ):
        annotator_in_pipeline.descriptor.parameters.global_with_depth = True
        cas = annotator_in_pipeline.get_cas()
        width, height = kinect_intrinsics.width, kinect_intrinsics.height
        scalex, scaley = scale_factor

        cas.set(robokudo.cas.CASViews.COLOR2DEPTH_RATIO, (scalex, scaley))
        mask = np.zeros((int(height * scaley), int(width * scalex), 3), dtype=np.uint8)
        mask = resize_mask(annotator_in_pipeline, mask)
        assert mask.shape[:2] == (
            1280,
            1024,
        ), f"Mask should be rescaled by scalex={scalex}, scaley={scaley}"

    def test_resize_mask_invalid_depth(self, annotator_in_pipeline: BaseAnnotator):
        annotator_in_pipeline.descriptor.parameters.global_with_depth = True

        cas = annotator_in_pipeline.get_cas()
        cas.set(robokudo.cas.CASViews.COLOR2DEPTH_RATIO, None)

        mask = np.zeros((640, 480, 3), dtype=np.uint8)
        assert pytest.raises(RuntimeError, resize_mask, annotator_in_pipeline, mask)

    def test_resize_mask_no_depth(
        self,
        annotator_in_pipeline: BaseAnnotator,
        kinect_intrinsics: o3d.camera.PinholeCameraIntrinsic,
    ):
        annotator_in_pipeline.descriptor.parameters.global_with_depth = False
        width, height = kinect_intrinsics.width, kinect_intrinsics.height
        mask = np.zeros((height, width, 3), dtype=np.uint8)

        resized_mask = resize_mask(annotator_in_pipeline, mask)

        assert np.array_equal(mask, resized_mask)

    def test_generate_source_name(self, annotator_in_pipeline: BaseAnnotator):
        name = generate_source_name(annotator_in_pipeline)
        assert name, f"source name should be non-empty"
        assert type(name) is str, f"source name should be a string"
