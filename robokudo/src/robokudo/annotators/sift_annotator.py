from __future__ import annotations

# %% Imports
import colorsys

import cv2
import numpy
import open3d
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING, Sequence

from robokudo.annotators.core import ThreadedAnnotator
from robokudo.cas import CASViews
from robokudo.types.annotation import SIFTAnnotation
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.error_handling import catch_and_raise_to_blackboard
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel

if TYPE_CHECKING:
    import numpy.typing as numpy_types

    from robokudo.types.cv import ColorToDepthRatio


# %% Feature extraction
class SIFTAnnotator(ThreadedAnnotator):
    """
    Annotator for SIFT feature extraction and matching using OpenCV.
    """

    def __init__(
        self,
        name: str = "SIFTAnnotator",
    ) -> None:
        super().__init__(name)

        self._feature_extractor: cv2.SIFT = cv2.SIFT_create()
        """
        SIFT feature extractor.
        """

    @catch_and_raise_to_blackboard
    def compute(self) -> Status:
        """
        Compute the SIFT features of the current image and match them to the last image.

        :return: The status of the computation.
        """
        cas = self.get_cas()

        object_hypotheses: list[ObjectHypothesis] = cas.filter_annotations_by_type(
            ObjectHypothesis
        )
        if len(object_hypotheses) == 0:
            return Status.FAILURE

        color_to_depth_ratio = cas.color_to_depth_ratio
        camera_model = cas.require_camera_observation().effective_camera_model

        color_image = cas.get_copy(CASViews.COLOR_IMAGE)
        depth_image = cas.get_copy(CASViews.DEPTH_IMAGE)
        grey_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        visualization_image = color_image.copy()

        all_keypoints: list[cv2.KeyPoint] = []
        for object_hypothesis in object_hypotheses:
            region_corners = object_hypothesis.roi.roi.get_corner_points()

            region_image = grey_image[
                region_corners[1] : region_corners[3],
                region_corners[0] : region_corners[2],
            ]
            if object_hypothesis.roi.mask.shape[:2] != region_image.shape[:2]:
                region_mask = object_hypothesis.roi.mask[
                    region_corners[1] : region_corners[3],
                    region_corners[0] : region_corners[2],
                ]
            else:
                region_mask = object_hypothesis.roi.mask

            keypoints, descriptors = self._feature_extractor.detectAndCompute(
                region_image, region_mask
            )

            for keypoint in keypoints:
                keypoint.pt = (
                    keypoint.pt[0] + region_corners[0],
                    keypoint.pt[1] + region_corners[1],
                )

            object_hypothesis.annotations.append(
                SIFTAnnotation.from_cv2(
                    keypoints=keypoints,
                    descriptors=descriptors,
                )
            )

            all_keypoints.extend(keypoints)

            visualization_image = self._draw_visualization(
                color_image,
                visualization_image,
                keypoints,
            )

        point_cloud = self._keypoints_to_point_cloud(
            all_keypoints, depth_image, camera_model, color_to_depth_ratio
        )
        self.get_annotator_output_struct().set_image(visualization_image)
        self.get_annotator_output_struct().set_geometries(point_cloud)
        return Status.SUCCESS

    def _draw_visualization(
        self,
        color_image: numpy_types.NDArray[numpy.uint8],
        visualization_image: numpy_types.NDArray[numpy.uint8],
        current_keypoints: Sequence[cv2.KeyPoint],
    ) -> numpy_types.NDArray[numpy.uint8]:
        """
        Add visualizations to the image.

        :param color_image: The current color image.
        :param visualization_image: The image to add visualizations to.
        :param current_keypoints: The keypoints from the current image.
        :return: The image with visualizations.
        """
        return cv2.drawKeypoints(
            color_image,
            current_keypoints,
            visualization_image,
            flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG
            | cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
        )

    def _keypoints_to_point_cloud(
        self,
        keypoints: list[cv2.KeyPoint],
        depth_image: numpy_types.NDArray,
        intrinsics: PinholeCameraModel,
        color_to_depth_ratio: ColorToDepthRatio,
        depth_scale: float = 1000.0,
        maximum_depth: float = 3.0,
    ) -> open3d.geometry.PointCloud:
        """
        Project the keypoints to a 3D point cloud.

        :param keypoints: The keypoints to project to 3D.
        :param depth_image: The depth image used for projection.
        :param intrinsics: The camera intrinsics used for projection.
        :param color_to_depth_ratio: Size of the depth image relative to the color
            image.
        :param depth_scale: The depth scale of the depth image.
        :param maximum_depth: The maximum depth to project.
        :return: The 3D point cloud.
        """
        intrinsic_matrix = intrinsics.intrinsic_matrix
        focal_length_x, focal_length_y = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        principal_point_x, principal_point_y = (
            intrinsic_matrix[0, 2],
            intrinsic_matrix[1, 2],
        )

        spatial_points = []
        responses = []
        for keypoint in keypoints:
            pixel_column, pixel_row = round(
                keypoint.pt[0] * color_to_depth_ratio.horizontal
            ), round(keypoint.pt[1] * color_to_depth_ratio.vertical)
            if not (
                0 <= pixel_row < depth_image.shape[0]
                and 0 <= pixel_column < depth_image.shape[1]
            ):
                continue

            depth = depth_image[pixel_row, pixel_column] / depth_scale
            if depth <= 0 or depth > maximum_depth:
                continue

            position_x = (pixel_column - principal_point_x) * depth / focal_length_x
            position_y = (pixel_row - principal_point_y) * depth / focal_length_y
            spatial_points.append([position_x, position_y, depth])
            responses.append(keypoint.response)

        colors = self._keypoint_colors(keypoints, numpy.array(responses))

        point_cloud = open3d.geometry.PointCloud()
        point_cloud.points = open3d.utility.Vector3dVector(
            numpy.array(spatial_points, dtype=numpy.float32)
        )
        point_cloud.colors = open3d.utility.Vector3dVector(colors)
        return point_cloud

    @staticmethod
    def _keypoint_colors(
        keypoints: list[cv2.KeyPoint],
        responses: numpy_types.NDArray,
    ) -> numpy_types.NDArray[numpy.float64]:
        """
        Get RGB colors for keypoints based on their responses and angles.

        :param keypoints: The keypoints.
        :param responses: The keypoints responses.
        :return: The RGB colors.
        """

        def _normalize(response_values: numpy_types.NDArray) -> numpy_types.NDArray:
            """
            Normalize responses to their observed range.
            """
            minimum_response, maximum_response = (
                response_values.min(),
                response_values.max(),
            )
            return (response_values - minimum_response) / (
                maximum_response - minimum_response + 1e-8
            )

        brightness_values = 0.4 + 0.6 * _normalize(responses)

        colors = []
        for keypoint, brightness in zip(keypoints, brightness_values):
            seed = hash(
                (
                    round(keypoint.pt[0], 1),
                    round(keypoint.pt[1], 1),
                    round(keypoint.size, 1),
                )
            )
            random_generator = numpy.random.default_rng(abs(seed) % (2**32))

            hue = random_generator.uniform(0.0, 1.0)
            red, green, blue = colorsys.hsv_to_rgb(hue, 1.0, brightness)
            colors.append([red, green, blue])

        return numpy.array(colors, dtype=numpy.float64)
