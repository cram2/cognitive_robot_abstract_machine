from __future__ import annotations

import colorsys
import copy

import cv2
import numpy as np
import open3d as o3d
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING, List, Sequence, Tuple

from robokudo.annotators.core import ThreadedAnnotator
from robokudo.cas import CASViews
from robokudo.types.annotation import SIFTAnnotation
from robokudo.types.scene import ObjectHypothesis

if TYPE_CHECKING:
    import numpy.typing as npt


class SIFTAnnotator(ThreadedAnnotator):
    """Annotator for SIFT feature extraction and matching using OpenCV."""

    def __init__(
        self,
        name: str = "SIFTAnnotator",
    ) -> None:
        super().__init__(name)

        self._sift: cv2.SIFT = cv2.SIFT_create()
        """SIFT feature extractor."""

    def compute(self) -> Status:
        """Compute the SIFT features of the current image and match them to the last image.

        :return: The status of the computation.
        """
        cas = self.get_cas()

        ohs: List[ObjectHypothesis] = cas.filter_annotations_by_type(ObjectHypothesis)
        if len(ohs) == 0:
            return Status.FAILURE

        intrinsic = cas.cam_intrinsic
        if intrinsic is None:
            return Status.FAILURE

        depth_ratio = cas.color2depth_ratio
        if depth_ratio is None:
            return Status.FAILURE

        color_image = cas.get_copy(CASViews.COLOR_IMAGE)
        depth_image = cas.get_copy(CASViews.DEPTH_IMAGE)
        grey_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        vis_image = color_image.copy()

        all_keypoints: List[cv2.KeyPoint] = []
        for i, oh in enumerate(ohs):
            roi = oh.roi.roi.get_corner_points()

            roi_image = grey_image[roi[1] : roi[3], roi[0] : roi[2]]
            if oh.roi.mask.shape[:2] != roi_image.shape[:2]:
                roi_mask = oh.roi.mask[roi[1] : roi[3], roi[0] : roi[2]]
            else:
                roi_mask = oh.roi.mask

            keypoints, descriptors = self._sift.detectAndCompute(roi_image, roi_mask)

            for k in keypoints:
                k.pt = (k.pt[0] + roi[0], k.pt[1] + roi[1])

            oh.annotations.append(
                SIFTAnnotation.from_cv2(
                    keypoints=keypoints,
                    descriptors=descriptors,
                )
            )

            all_keypoints.extend(keypoints)

            vis_image = self._draw_visualization(
                color_image,
                vis_image,
                keypoints,
            )

        pcd = self._keypoints_to_point_cloud(
            all_keypoints, depth_image, intrinsic, depth_ratio
        )
        self.get_annotator_output_struct().set_image(vis_image)
        self.get_annotator_output_struct().set_geometries(pcd)
        return Status.SUCCESS

    def _draw_visualization(
        self,
        color_image: npt.NDArray[np.uint8],
        vis_image: npt.NDArray[np.uint8],
        current_kp: Sequence[cv2.KeyPoint],
    ) -> npt.NDArray[np.uint8]:
        """Add visualizations to the image.

        :param color_image: The current color image.
        :param vis_image: The image to add visualizations to.
        :param current_kp: The keypoints from the current image.
        :return: The image with visualizations.
        """
        return cv2.drawKeypoints(
            color_image,
            current_kp,
            vis_image,
            flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG
            | cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
        )

    def _keypoints_to_point_cloud(
        self,
        keypoints: List[cv2.KeyPoint],
        depth_image: npt.NDArray,
        intrinsics: o3d.camera.PinholeCameraIntrinsic,
        depth_ratio: Tuple[float, float] = (1.0, 1.0),
        depth_scale: float = 1000.0,
        max_depth: float = 3.0,
    ) -> o3d.geometry.PointCloud:
        """Project the keypoints to a 3D point cloud.

        :param keypoints: The keypoints to project to 3D.
        :param depth_image: The depth image used for projection.
        :param intrinsics: The camera intrinsics used for projection.
        :param depth_ratio: The ratio of the depth image to the color image.
        :param depth_scale: The depth scale of the depth image.
        :param max_depth: The maximum depth to project.
        :return: The 3D point cloud.
        """
        intrinsic_mat = intrinsics.intrinsic_matrix
        fx, fy = intrinsic_mat[0, 0], intrinsic_mat[1, 1]
        cx, cy = intrinsic_mat[0, 2], intrinsic_mat[1, 2]

        points_3d = []
        responses = []
        for kp in keypoints:
            u, v = round(kp.pt[0] * depth_ratio[0]), round(kp.pt[1] * depth_ratio[1])
            if not (0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]):
                continue

            z = depth_image[v, u] / depth_scale
            if z <= 0 or z > max_depth:
                continue

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points_3d.append([x, y, z])
            responses.append(kp.response)

        colors = self._keypoint_colors(keypoints, np.array(responses))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points_3d, dtype=np.float32))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    @staticmethod
    def _keypoint_colors(
        keypoints: List[cv2.KeyPoint],
        responses: npt.NDArray,
    ) -> npt.NDArray[np.float64]:
        """Get rgb colors for keypoints based on their responses and angles.

        :param keypoints: The keypoints.
        :param responses: The keypoints responses.
        :return: The rgb colors.
        """

        def _normalize(arr: npt.NDArray) -> npt.NDArray:
            lo, hi = arr.min(), arr.max()
            return (arr - lo) / (hi - lo + 1e-8)

        values = 0.4 + 0.6 * _normalize(responses)

        rgb = []
        for kp, v in zip(keypoints, values):
            seed = hash((round(kp.pt[0], 1), round(kp.pt[1], 1), round(kp.size, 1)))
            rng: npt.NDArray[np.float32] = np.random.default_rng(abs(seed) % (2**32))

            hue = rng.uniform(0.0, 1.0)
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, v)
            rgb.append([r, g, b])

        return np.array(rgb, dtype=np.float64)
