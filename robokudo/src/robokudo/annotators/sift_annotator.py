from __future__ import annotations

import colorsys
import copy
from enum import Enum
from itertools import chain

import cv2
import numpy as np
import open3d as o3d
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from robokudo.annotators.core import ThreadedAnnotator
from robokudo.cas import CASViews
from robokudo.types.annotation import SIFTAnnotation
from robokudo.types.scene import ObjectHypothesis

if TYPE_CHECKING:
    import numpy.typing as npt


class SIFTAnnotatorVisualizationModes(Enum):
    """Enum for SIFTAnnotator visualization modes."""

    KEYPOINTS = "keypoints"
    MATCHES = "matches"


class SIFTAnnotatorMode(Enum):
    """Enum for SIFTAnnotator detection modes."""

    OBJECT = "object"
    IMAGE = "image"


class SIFTAnnotator(ThreadedAnnotator):
    """Annotator for SIFT feature extraction and matching using OpenCV."""

    class Descriptor(ThreadedAnnotator.Descriptor):
        """Descriptor for SIFTAnnotator."""

        class Parameters(ThreadedAnnotator.Descriptor.Parameters):
            """Parameters for SIFTAnnotator."""

            def __init__(self) -> None:
                self.visualization_mode: SIFTAnnotatorVisualizationModes = (
                    SIFTAnnotatorVisualizationModes.MATCHES
                )
                """The visualization mode for the SIFTAnnotator.
                
                * Matches will show the keypoint matches between the last and current image.
                * Keypoints will show the keypoints of the current image.
                """

                self.detection_mode = SIFTAnnotatorMode.OBJECT
                """The detection mode for the SIFTAnnotator.
                
                * Object mode will run SIFT detection and matching on the masks of oh only and try to match them.
                * Image mode will run SIFT detection and matching on the full color image.
                """

                self.match_distance_threshold: float = 0.75
                """Distance threshold for Lowe's ratio test."""

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "SIFTAnnotator",
        descriptor: "SIFTAnnotator.Descriptor" = Descriptor(),
    ) -> None:
        super().__init__(name, descriptor)

        self._sift: cv2.SIFT = cv2.SIFT_create()
        """SIFT feature extractor."""

        self._matcher: cv2.BFMatcher = cv2.BFMatcher()
        """SIFT feature matcher."""

        self._last_image: Optional[npt.NDArray[np.uint8]] = None
        """The last image processed by the annotator."""

        self._last_keypoints: Sequence[cv2.KeyPoint] = []
        """The keypoints from the last image."""

        self._last_descriptors: npt.NDArray[np.float32] = np.array([])
        """The descriptors from the last image."""

        self._last_object_keypoints: Dict[int, Sequence[cv2.KeyPoint]] = {}
        """The per-object keypoints of the last image."""

        self._last_object_descriptors: Dict[int, npt.NDArray[np.float32]] = {}
        """The per-object descriptors of the last image."""

    def compute_matches(
        self,
        last_descriptors: npt.NDArray[np.float32],
        current_descriptors: npt.NDArray[np.float32],
    ) -> List[cv2.DMatch]:
        """Compute the matches between the last and current descriptors.

        :param last_descriptors: The descriptors from the last image.
        :param current_descriptors: The descriptors from the current image.
        :return: The list of matches.
        """
        distance_threshold = self.descriptor.parameters.match_distance_threshold
        all_matches = self._matcher.knnMatch(last_descriptors, current_descriptors, k=2)
        matches = [
            m for m, n in all_matches if m.distance < distance_threshold * n.distance
        ]
        return matches

    def compute(self) -> Status:
        """Compute the SIFT features of the current image and match them to the last image.

        :return: The status of the computation.
        """
        cas = self.get_cas()
        if cas is None:
            return Status.FAILURE

        ohs: List[ObjectHypothesis] = cas.filter_annotations_by_type(ObjectHypothesis)
        if len(ohs) == 0:
            return Status.FAILURE

        color_image = copy.deepcopy(self.get_cas().get(CASViews.COLOR_IMAGE))
        depth_image = copy.deepcopy(self.get_cas().get(CASViews.DEPTH_IMAGE))
        grey_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        intrinsic: o3d.camera.PinholeCameraIntrinsic = cas.get(CASViews.CAM_INTRINSIC)
        depth_ratio = cas.get(CASViews.COLOR2DEPTH_RATIO)

        mode = self.descriptor.parameters.detection_mode
        vis_mode = self.descriptor.parameters.visualization_mode

        vis_image = self._get_vis_image(color_image, vis_mode)

        if mode == SIFTAnnotatorMode.IMAGE:
            all_keypoints, descriptors = self._sift.detectAndCompute(grey_image, None)

            matches: List[cv2.DMatch] = []
            if self._last_image is not None:
                matches = self.compute_matches(self._last_descriptors, descriptors)
                self.rk_logger.debug(f"Found {len(matches)} matches")

            vis_image = self._draw_visualization(
                color_image,
                vis_image,
                vis_mode,
                self._last_keypoints,
                all_keypoints,
                matches,
            )

            # Full image SIFT features
            sift_annotation = SIFTAnnotation(
                keypoints=all_keypoints,
                descriptors=descriptors,
            )
            cas.annotations.append(sift_annotation)

            self._last_keypoints = all_keypoints
            self._last_descriptors = descriptors
        elif mode == SIFTAnnotatorMode.OBJECT:
            match_map: Dict[int, Tuple[Optional[int], float]] = {}

            all_best_matches: List[cv2.DMatch] = []
            object_keypoints: Dict[int, Sequence[cv2.KeyPoint]] = {}
            object_descriptors: Dict[int, npt.NDArray[np.float32]] = {}
            for i, oh in enumerate(ohs):
                roi = oh.roi.roi.get_corner_points()

                # Get ROI image and mask
                roi_image = grey_image[roi[1] : roi[3], roi[0] : roi[2]]
                if oh.roi.mask.shape[:2] != roi_image.shape[:2]:
                    roi_mask = oh.roi.mask[roi[1] : roi[3], roi[0] : roi[2]]
                else:
                    roi_mask = oh.roi.mask

                keypoints, descriptors = self._sift.detectAndCompute(
                    roi_image, roi_mask
                )

                # Correct keypoint coordinates to full image
                for k in keypoints:
                    k.pt = (k.pt[0] + roi[0], k.pt[1] + roi[1])

                sift_annotation = SIFTAnnotation(
                    keypoints=keypoints,
                    descriptors=descriptors,
                )
                oh.annotations.append(sift_annotation)

                object_keypoints[i] = keypoints
                object_descriptors[i] = descriptors

                if self._last_image is None:
                    continue

                # Find the object from the last frame with the highest match ratio
                best_score = 0.0
                best_match_id = None
                best_matches: List[cv2.DMatch] = []
                for object_id, d in self._last_object_descriptors.items():
                    matches = self.compute_matches(d, descriptors)
                    score = len(matches) / max(len(descriptors), 1.0)
                    if score > best_score:
                        best_score = score
                        best_match_id = object_id
                        best_matches = matches

                if best_match_id is None:
                    continue

                vis_image = self._draw_visualization(
                    color_image,
                    vis_image,
                    vis_mode,
                    self._last_object_keypoints[best_match_id],
                    keypoints,
                    best_matches,
                )

                match_map[i] = (best_match_id, best_score)
                all_best_matches.extend(best_matches)

            match_string = [
                f"\n{o1}->{match[0]} ({match[1]:.2f})"
                for o1, match in match_map.items()
                if match is not None
            ]
            self.rk_logger.debug(f"Matched objects:{''.join(match_string)}")

            all_keypoints = list(chain.from_iterable(object_keypoints.values()))
            self._last_object_keypoints = object_keypoints
            self._last_object_descriptors = object_descriptors
        else:
            self.rk_logger.warning(f"Unknown SIFTAnnotator mode: {mode}")
            return Status.FAILURE

        self._last_image = color_image

        pcd = self._keypoints_to_point_cloud(
            all_keypoints, depth_image, intrinsic, depth_ratio
        )
        self.get_annotator_output_struct().set_image(vis_image)
        self.get_annotator_output_struct().set_geometries(pcd)
        return Status.SUCCESS

    def _get_vis_image(
        self, color_image: npt.NDArray[np.uint8], vis_mode: SIFTAnnotatorMode
    ) -> npt.NDArray[np.uint8]:
        """Get the base visualization image for the visualization mode.

        :param color_image: The current color image.
        :param vis_mode: The visualization mode.
        :return: The base visualization image.
        """
        if vis_mode == SIFTAnnotatorVisualizationModes.MATCHES:
            if self._last_image is not None:
                return np.concatenate([self._last_image, color_image], axis=1)
            else:
                return np.concatenate([np.zeros_like(color_image), color_image], axis=1)
        elif vis_mode == SIFTAnnotatorVisualizationModes.KEYPOINTS:
            return color_image
        else:
            self.rk_logger.warning(
                f"Unknown visualization mode: {self.descriptor.parameters.visualization_mode}"
            )
            return color_image

    def _draw_visualization(
        self,
        color_image: npt.NDArray[np.uint8],
        vis_image: npt.NDArray[np.uint8],
        vis_mode: SIFTAnnotatorMode,
        last_kp: Sequence[cv2.KeyPoint],
        current_kp: Sequence[cv2.KeyPoint],
        matches: Sequence[cv2.DMatch],
    ) -> npt.NDArray[np.uint8]:
        """Add visualizations to the image.

        :param color_image: The current color image.
        :param vis_image: The image to add visualizations to.
        :param last_kp: The keypoints from the last image.
        :param current_kp: The keypoints from the current image.
        :param matches: The matches between the keypoints of last and current image.
        :return: The image with visualizations.
        """
        if vis_mode == SIFTAnnotatorVisualizationModes.MATCHES:
            if len(matches) == 0:
                return color_image
            return cv2.drawMatches(
                self._last_image,
                last_kp,
                color_image,
                current_kp,
                matches,
                vis_image,
                matchColor=(0, 255, 0),
                singlePointColor=(255, 0, 0),
                flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG
                | cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
            )
        elif vis_mode == SIFTAnnotatorVisualizationModes.KEYPOINTS:
            return cv2.drawKeypoints(
                color_image,
                current_kp,
                vis_image,
                flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG
                | cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
            )
        return vis_image

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
