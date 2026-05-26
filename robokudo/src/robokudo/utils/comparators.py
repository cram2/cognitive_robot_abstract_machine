from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from typing_extensions import TYPE_CHECKING, List, Any, Union, Iterable

from robokudo.types.annotation import PositionAnnotation
from robokudo.types.tf import Pose
from robokudo.utils.non_maxima_suppression import _iou

if TYPE_CHECKING:
    import numpy.typing as npt
    from robokudo.types.annotation import (
        BoundingBox3DAnnotation,
        Classification,
        ColorHistogram,
        SemanticColor,
    )
    from robokudo.types.cv import Rect, ImageROI


class FeatureComparator:
    """Base class for feature comparators."""

    def __init__(self, weight: float):
        self.weight = weight
        """Weight of this comparator in the final similarity score."""

    def compute_similarity(self, query_value: Any, obj_value: Any) -> float:
        """Computes similarity between query and object values.

        :param query_value: The translation to use as a baseline for comparison.
        :param obj_value: The value to compare against query_value.
        :returns: A similarity score between 0.0 (no similarity at all) and 1.0 (completely identical).
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class TranslationComparator(FeatureComparator):
    """Extended FeatureComparator that computes similarity based on translation distance between query and object values."""

    def __init__(self, weight: float, max_distance: float = 1.0):
        super().__init__(weight)

        self.max_distance = max_distance
        """Maximum distance between query and object values for which the similarity is 0.0."""

    def compute_similarity(
        self,
        query_value: Union[Iterable[float], PositionAnnotation],
        obj_value: Union[Iterable[float], PositionAnnotation],
    ) -> float:
        """Computes similarity based on translation distance between query and object values.

        :param query_value: The translation to use as a baseline for comparison.
        :param obj_value: The translation to compare against `query_value`.
        :returns: A similarity score between 0.0 (distance equal to or larger than `self.max_distance`) and 1.0 (completely identical).
        """
        if isinstance(query_value, PositionAnnotation):
            query_translation = query_value.translation
        else:
            query_translation = query_value

        if isinstance(obj_value, PositionAnnotation):
            obj_translation = obj_value.translation
        else:
            obj_translation = obj_value

        distance = euclidean(query_translation, obj_translation)
        return max(min(1.0 - (distance / self.max_distance), 1.0), 0.0)


class OrientationComparator(FeatureComparator):
    """Extended `FeatureComparator` that computes orientation similarity by dot product."""

    def compute_similarity(
        self,
        query_value: Union[Iterable[float], npt.NDArray[np.float64]],
        obj_value: Union[Iterable[float], npt.NDArray[np.float64]],
    ) -> float:
        """Computes similarity of both orientation quaternions by dot product.

        :param query_value: The quaternion to use as a baseline for comparison.
        :param obj_value: The quaternion to compare against `query_value`.
        :returns: A similarity score between 0.0 (angle=180d) and 1.0 (angle=0d).
        """
        q1 = query_value / np.linalg.norm(query_value)
        q2 = obj_value / np.linalg.norm(obj_value)

        dot = np.dot(q1, q2)
        dot = np.clip(dot, -1.0, 1.0)
        similarity = (1.0 + dot) / 2.0
        return similarity


class BboxComparator(FeatureComparator):
    """Extended FeatureComparator that computes similarity based on the bounding box size difference between query and object values."""

    def compute_similarity(
        self, query_value: BoundingBox3DAnnotation, obj_value: BoundingBox3DAnnotation
    ) -> float:
        """Computes similarity between bounding boxes based on their sizes.

        :param query_value: The bounding box to use as a baseline for comparison.
        :param obj_value: The bounding box to compare against `query_value`.
        :returns: A similarity score between 0.0 (`obj_value` is 0% the size of `query_value`) and 1.0 (identical bounding box sizes).
        """
        sorted_obj_size = np.sort(
            [obj_value.x_length, obj_value.y_length, obj_value.z_length]
        )
        sorted_query_size = np.sort(
            [query_value.x_length, query_value.y_length, query_value.z_length]
        )
        size_diff = np.abs(sorted_obj_size - sorted_query_size).sum()
        return 1.0 / (1.0 + size_diff)


class SizeComparator(FeatureComparator):
    """Extended FeatureComparator that computes similarity based on the sum difference between query and object values."""

    def compute_similarity(self, query_value: List, obj_value: List) -> float:
        """Computes similarity of two lists based on their sum differences.

        :param query_value: The list to use as a baseline for comparison.
        :param obj_value: The list to compare against `query_value`.
        :returns: A similarity score between 0.0 (sum of `obj_value` is 100% smaller or larger than sum of `query_value`) and 1.0 (identical sums).
        """
        sorted_obj_size = np.sort(query_value)
        sorted_query_size = np.sort(obj_value)
        size_diff = np.abs(sorted_obj_size - sorted_query_size).sum()
        return 1.0 / (1.0 + size_diff)


class HistogramComparator(FeatureComparator):
    """Extended FeatureComparator that computes similarity based on the cv2.HISTCMP_CORREL comparison between query and object values."""

    def compute_similarity(
        self, query_value: ColorHistogram, obj_value: ColorHistogram
    ) -> float:
        """Computes similarity between ColorHistogram objects based on the cv2.HISTCMP_CORREL metric.

        :param query_value: The histogram to use as a baseline for comparison.
        :param obj_value: The histogram to compare against `query_value`.
        :returns: A similarity score between 0.0 (completely different) and 1.0 (identical).
        """
        return abs(
            cv2.compareHist(query_value.hist, obj_value.hist, cv2.HISTCMP_CORREL)
        )


class SemanticColorComparator(FeatureComparator):
    """Extended `FeatureComparator` that computes similarity based on string similarity and color ratio difference between query and object values."""

    def compute_similarity(
        self, query_value: SemanticColor, obj_value: SemanticColor
    ) -> float:
        """Computes similarity between SemanticColor objects based on the cv2.HISTCMP_CORREL metric.

        :param query_value: The semantic color annotation to use as a baseline for comparison.
        :param obj_value: The semantic color annotation to compare against `query_value`.
        :returns: A similarity score between 0.0 (completely different colors or color ratio) and 1.0 (identical color and color ratio).
        """
        same_color = query_value.color == obj_value.color
        if not same_color:
            return 0.0
        return 1.0 - abs(query_value.ratio - obj_value.ratio)


class ClassificationComparator(FeatureComparator):
    """Extended `FeatureComparator` that computes similarity based on type, classname and confidence."""

    def compute_similarity(
        self, query_value: Classification, obj_value: Classification
    ) -> float:
        """Computes similarity between SemanticColor objects based on the cv2.HISTCMP_CORREL metric.

        :param query_value: The semantic color annotation to use as a baseline for comparison.
        :param obj_value: The semantic color annotation to compare against `query_value`.
        :returns: A similarity score between 0.0 (completely different colors or color ratio) and 1.0 (identical color and color ratio).
        """

        # Classification type is not comparable at all
        if query_value.classification_type != obj_value.classification_type:
            return 0.0

        # Class names are completely different
        if query_value.classname != obj_value.classname:
            return 0.0

        return 1.0 - abs(query_value.confidence - obj_value.confidence)


class AdditionalDataComparator(FeatureComparator):
    """Extended `FeatureComparator` that computes similarity based on numerical difference or simple value comparison between query and object values."""

    def compute_similarity(self, query_value: Any, obj_value: Any) -> float:
        """Computes similarity between any value or object based on numerical difference or value equality.

        If `query_value` and `obj_value` are numerical this FeatureComparator will normalize their numerical difference.
        Otherwise a simple equality check is performed, 0.0 is returned if they are not equal, 1.0 if they are equal.

        :param query_value: The object or value to use as a baseline for comparison.
        :param obj_value: The object or value to compare against `query_value`.
        :returns: A similarity score between 0.0 (no similarity) and 1.0 (identical values).
        """
        # Assuming simple equality check for additional data. Modify if needed.
        if isinstance(query_value, (int, float)) and isinstance(
            obj_value, (int, float)
        ):
            return 1.0 / (
                1.0 + abs(query_value - obj_value)
            )  # Normalize numerical difference
        return 1.0 if query_value == obj_value else 0.0


class RoiComparator(FeatureComparator):
    """Extended `FeatureComparator` that computes similarity based on overlap percentage between query and object values."""

    def compute_similarity(self, query_value: Rect, obj_value: Rect) -> float:
        """Computes the similarity of two Region of Interests by calculating their IoU.

        :param query_value: The rectangle to use as a baseline for comparison.
        :param obj_value: The rectangle to compare against `query_value`.
        :returns: A similarity score between 0.0 (no overlap) and 1.0 (100% overlap).
        """
        return _iou(query_value.get_corner_points(), obj_value.get_corner_points())


class MaskComparator(FeatureComparator):
    """Extended `FeatureComparator` that computes the IoU of two masks."""

    def compute_similarity(
        self, query_value: npt.NDArray, obj_value: npt.NDArray
    ) -> float:
        """Computes the similarity of two masks by calculating their IoU.

        :param query_value: The mask to use as a baseline for comparison.
        :param obj_value: The mask to compare against `query_value`.
        :returns: A similarity score between 0.0 (no overlap) and 1.0 (100% overlap).
        """
        intersection = np.logical_and(query_value, obj_value)
        if np.any(intersection):
            union = np.logical_or(query_value, obj_value)
            mask_iou = np.sum(intersection) / np.sum(union)
            return mask_iou
        else:
            return 0.0


class ImageROIComparator(FeatureComparator):
    """Extended `FeatureComparator` that computes the IoU of two images ROIs."""

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight)

        self.roi_comparator = RoiComparator(weight=1.0)
        """Comparator to compute the IoU of two images ROIs."""

        self.mask_comparator = MaskComparator(weight=1.0)
        """Comparator to compute the IoU of two image masks."""

    def compute_similarity(self, query_value: ImageROI, obj_value: ImageROI) -> float:
        """Computes the similarity of two ImageROI annotations by calculating the average IoU of both bounding boxes and masks.

        :param query_value: The ImageROI to use as a baseline for comparison.
        :param obj_value: The ImageROI to compare against `query_value`.
        :returns: A similarity score between 0.0 (no overlap) and 1.0 (100% overlap).
        """
        query_roi = query_value.roi
        obj_roi = obj_value.roi

        roi_sim = self.roi_comparator.compute_similarity(query_roi, obj_roi)

        query_mask = query_value.mask
        obj_mask = obj_value.mask

        # Crop masks to roi if needed
        if query_mask.shape[:2] != (query_roi.height, query_roi.width):
            xyxy = query_roi.get_corner_points()
            query_mask = query_mask[xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        if obj_mask.shape[:2] != (obj_roi.height, obj_roi.width):
            xyxy = obj_roi.get_corner_points()
            obj_mask = obj_mask[xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]

        # Pad the masks to be the same shape, fill new areas with zeros
        target_shape = (
            max(query_mask.shape[0], obj_mask.shape[0]),
            max(query_mask.shape[1], obj_mask.shape[1]),
        )
        padded_query = np.pad(
            query_mask,
            (
                (0, target_shape[0] - query_mask.shape[0]),
                (0, target_shape[1] - query_mask.shape[1]),
            ),
            mode="constant",
            constant_values=0,
        )

        padded_obj = np.pad(
            obj_mask,
            (
                (0, target_shape[0] - obj_mask.shape[0]),
                (0, target_shape[1] - obj_mask.shape[1]),
            ),
            mode="constant",
            constant_values=0,
        )

        mask_sim = self.mask_comparator.compute_similarity(padded_query, padded_obj)
        return (roi_sim + mask_sim) / 2.0


class PoseComparator(FeatureComparator):
    """Extended `FeatureComparator` that computes the similarity of two poses."""

    def __init__(self, weight: float = 1.0, max_distance: float = 1.0) -> None:
        super().__init__(weight)

        self.translation_comparator = TranslationComparator(
            weight=1.0, max_distance=max_distance
        )
        """Comparator to compute the similarity of two positions."""

        self.orientation_comparator = OrientationComparator(weight=1.0)
        """Comparator to compute the similarity of two orientations."""

    def compute_similarity(self, query_value: Pose, obj_value: Pose) -> float:
        """Computes the similarity of two poses by comparing translation and orientation values.

        :param query_value: The Pose to use as a baseline for comparison.
        :param obj_value: The Pose to compare against `query_value`.
        :returns: A similarity score between 0.0 (no similarity at all) and 1.0 (completely identical).
        """
        position_sim = self.translation_comparator.compute_similarity(
            query_value.translation, obj_value.translation
        )

        pose_sim = self.translation_comparator.compute_similarity(
            query_value.rotation, obj_value.rotation
        )

        return (position_sim + pose_sim) / 2.0
