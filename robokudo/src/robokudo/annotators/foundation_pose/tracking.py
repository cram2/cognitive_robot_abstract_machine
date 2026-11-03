"""
Poses of a frame, kept to track the objects in the next frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy
from typing_extensions import TYPE_CHECKING

from robokudo.types.cv import Rect
from robokudo.utils.non_maxima_suppression import _iou

if TYPE_CHECKING:
    import numpy.typing as numpy_typing


# %% Posed objects


@dataclass(eq=False)
class PosedObject:
    """
    An object whose pose was determined in a frame.

    Posed objects are equal only to themselves.
    """

    class_name: str
    """
    Class name of the object's CAD model.
    """

    pose: numpy_typing.NDArray[numpy.float64]
    """
    Pose of the CAD model in the camera frame, as a 4x4 homogeneous transform.
    """

    mask_region: Rect
    """
    The smallest rectangle containing the object's mask, in depth image pixels.
    """


# %% Matching across frames


@dataclass
class PreviousFramePoses:
    """
    Objects posed in the previous frame that are not yet matched to an object of the
    current frame.
    """

    unmatched_objects: list[PosedObject] = field(default_factory=list)
    """
    Posed objects still available for matching.
    """

    def take_match(
        self, class_name: str, mask_region: Rect, minimum_overlap: float
    ) -> numpy_typing.NDArray[numpy.float64] | None:
        """
        Remove and return the pose of the unmatched object of a class whose mask region
        overlaps most with a mask region of the current frame.

        :param class_name: Class name of the object of the current frame.
        :param mask_region: Mask region of the object of the current frame.
        :param minimum_overlap: Intersection over union below which mask regions do not
            match.
        :return: The pose of the matching object, or ``None`` if none overlaps enough.
        """
        candidates = [
            posed_object
            for posed_object in self.unmatched_objects
            if posed_object.class_name == class_name
        ]
        if not candidates:
            return None
        overlaps = [
            _iou(
                candidate.mask_region.get_corner_points(),
                mask_region.get_corner_points(),
            )
            for candidate in candidates
        ]
        best_overlap = max(overlaps)
        if best_overlap < minimum_overlap:
            return None
        match = candidates[overlaps.index(best_overlap)]
        self.unmatched_objects.remove(match)
        return match.pose
