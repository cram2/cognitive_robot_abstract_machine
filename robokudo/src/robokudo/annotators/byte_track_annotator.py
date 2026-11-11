from __future__ import annotations

import copy

# %% Imports
from dataclasses import dataclass, field

import numpy as np
import supervision as sv
from py_trees.common import Status
from supervision.config import CLASS_NAME_DATA_FIELD
from trackers import ByteTrackTracker
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.types.annotation import Classification
from robokudo.types.scene import ObjectHypothesis

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass(eq=False)
class ByteTrackAnnotator(BaseAnnotator):
    """
    A ByteTrack annotator that tracks detected objects across pipeline ticks.

    The annotator is capable of utilizing:

    * ObjectHypothesis ROI
    * ObjectHypothesis ROI + Mask
    * ObjectHypothesis ROI + Mask + Classification annotation
    """

    name: str = __qualname__
    """
    Name of this node in the behaviour tree.
    """

    tracker: ByteTrackTracker = field(init=False, repr=False)
    """
    Main ByteTrack instance for object instance tracking.
    """

    last_timestamp_nanoseconds: int = field(default=0, init=False, repr=False)
    """
    Acquisition time of the previous observation, used to detect backward time jumps.
    """

    label_annotator: sv.RichLabelAnnotator = field(init=False, repr=False)
    """
    A special label annotator, that cannot simply be used like the annotators in
    `self.visualization_annotators`.
    """

    visualization_annotators: list[
        sv.BoxAnnotator | sv.MaskAnnotator | sv.TraceAnnotator
    ] = field(init=False, repr=False)
    """
    A list of simple supervision annotators to use for visualization.

    They will be applied in order.
    """

    def __post_init__(self) -> None:
        """
        Initialize annotator resources after
        :meth:`robokudo.annotators.core.BaseAnnotator.__post_init__`.
        """
        super().__post_init__()

        self.tracker: ByteTrackTracker = ByteTrackTracker()

        self.label_annotator = sv.RichLabelAnnotator(color_lookup=sv.ColorLookup.TRACK)

        self.visualization_annotators = [
            sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK),
            sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX),
            sv.TraceAnnotator(color_lookup=sv.ColorLookup.TRACK),
        ]

    def update(self) -> Status:
        cas = self.get_cas()
        observation = cas.require_camera_observation()

        timestamp_nanoseconds = observation.timestamp_nanoseconds
        if timestamp_nanoseconds < self.last_timestamp_nanoseconds:
            # This is expected when looping data
            self.rk_logger.debug("Time moved backward, resetting tracker.")
            self.tracker.reset()
        self.last_timestamp_nanoseconds = timestamp_nanoseconds

        object_hypotheses: list[ObjectHypothesis] = cas.filter_annotations_by_type(
            ObjectHypothesis
        )
        if len(object_hypotheses) == 0:
            self.rk_logger.debug(
                f"{len(object_hypotheses)} object hypothesis found, nothing to do."
            )
            return Status.SUCCESS

        rois_xyxy = np.array(
            [
                object_hypothesis.roi.roi.get_corner_points()
                for object_hypothesis in object_hypotheses
            ]
        )
        if len(rois_xyxy) == 0:
            self.rk_logger.debug(
                f"No ROI found in {len(object_hypotheses)} object hypothesis."
            )
            return Status.FAILURE

        detections = {
            "xyxy": rois_xyxy,
            "confidence": np.ones(len(rois_xyxy), dtype=float),  # Default confidence
        }

        masks = [object_hypothesis.roi.mask for object_hypothesis in object_hypotheses]
        masks = list(filter(lambda m: m is not None, masks))
        if len(masks) == len(rois_xyxy):
            # Masks must be restored to full color image size
            restored_masks = []
            for roi, mask in zip(rois_xyxy, masks):
                restored_mask = np.zeros(
                    (
                        observation.resolution.height,
                        observation.resolution.width,
                    ),
                    dtype=np.uint8,
                )
                restored_mask[roi[1] : roi[3], roi[0] : roi[2]] = mask
                restored_masks.append(restored_mask)
            detections["mask"] = np.array(restored_masks).astype(bool)

            self.rk_logger.debug(
                f"Using {len(restored_masks)} masks for object tracking."
            )

        classifications: list[Classification] = []
        for object_hypothesis in object_hypotheses:
            classification = cas.filter_by_type(
                Classification, object_hypothesis.annotations
            )
            if len(classification) == 0:
                break
            classifications.append(classification[0])
        if len(classifications) == len(rois_xyxy):
            detections["class_id"] = np.zeros(len(object_hypotheses))
            detections["confidence"] = np.zeros(len(object_hypotheses))
            detections["data"][CLASS_NAME_DATA_FIELD] = np.full(
                len(object_hypotheses), "NONE", dtype=str
            )

            for i, (object_hypothesis, cls) in enumerate(
                zip(object_hypotheses, classifications)
            ):
                detections["class_id"][i] = cls.class_id
                detections["confidence"][i] = cls.confidence
                detections["data"][CLASS_NAME_DATA_FIELD] = cls.classname

            self.rk_logger.debug(
                f"Using {len(object_hypotheses)} class ids for object tracking."
            )

        detections = sv.Detections(**detections)
        tracked_detections = self.tracker.update(detections)

        self.visualize(tracked_detections)

        return Status.SUCCESS

    def visualize(self, tracked_detections: sv.Detections) -> None:
        """
        Visualize the tracked detections and set the resulting image in the annotator
        output struct.

        :param tracked_detections: The tracked detections to visualize.
        """
        vis_image: npt.NDArray[np.uint8] = copy.deepcopy(
            self.get_cas().get(CASViews.COLOR_IMAGE)
        )

        if (
            tracked_detections.tracker_id is not None
            and len(tracked_detections.tracker_id) > 0
        ):
            for annotator in self.visualization_annotators:
                vis_image = annotator.annotate(vis_image, tracked_detections)

            labels = []
            for i, track_id in enumerate(tracked_detections.tracker_id):
                label = f"{track_id}: "
                if tracked_detections.class_id is not None:
                    label += f"{tracked_detections['class_name'][i]}"
                else:
                    label += "unknown"
                label += f" ({tracked_detections.confidence[i]:.2f})"

                labels.append(label)
            vis_image = self.label_annotator.annotate(
                scene=vis_image, detections=tracked_detections, labels=labels
            )

        self.get_annotator_output_struct().set_image(vis_image)
