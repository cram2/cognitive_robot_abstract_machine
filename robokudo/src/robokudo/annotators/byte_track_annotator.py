import copy

import numpy as np
import open3d as o3d
import supervision as sv
from py_trees.common import Status
from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.types.annotation import Classification
from robokudo.types.scene import ObjectHypothesis
from supervision.detection.tools.transformers import append_class_names_to_data
from typing_extensions import TYPE_CHECKING, List

if TYPE_CHECKING:
    import numpy.typing as npt


class ByteTrackAnnotator(BaseAnnotator):
    """A ByteTrack annotator that tracks detected objects across pipeline ticks.

    The annotator is capable of utilizing:

    * ObjectHypothesis ROI
    * ObjectHypothesis ROI + Mask
    * ObjectHypothesis ROI + Mask + Classification annotation
    """

    def __init__(self, name: str = "ByteTrackAnnotator"):
        super().__init__(name)

        self.tracker: sv.ByteTrack = sv.ByteTrack()
        """Main ByteTrack instance for object instance tracking."""

        self.last_ts: float = 0.0
        """Timestamp of the last data received, used for tracker reset detection."""

        self.label_annotator = sv.RichLabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
        """A special label annotator, that cannot simply be used like the annotators in `self.vis_annotators`."""

        self.vis_annotators = [
            sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK),
            sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX),
            sv.TraceAnnotator(color_lookup=sv.ColorLookup.TRACK),
        ]
        """A list of simple supervision annotators to use for visualization. They will be applied in order."""

    def update(self) -> Status:
        cas = self.get_cas()

        data_ts = cas.get(CASViews.DATA_TIMESTAMP)
        if data_ts < self.last_ts:
            # This is expected when looping data
            self.rk_logger.debug(f"Time moved backward, resetting tracker.")
            self.tracker.reset()
        self.last_ts = data_ts

        ohs: List[ObjectHypothesis] = cas.filter_annotations_by_type(ObjectHypothesis)
        if len(ohs) == 0:
            self.rk_logger.debug(f"{len(ohs)} object hypothesis found, nothing to do.")
            return Status.SUCCESS

        rois_xyxy = np.array([oh.roi.roi.get_corner_points() for oh in ohs])
        if len(rois_xyxy) == 0:
            self.rk_logger.debug(f"No ROI found in {len(ohs)} object hypothesis.")
            return Status.FAILURE

        detections = {
            "xyxy": rois_xyxy,
            "confidence": np.ones(len(rois_xyxy), dtype=float),  # Default confidence
        }

        masks = [oh.roi.mask for oh in ohs]
        masks = list(filter(lambda m: m is not None, masks))
        if len(masks) == len(rois_xyxy):
            cam_intrinsic: o3d.cuda.pybind.camera.PinholeCameraIntrinsic = cas.get(
                CASViews.CAM_INTRINSIC
            )

            # Masks must be restored to full color image size
            restored_masks = []
            for roi, mask in zip(rois_xyxy, masks):
                restored_mask = np.zeros(
                    (cam_intrinsic.height, cam_intrinsic.width), dtype=np.uint8
                )
                restored_mask[roi[1] : roi[3], roi[0] : roi[2]] = mask
                restored_masks.append(restored_mask)
            detections["mask"] = np.array(restored_masks).astype(bool)

            self.rk_logger.debug(
                f"Using {len(restored_masks)} masks for object tracking."
            )

        classifications: List[Classification] = []
        for oh in ohs:
            classification = cas.filter_by_type(Classification, oh.annotations)
            if len(classification) == 0:
                break
            classifications.append(classification[0])
        if len(classifications) == len(rois_xyxy):
            id2label = {}
            class_ids = []
            class_names = []
            class_confidences = []

            for oh, cls in zip(ohs, classifications):
                id2label[cls.class_id] = cls.classname
                class_ids.append(cls.class_id)
                class_names.append(cls.classname)
                class_confidences.append(cls.confidence)

            detections["class_id"] = np.array(class_ids)
            detections["confidence"] = np.array(class_confidences)
            detections["data"] = append_class_names_to_data(class_ids, id2label, {})

            self.rk_logger.debug(
                f"Using {len(class_ids)} class ids for object tracking."
            )

        detections = sv.Detections(**detections)
        tracked_detections = self.tracker.update_with_detections(detections)

        self.visualize(tracked_detections)

        return Status.SUCCESS

    def visualize(self, tracked_detections: sv.Detections) -> None:
        """Visualize the tracked detections and set the resulting image in the annotator output struct.

        :param tracked_detections: The tracked detections to visualize.
        """
        vis_image: npt.NDArray[np.uint8] = copy.deepcopy(
            self.get_cas().get(CASViews.COLOR_IMAGE)
        )

        if (
            tracked_detections.tracker_id is not None
            and len(tracked_detections.tracker_id) > 0
        ):
            for annotator in self.vis_annotators:
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
