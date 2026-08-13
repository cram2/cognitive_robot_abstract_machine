from __future__ import annotations

import copy

import cv2
import torch
from py_trees.common import Status
from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.types.annotation import Classification
from robokudo.types.cv import ImageROI
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.comparators import RoiComparator
from typing_extensions import TYPE_CHECKING, List
from ultralytics import YOLO

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


class SimpleYoloAnnotator(BaseAnnotator):

    def __init__(self, name: str = "SimpleYoloAnnotator") -> None:
        super().__init__(name=name)

        self.model = YOLO("yolov8n_7.pt")
        """The YOLO model instance."""

        self.id2name = self.model.names
        """The YOLO models id to name map."""

        self.roi_comparator = RoiComparator(1.0)
        """A comparator for ROIs used to associate YOLO and RoboKudo ROIs."""

    def update(self) -> Status:
        """Run YOLO inference on the cas color image and combine classifications with existing object hypotheses.

        Runs objects detection on the image and uses the RoiComparator find the closest bounding box created for
        an ObjectHypothesis by previous annotator. If there is a similar ROI present in an ObjectHypothesis,
        a Classification is attached to it.
        """
        visualization_img = copy.deepcopy(self.get_cas().get(CASViews.COLOR_IMAGE))

        ohs: List[ObjectHypothesis] = self.get_cas().filter_annotations_by_type(
            ObjectHypothesis
        )

        with torch.no_grad():
            result_tensor = self.model(visualization_img, conf=0.9)[0]

        result_np = result_tensor.boxes.cpu().numpy()
        object_hypotheses = []
        for obj_id, result in enumerate(result_np):
            bbox = result.xyxy[0]

            cls = result.cls[0]
            name = self.id2name[cls]

            classification = Classification()
            classification.source = self.name
            classification.classname = name
            classification.confidence = result.conf[0]

            roi = ImageROI()
            roi.roi.pos.x = int(bbox[0])
            roi.roi.pos.y = int(bbox[1])
            roi.roi.width = int(bbox[2] - bbox[0])
            roi.roi.height = int(bbox[3] - bbox[1])

            # Attempt to attach the classification to an existing object hypothesis
            best_oh = None
            best_conf = -float("inf")
            for oh in ohs:
                conf = self.roi_comparator.compute_similarity(oh.roi.roi, roi.roi)

                if conf > 0.65 and conf > best_conf:
                    best_oh = oh
                    best_conf = conf

            if best_oh is None:
                object_hypothesis = ObjectHypothesis()
                object_hypothesis.source = self.name
                object_hypothesis.id = str(obj_id)
                object_hypothesis.roi = roi

                object_hypotheses.append(object_hypothesis)
            else:
                object_hypothesis = best_oh

            object_hypothesis.annotations.append(classification)
            visualization_img = self.add_to_image(
                object_hypothesis, classification, visualization_img
            )

        self.get_cas().annotations.extend(object_hypotheses)
        self.get_annotator_output_struct().set_image(visualization_img)

        return Status.SUCCESS

    @staticmethod
    def add_to_image(
        obj: ObjectHypothesis,
        classification: Classification,
        image: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.uint8]:
        """Add the object hypothesis along with the classification name and confidence to the visualization image."""
        x1, y1, x2, y2 = (
            obj.roi.roi.pos.x,
            obj.roi.roi.pos.y,
            obj.roi.roi.pos.x + obj.roi.roi.width,
            obj.roi.roi.pos.y + obj.roi.roi.height,
        )

        vis_text = f"{classification.classname}, {classification.confidence:.2f}"
        font = cv2.FONT_HERSHEY_COMPLEX
        image = cv2.putText(image, vis_text, (x1, y1 - 5), font, 0.5, (0, 0, 255), 1, 2)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        return image
