"""
Object detection with YOLO models.

Every detection becomes an object hypothesis, or a human hypothesis for the ``person``
class, that carries the detected class as a classification. Optionally:

* *Precision mode* refines each hypothesis mask with the Segment Anything Model (SAM)
  instead of filling the whole bounding box.
* Existing object hypotheses that overlap a detection are classified instead of being
  duplicated by a new hypothesis.

.. note::
   The models need the packages of RoboKudo's optional ``dev`` extra.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from krrood.ormatic.utils import classproperty
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING, Any
from ultralytics import SAM, YOLO
from ultralytics.utils.plotting import colors

from robokudo.annotators.core import ThreadedAnnotator
from robokudo.cas import CASViews
from robokudo.defs import PACKAGE_NAME
from robokudo.exceptions import DetectionModelNotConfigured
from robokudo.types.annotation import Classification
from robokudo.types.cv import Rect
from robokudo.types.scene import HumanHypothesis, ObjectHypothesis
from robokudo.utils.annotator_helper import (
    draw_bounding_boxes_from_object_hypotheses,
    resize_mask,
    scale_camera_intrinsics,
)
from robokudo.utils.comparators import RoiComparator
from robokudo.utils.cv_helper import crop_image, get_scaled_color_image_for_depth_image
from robokudo.utils.decorators import timer_decorator
from robokudo.utils.file_loader import FileLoader
from robokudo.utils.module_loader import ModuleLoader
from robokudo.utils.o3d_helper import get_cloud_from_rgb_depth_and_mask

if TYPE_CHECKING:
    import numpy.typing as npt

# %% detections


@dataclass
class YoloDetection:
    """
    An object the YOLO model detected in the detection image.
    """

    classification: Classification
    """
    The detected class and the confidence of the detection.
    """

    bounding_box: Rect
    """
    The bounding box of the detected object, in detection image pixels.
    """


@dataclass
class AnnotatedDetection:
    """
    A detection together with the object hypothesis it annotates.
    """

    detection: YoloDetection
    """
    The detection whose classification was added to the hypothesis.
    """

    hypothesis: ObjectHypothesis
    """
    The object hypothesis the detection annotates.
    """

    is_new: bool
    """
    Whether this annotator created the hypothesis, rather than an earlier one.
    """


# %% annotator


class YoloAnnotator(ThreadedAnnotator):
    """
    Detects objects with a YOLO model and annotates them as classified hypotheses.
    """

    class Descriptor(ThreadedAnnotator.Descriptor):
        """
        Descriptor for the YoloAnnotator.
        """

        class Parameters:
            """
            Parameters for the YoloAnnotator.
            """

            def __init__(self) -> None:
                self.models_directory: Path = (
                    ModuleLoader.get_module_path(PACKAGE_NAME) / "models"
                )
                """
                Directory the pretrained models are loaded from, and downloaded into on
                first use.

                Created during setup if it does not exist.
                """
                self.pretrained_model: str | None = None
                """
                Name of a model ultralytics provides, e.g. ``yolov8m.pt``, inside
                :attr:`models_directory`.

                Takes precedence over weights inside a ROS package.
                """
                self.ros_package_name: str | None = None
                """
                Name of the ROS package holding custom weights.
                """
                self.weights_path: str | None = None
                """
                Path of the custom weights, relative to the ROS package.
                """
                self.class_names_path: str = "data/json/id2name.json"
                """
                Path of a JSON file mapping class ids to class names, relative to the
                ROS package.

                Only read for custom weights, since pretrained models carry their class
                names.
                """
                self.threshold: float = 0.9
                """
                Minimum confidence of a detection to be annotated.
                """
                self.person_class_name: str = "person"
                """
                Class name of the detections that become human hypotheses.
                """
                self.ignored_class_names: list[str] = []
                """
                Classes whose detections are neither annotated nor visualized.
                """
                self.precision_mode: bool = False
                """
                Whether SAM segments each detection instead of its mask filling the
                bounding box.
                """
                self.segmentation_model: str = "mobile_sam.pt"
                """
                Name of the SAM model ultralytics provides, inside
                :attr:`models_directory`, loaded in precision mode.
                """
                self.precision_mode_apply_erosion: bool = False
                """
                Whether to erode the SAM masks in precision mode.
                """
                self.precision_mode_erosion_kernel_size: int = 2
                """
                Edge length in pixels of the square kernel the SAM masks are eroded
                with.
                """
                self.depth_truncate: float = 3.5
                """
                Distance in meters beyond which depth measurements are left out of the
                hypothesis point clouds.
                """
                self.classify_existing_hypotheses: bool = False
                """
                Whether a detection overlapping an existing object hypothesis classifies
                that hypothesis instead of creating a new one.
                """
                self.minimum_hypothesis_overlap: float = 0.65
                """
                Intersection over union a detection must exceed with an existing object
                hypothesis to classify it.
                """

            def model_location(self, annotator_name: str) -> str:
                """
                :param annotator_name: Name of the annotator loading the model, for the
                    error.
                :return: The path of the pretrained model inside
                    :attr:`models_directory`, or else the path of the custom weights.
                :raises DetectionModelNotConfigured: If neither is configured.
                """
                if self.pretrained_model is not None:
                    return str(self.models_directory / self.pretrained_model)
                if self.ros_package_name is None or self.weights_path is None:
                    raise DetectionModelNotConfigured(annotator_name=annotator_name)
                return str(self.file_in_ros_package(self.weights_path))

            def file_in_ros_package(self, relative_path: str) -> Path:
                """
                :param relative_path: Path of a file, relative to the configured ROS
                    package.
                :return: The absolute path of the file.
                """
                return FileLoader.get_path_to_file_in_ros_package(
                    ros_pkg_name=self.ros_package_name, relative_path=relative_path
                )

    @classproperty
    def mask_foreground(cls) -> int:
        """
        Value of the object pixels in the masks this annotator computes.
        """
        return 255

    def __init__(
        self,
        name: str = "YoloAnnotator",
        descriptor: YoloAnnotator.Descriptor | None = None,
    ) -> None:
        super().__init__(name, descriptor)

        self.model_location: str = self.descriptor.parameters.model_location(self.name)
        """
        Where the YOLO model is loaded from.
        """
        self.model: YOLO | None = None
        """
        The YOLO detection model, loaded during setup.
        """
        self.class_names: dict[int, str] = {}
        """
        Name of each class the model detects, by class id, loaded during setup.
        """
        self.segmentation_model: SAM | None = None
        """
        The SAM model refining the hypothesis masks in precision mode, loaded during
        setup.
        """
        self.camera_intrinsics: o3d.camera.PinholeCameraIntrinsic | None = None
        """
        Projection model at the depth image resolution.
        """
        self.roi_comparator = RoiComparator(weight=1.0)
        """
        Comparator measuring how much a detection overlaps an existing hypothesis.
        """

    def setup(self, **kwargs: Any) -> bool:
        """
        Load the models, so that building a pipeline does not download them.
        """
        parameters: YoloAnnotator.Descriptor.Parameters = self.descriptor.parameters
        parameters.models_directory.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(self.model_location)
        self.class_names = self._load_class_names()
        if parameters.precision_mode:
            self.segmentation_model = SAM(
                str(parameters.models_directory / parameters.segmentation_model)
            )
        return super().setup(**kwargs)

    def _load_class_names(self) -> dict[int, str]:
        """
        :return: Name of each class the model detects, by class id.
        """
        parameters: YoloAnnotator.Descriptor.Parameters = self.descriptor.parameters
        if parameters.pretrained_model is not None:
            return dict(self.model.names)
        class_names_file = parameters.file_in_ros_package(parameters.class_names_path)
        with open(class_names_file, "r", encoding="utf-8") as file:
            return {int(class_id): name for class_id, name in json.load(file).items()}

    @timer_decorator
    def compute(self) -> Status:
        """
        Detect objects and annotate them as classified object hypotheses.
        """
        parameters: YoloAnnotator.Descriptor.Parameters = self.descriptor.parameters
        if parameters.global_with_depth:
            self.camera_intrinsics = self.get_cas().get_copy(CASViews.CAMERA_INTRINSIC)
            scale_camera_intrinsics(self)

        detection_image = self._detection_image()
        existing_hypotheses = (
            self.get_cas().filter_annotations_by_type(ObjectHypothesis)
            if parameters.classify_existing_hypotheses
            else []
        )

        annotated_detections = [
            self._classify(detection, detection_image, existing_hypotheses)
            for detection in self._detect(detection_image)
            if detection.classification.classname.strip()
            not in self._ignored_class_names()
        ]
        self.get_cas().annotations.extend(
            annotated.hypothesis
            for annotated in annotated_detections
            if annotated.is_new
        )

        if parameters.global_with_visualization:
            self._visualize(annotated_detections)
        return Status.SUCCESS

    def _detection_image(self) -> npt.NDArray[np.uint8]:
        """
        :return: The color image, scaled to the depth image resolution when depth is
            used.
        """
        color_image = self.get_cas().get_copy(CASViews.COLOR_IMAGE)
        if not self.descriptor.parameters.global_with_depth:
            return color_image
        return get_scaled_color_image_for_depth_image(self.get_cas(), color_image)

    def _detect(self, detection_image: npt.NDArray[np.uint8]) -> list[YoloDetection]:
        """
        :return: The objects the model detects in the image with sufficient confidence.
        """
        with torch.no_grad():
            results = self.model.predict(
                detection_image, conf=self.descriptor.parameters.threshold, show=False
            )
        detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for corners, confidence, class_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
                detections.append(
                    YoloDetection(
                        classification=Classification(
                            source=self.name,
                            classname=self._classname(int(class_id)),
                            confidence=float(confidence),
                            class_id=int(class_id),
                        ),
                        bounding_box=Rect.from_corner_points(*corners),
                    )
                )
        return detections

    def _classname(self, class_id: int) -> str:
        """
        :return: The name of the class with the given id.
        """
        return self.class_names.get(class_id, f"unknown. ID: {class_id}")

    def _ignored_class_names(self) -> set[str]:
        """
        :return: The configured ignored class names, without surrounding whitespace.
        """
        return {
            str(class_name).strip()
            for class_name in self.descriptor.parameters.ignored_class_names
        }

    def _classify(
        self,
        detection: YoloDetection,
        detection_image: npt.NDArray[np.uint8],
        existing_hypotheses: list[ObjectHypothesis],
    ) -> AnnotatedDetection:
        """
        Classify the existing hypothesis the detection overlaps, or else a new one.
        """
        region = self._color_image_region(detection)
        overlapped_hypothesis = self._most_overlapped_hypothesis(
            region, existing_hypotheses
        )
        is_new = overlapped_hypothesis is None
        hypothesis = (
            self._create_hypothesis(detection, detection_image, region)
            if is_new
            else overlapped_hypothesis
        )
        hypothesis.annotations.append(detection.classification)
        return AnnotatedDetection(
            detection=detection, hypothesis=hypothesis, is_new=is_new
        )

    def _color_image_region(self, detection: YoloDetection) -> Rect:
        """
        :return: The bounding box of the detection, in color image pixels.
        """
        if not self.descriptor.parameters.global_with_depth:
            return detection.bounding_box
        color_to_depth_ratio = self.get_cas().get(CASViews.COLOR_TO_DEPTH_RATIO)
        left, top, right, bottom = detection.bounding_box.get_corner_points()
        return Rect.from_corner_points(
            x1=left / color_to_depth_ratio.horizontal,
            y1=top / color_to_depth_ratio.vertical,
            x2=right / color_to_depth_ratio.horizontal,
            y2=bottom / color_to_depth_ratio.vertical,
        )

    def _most_overlapped_hypothesis(
        self, region: Rect, hypotheses: list[ObjectHypothesis]
    ) -> ObjectHypothesis | None:
        """
        :return: The hypothesis overlapping the region the most, if it overlaps by more
            than the configured minimum.
        """
        if len(hypotheses) == 0:
            return None
        most_overlapped = max(
            hypotheses,
            key=lambda hypothesis: self.roi_comparator.compute_similarity(
                hypothesis.roi.roi, region
            ),
        )
        overlap = self.roi_comparator.compute_similarity(
            most_overlapped.roi.roi, region
        )
        if overlap <= self.descriptor.parameters.minimum_hypothesis_overlap:
            return None
        return most_overlapped

    def _create_hypothesis(
        self,
        detection: YoloDetection,
        detection_image: npt.NDArray[np.uint8],
        region: Rect,
    ) -> ObjectHypothesis:
        """
        :return: A hypothesis for the detected object covering the given color image
            region.
        """
        hypothesis = (
            HumanHypothesis()
            if detection.classification.classname
            == self.descriptor.parameters.person_class_name
            else ObjectHypothesis()
        )
        hypothesis.source = self.name
        hypothesis.roi.roi = region

        detection_mask = self._detection_mask(detection, detection_image)
        if self.descriptor.parameters.global_with_depth:
            hypothesis.points = self._point_cloud(detection_mask, detection_image)
        hypothesis.roi.mask = crop_image(
            resize_mask(self, detection_mask),
            (region.pos.x, region.pos.y),
            (region.width, region.height),
        )
        return hypothesis

    def _detection_mask(
        self, detection: YoloDetection, detection_image: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """
        :return: The mask of the detected object on the detection image.
        """
        parameters: YoloAnnotator.Descriptor.Parameters = self.descriptor.parameters
        if not parameters.precision_mode:
            x1, y1, x2, y2 = detection.bounding_box.get_corner_points()
            mask = np.zeros(detection_image.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = self.mask_foreground
            return mask

        segmentation = self.segmentation_model.predict(
            detection_image,
            bboxes=[list(detection.bounding_box.get_corner_points())],
        )[0]
        mask = (
            segmentation.masks.data.cpu().numpy()[0].astype(np.uint8)
            * self.mask_foreground
        )
        torch.cuda.empty_cache()
        if not parameters.precision_mode_apply_erosion:
            return mask
        kernel_size = parameters.precision_mode_erosion_kernel_size
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.erode(mask, kernel, iterations=1)

    def _point_cloud(
        self,
        detection_mask: npt.NDArray[np.uint8],
        detection_image: npt.NDArray[np.uint8],
    ) -> o3d.geometry.PointCloud:
        """
        :return: The point cloud of the depth measurements inside the detection mask.
        """
        return get_cloud_from_rgb_depth_and_mask(
            rgb_image=cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB),
            depth_image=self.get_cas().get(CASViews.DEPTH_IMAGE),
            mask=detection_mask,
            camera_intrinsics=self.camera_intrinsics,
            depth_truncate=self.descriptor.parameters.depth_truncate,
            mask_true_value=self.mask_foreground,
        )

    def _visualize(self, annotated_detections: list[AnnotatedDetection]) -> None:
        """
        Draw the annotated hypotheses into the color image, together with their masks in
        precision mode, and output the point clouds of the new hypotheses.
        """
        parameters: YoloAnnotator.Descriptor.Parameters = self.descriptor.parameters
        visualization_image = self.get_cas().get_copy(CASViews.COLOR_IMAGE)
        for index, annotated in enumerate(annotated_detections):
            if parameters.precision_mode and annotated.is_new:
                x1, y1, _, _ = annotated.hypothesis.roi.roi.get_corner_points()
                mask = annotated.hypothesis.roi.mask
                region = visualization_image[
                    y1 : y1 + mask.shape[0], x1 : x1 + mask.shape[1]
                ]
                region[mask == self.mask_foreground] = colors(index, bgr=True)
            classification = annotated.detection.classification
            draw_bounding_boxes_from_object_hypotheses(
                visualization_image,
                [annotated.hypothesis],
                lambda _: f"{classification.classname}, {classification.confidence:.2f}",
            )
        self.get_annotator_output_struct().set_image(visualization_image)
        if parameters.global_with_depth:
            self.get_annotator_output_struct().set_geometries(
                [
                    annotated.hypothesis.points
                    for annotated in annotated_detections
                    if annotated.is_new
                ]
            )
