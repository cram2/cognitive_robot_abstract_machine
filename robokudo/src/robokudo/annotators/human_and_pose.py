"""
Human detection with body keypoint estimation.

Detects humans with a YOLO pose model and annotates each one as a human hypothesis that
carries a person classification and its 2D body keypoints. With depth, each hypothesis
also gets the point cloud of its bounding box and, optionally, one point cloud per
clearly visible keypoint.

.. note::
   The model needs the packages of RoboKudo's optional ``dev`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import cv2
import numpy as np
import open3d
from krrood.ormatic.utils import classproperty
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING, Any
from ultralytics import YOLO

from robokudo.annotators.core import ThreadedAnnotator
from robokudo.annotators.exceptions import NotAPoseModelException
from robokudo.cas import CASViews
from robokudo.defs import PACKAGE_NAME
from robokudo.types.annotation import Classification, CloudAnnotation
from robokudo.types.cv import Rect
from robokudo.types.human import (
    BodyKeypoint,
    BodyKeypointCloud,
    KeypointAnnotation,
    KeypointKind,
)
from robokudo.types.scene import HumanHypothesis
from robokudo.utils.cv_helper import get_scaled_color_image_for_depth_image
from robokudo.utils.decorators import record_time
from robokudo.utils.module_loader import ModuleLoader
from robokudo.utils.o3d_helper import (
    concatenate_clouds,
    get_cloud_from_rgb_depth_and_mask,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from ultralytics.engine.results import Results

    from robokudo.types.cv import ColorToDepthRatio
    from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel

# %% body keypoints


class PoseKeypoint(IntEnum):
    """
    Index of each body keypoint in the keypoints a YOLO pose model reports.
    """

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


# %% depth


@dataclass
class DepthFrame:
    """
    The depth measurements point clouds are built from, with matching color.
    """

    depth_image: npt.NDArray[np.uint16]
    """
    The depth image.
    """

    rgb_image: npt.NDArray[np.uint8]
    """
    The color image in RGB order, scaled to the depth image resolution.
    """

    color_to_depth_ratio: ColorToDepthRatio
    """
    Size of the depth image relative to the color image.
    """

    camera_model: PinholeCameraModel
    """
    Projection model at the depth image resolution.
    """

    @classproperty
    def mask_foreground(cls) -> int:
        """
        Value of the pixels a point cloud is built from in a depth mask.
        """
        return 255

    def box_cloud(self, box: Rect, depth_truncate: float) -> open3d.geometry.PointCloud:
        """
        Build the point cloud of the depth measurements inside a box.

        :param box: The box in color image pixels.
        :param depth_truncate: Distance in meters beyond which measurements are left
            out.
        :return: The colored point cloud of the depth measurements inside the box.
        """
        left, top, right, bottom = self.color_to_depth_ratio.depth_region(
            box
        ).get_corner_points()
        mask = np.zeros_like(self.depth_image, dtype=np.uint8)
        mask[top:bottom, left:right] = self.mask_foreground
        return self._masked_cloud(mask, depth_truncate=depth_truncate)

    def disc_cloud(self, x: float, y: float, radius: int) -> open3d.geometry.PointCloud:
        """
        Build the point cloud of the depth measurements inside a disc.

        :param x: Horizontal disc center in color image pixels.
        :param y: Vertical disc center in color image pixels.
        :param radius: Disc radius in depth image pixels.
        :return: The colored point cloud of the depth measurements inside the disc.
        """
        center = self.color_to_depth_ratio.depth_pixel(x, y)
        mask = np.zeros_like(self.depth_image, dtype=np.uint8)
        cv2.circle(
            mask,
            (center.x, center.y),
            radius,
            self.mask_foreground,
            thickness=cv2.FILLED,
        )
        return self._masked_cloud(mask)

    def _masked_cloud(
        self, mask: npt.NDArray[np.uint8], **cloud_settings: float
    ) -> open3d.geometry.PointCloud:
        """
        Build the point cloud of the depth measurements a mask marks.

        :param mask: Mask on the depth image whose pixels with value
            :attr:`mask_foreground` the cloud is built from.
        :param cloud_settings: Further settings of
            :func:`~robokudo.utils.o3d_helper.get_cloud_from_rgb_depth_and_mask`.
        :return: The colored point cloud of the depth measurements inside the mask.
        """
        return get_cloud_from_rgb_depth_and_mask(
            rgb_image=self.rgb_image,
            depth_image=self.depth_image,
            mask=mask,
            camera_model=self.camera_model,
            mask_true_value=self.mask_foreground,
            **cloud_settings,
        )


# %% annotator


class HumanAndPoseAnnotator(ThreadedAnnotator):
    """
    Detects humans with a YOLO pose model and annotates them with their keypoints.
    """

    class Descriptor(ThreadedAnnotator.Descriptor):
        """
        Descriptor for the HumanAndPoseAnnotator.
        """

        class Parameters:
            """
            Parameters for the HumanAndPoseAnnotator.
            """

            def __init__(self) -> None:
                self.models_directory: Path = (
                    ModuleLoader.get_module_path(PACKAGE_NAME) / "models"
                )
                """
                Directory the pose model weights are loaded from.

                Created during setup if it does not exist.
                """
                self.pretrained_model: str = "yolov8n-pose.pt"
                """
                File name of the YOLO pose model weights inside
                :attr:`models_directory`.

                A model ultralytics provides is downloaded there on first use; custom
                trained weights must already be there.

                .. note::
                   The model must report body keypoints, otherwise
                   :meth:`HumanAndPoseAnnotator.compute` raises
                   :class:`~robokudo.annotators.exceptions.NotAPoseModelException`.
                """
                self.person_class_name: str = "person"
                """
                Class name the human hypotheses are classified as.
                """
                self.confidence_threshold: float = 0.7
                """
                Minimum confidence of a human detection to be annotated.
                """
                self.non_maximum_suppression_overlap: float = 0.5
                """
                Intersection over union above which the model merges overlapping
                detections.
                """
                self.global_with_depth: bool = False
                """
                Whether to build point clouds from the depth image.
                """
                self.shrink_bounding_box: float | None = None
                """
                Fraction in [0, 1] of its width and height the bounding box of each
                human is shrunk by around its center, or None to keep it.

                Shrinking may interfere with activity recognition.
                """
                self.shrink_as_annotation: bool = False
                """
                Whether, with depth, the shrunk box only yields a cloud annotation
                instead of replacing the region of the human hypothesis.
                """
                self.generate_clouds_for_keypoints: bool = False
                """
                Whether, with depth, to build a point cloud around every keypoint.

                .. warning::
                   This is computationally expensive.
                """
                self.keypoint_confidence_threshold_for_clouds: float = 0.6
                """
                Minimum confidence of a keypoint to get a non-empty point cloud.
                """
                self.keypoint_masking_radius: int = 6
                """
                Radius in depth image pixels around each keypoint its cloud is built
                from.
                """
                self.warmup_annotator: bool = True
                """
                Whether setup runs the model once, so that the first real detection is
                not slowed down by its initialization.
                """
                self.warmup_image_shape: tuple[int, int, int] = (512, 512, 3)
                """
                Shape (height, width, channels) of the blank image the model is warmed
                up on.
                """
                self.depth_truncate: float = 3.5
                """
                Distance in meters beyond which depth measurements are left out of the
                box clouds, removing background from the human detection.
                """

    def __init__(
        self,
        name: str = "HumanAndPoseAnnotator",
        descriptor: HumanAndPoseAnnotator.Descriptor | None = None,
    ) -> None:
        super().__init__(name, descriptor)
        self.model: YOLO | None = None
        """
        The YOLO pose model, loaded during setup.
        """

    def setup(self, **kwargs: Any) -> bool:
        """
        Extends :meth:`ThreadedAnnotator.setup` by loading the pose model from
        :attr:`Descriptor.Parameters.models_directory`, creating that directory if it
        does not exist, and, if configured, warming the model up.
        """
        parameters: HumanAndPoseAnnotator.Descriptor.Parameters = (
            self.descriptor.parameters
        )
        parameters.models_directory.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(
            str(parameters.models_directory / parameters.pretrained_model)
        )
        if parameters.warmup_annotator:
            self._predict(np.zeros(parameters.warmup_image_shape, dtype=np.uint8))
        return super().setup(**kwargs)

    def _predict(self, image: npt.NDArray[np.uint8]) -> Results:
        """
        Run the pose model on an image with the configured confidence threshold and non-
        maximum suppression overlap.

        :param image: The color image to detect humans in.
        :return: The humans the model detects in the image, with their bounding boxes,
            confidences and keypoints.
        """
        parameters: HumanAndPoseAnnotator.Descriptor.Parameters = (
            self.descriptor.parameters
        )
        return self.model.predict(
            image,
            conf=parameters.confidence_threshold,
            iou=parameters.non_maximum_suppression_overlap,
            save=False,
            verbose=False,
        )[0]

    @record_time
    def compute(self) -> Status:
        """
        Detect humans in the color image of the CAS and add one human hypothesis with
        its keypoints per detection to the CAS.

        :return: :attr:`Status.SUCCESS` once the hypotheses are added.
        :raises NotAPoseModelException: If the model reports no keypoints for a
            detection.
        """
        parameters: HumanAndPoseAnnotator.Descriptor.Parameters = (
            self.descriptor.parameters
        )
        depth_frame = self._depth_frame() if parameters.global_with_depth else None
        result = self._predict(self.get_cas().get(CASViews.COLOR_IMAGE))

        hypotheses = [
            self._hypothesis(result, index, depth_frame)
            for index in range(len(result.boxes))
        ]
        self.get_cas().annotations.extend(hypotheses)

        if parameters.global_with_visualization:
            self._visualize(result, hypotheses)
        return Status.SUCCESS

    def _depth_frame(self) -> DepthFrame:
        """
        Collect the depth measurements of the CAS and scale the color image and the
        camera model to the depth image resolution.

        :return: The depth measurements of the CAS with the matching color image.
        """
        cas = self.get_cas()
        color_to_depth_ratio = cas.color_to_depth_ratio
        scaled_color = get_scaled_color_image_for_depth_image(
            cas, cas.get(CASViews.COLOR_IMAGE)
        )
        return DepthFrame(
            depth_image=cas.get(CASViews.DEPTH_IMAGE),
            rgb_image=cv2.cvtColor(scaled_color, cv2.COLOR_BGR2RGB),
            color_to_depth_ratio=color_to_depth_ratio,
            camera_model=cas.require_camera_observation().effective_camera_model.scaled(
                scale_x=color_to_depth_ratio.horizontal,
                scale_y=color_to_depth_ratio.vertical,
            ),
        )

    def _hypothesis(
        self, result: Results, index: int, depth_frame: DepthFrame | None
    ) -> HumanHypothesis:
        """
        Turn one detection into a human hypothesis classified as a person and annotated
        with its 2D keypoints, and, with depth, with its point clouds.

        :param result: The detections of the pose model on the color image.
        :param index: Index of the detection in the result, which becomes the id of the
            hypothesis.
        :param depth_frame: The depth measurements to build point clouds from, or None
            to build none.
        :return: The human hypothesis of the detection.
        :raises NotAPoseModelException: If the model reports no keypoints.
        """
        parameters: HumanAndPoseAnnotator.Descriptor.Parameters = (
            self.descriptor.parameters
        )
        if result.keypoints is None:
            raise NotAPoseModelException(
                annotator_name=self.name, model=parameters.pretrained_model
            )
        box = Rect.from_corner_points(*result.boxes.xyxy[index].cpu().numpy())
        shrunk_box = (
            box if parameters.shrink_bounding_box is None else self._shrink_box(box)
        )
        keeps_full_box = depth_frame is not None and parameters.shrink_as_annotation

        hypothesis = HumanHypothesis()
        hypothesis.id = index
        hypothesis.source = self.name
        hypothesis.roi.roi = box if keeps_full_box else shrunk_box
        hypothesis.annotations.append(
            Classification(
                source=self.name,
                classname=parameters.person_class_name,
                confidence=float(result.boxes.conf[index]),
            )
        )

        keypoints = [
            BodyKeypoint(x=x, y=y, confidence=confidence)
            for x, y, confidence in result.keypoints.data[index].cpu().numpy().tolist()
        ]
        hypothesis.annotations.append(
            KeypointAnnotation(
                source=self.name,
                keypoints=keypoints,
                kind=KeypointKind.IMAGE_POSITION,
            )
        )
        if depth_frame is None:
            return hypothesis

        hypothesis.points = depth_frame.box_cloud(
            hypothesis.roi.roi, parameters.depth_truncate
        )
        if parameters.shrink_bounding_box is not None and keeps_full_box:
            hypothesis.annotations.append(
                CloudAnnotation(
                    source=self.name,
                    points=depth_frame.box_cloud(shrunk_box, parameters.depth_truncate),
                )
            )
        if parameters.generate_clouds_for_keypoints:
            hypothesis.annotations.append(
                KeypointAnnotation(
                    source=self.name,
                    keypoints=self._keypoint_clouds(keypoints, depth_frame),
                    kind=KeypointKind.POINT_CLOUD,
                )
            )
        return hypothesis

    def _shrink_box(self, box: Rect) -> Rect:
        """
        Shrink a box around its center by
        :attr:`Descriptor.Parameters.shrink_bounding_box` of its width and height.

        :param box: The box in color image pixels.
        :return: The shrunk box in color image pixels.
        """
        shrink_fraction = self.descriptor.parameters.shrink_bounding_box
        horizontal_margin = box.width * shrink_fraction / 2
        vertical_margin = box.height * shrink_fraction / 2
        return Rect.from_corner_points(
            x1=box.pos.x + horizontal_margin,
            y1=box.pos.y + vertical_margin,
            x2=box.pos.x + box.width - horizontal_margin,
            y2=box.pos.y + box.height - vertical_margin,
        )

    def _keypoint_clouds(
        self, keypoints: list[BodyKeypoint], depth_frame: DepthFrame
    ) -> list[BodyKeypointCloud]:
        """
        Build a point cloud from the disc of
        :attr:`Descriptor.Parameters.keypoint_masking_radius` around each keypoint.

        :param keypoints: The body keypoints of one human.
        :param depth_frame: The depth measurements to build the clouds from.
        :return: The cloud around each keypoint, empty for keypoints below
            :attr:`Descriptor.Parameters.keypoint_confidence_threshold_for_clouds`.
        """
        parameters: HumanAndPoseAnnotator.Descriptor.Parameters = (
            self.descriptor.parameters
        )
        return [
            BodyKeypointCloud(
                keypoint=keypoint,
                points=(
                    depth_frame.disc_cloud(
                        keypoint.x, keypoint.y, parameters.keypoint_masking_radius
                    )
                    if keypoint.confidence
                    >= parameters.keypoint_confidence_threshold_for_clouds
                    else open3d.geometry.PointCloud()
                ),
            )
            for keypoint in keypoints
        ]

    def _visualize(self, result: Results, hypotheses: list[HumanHypothesis]) -> None:
        """
        Output the color image with the detections drawn by the model and, if any
        hypothesis has one, the point clouds of the hypotheses merged into one.

        :param result: The detections of the pose model on the color image.
        :param hypotheses: The human hypotheses made from the detections.
        """
        self.get_annotator_output_struct().set_image(result.plot())
        clouds = [
            hypothesis.points
            for hypothesis in hypotheses
            if hypothesis.points is not None
        ]
        if len(clouds) > 0:
            self.get_annotator_output_struct().set_geometries(
                concatenate_clouds(clouds)
            )
