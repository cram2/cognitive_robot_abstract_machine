"""
Mocks of the ultralytics YOLO and SAM models that report predefined results.

They return genuine ultralytics result objects, so the annotators read them exactly as
they read the results of a real model, without downloading weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from typing_extensions import Any
from ultralytics.engine.results import Results

from robokudo.types.human import BodyKeypoint

# %% detection model


@dataclass
class NormalizedBox:
    """
    A bounding box given as fractions of the image width and height.
    """

    left: float
    """
    Fraction of the image width left of the box.
    """

    top: float
    """
    Fraction of the image height above the box.
    """

    right: float
    """
    Fraction of the image width left of the box's right edge.
    """

    bottom: float
    """
    Fraction of the image height above the box's bottom edge.
    """

    def corners_in(self, image: np.ndarray) -> list[float]:
        """
        :return: The corners (x1, y1, x2, y2) of the box in pixels of the given image.
        """
        height, width = image.shape[:2]
        return [
            self.left * width,
            self.top * height,
            self.right * width,
            self.bottom * height,
        ]


@dataclass
class NormalizedKeypoint:
    """
    A body keypoint given as fractions of the image width and height.
    """

    x: float
    """
    Fraction of the image width left of the keypoint.
    """

    y: float
    """
    Fraction of the image height above the keypoint.
    """

    confidence: float
    """
    Confidence that the keypoint is visible at this position.
    """

    def in_pixels_of(self, image: np.ndarray) -> BodyKeypoint:
        """
        :return: The keypoint in pixels of the given image.
        """
        height, width = image.shape[:2]
        return BodyKeypoint(
            x=self.x * width, y=self.y * height, confidence=self.confidence
        )


@dataclass
class MockDetection:
    """
    A detection the mock model reports on every image.
    """

    class_id: int
    """
    Identifier of the detected class.
    """

    confidence: float
    """
    Confidence of the detection.
    """

    box: NormalizedBox
    """
    Where the detection lies, relative to the image size.
    """

    keypoints: list[NormalizedKeypoint] = field(default_factory=list)
    """
    The body keypoints a pose model reports for the detection.
    """


@dataclass
class MockDetectionModel:
    """
    YOLO mock reporting its predefined detections on any image.
    """

    names: dict[int, str]
    """
    Name of each class the model detects, by class id.
    """

    detections: list[MockDetection] = field(default_factory=list)
    """
    The detections reported on every image.
    """

    received_confidence: float | None = None
    """
    The confidence threshold of the last prediction.
    """

    received_image_shape: tuple | None = None
    """
    Shape of the image of the last prediction.
    """

    received_prediction_settings: dict[str, Any] = field(default_factory=dict)
    """
    The further ultralytics prediction settings of the last prediction, by keyword.
    """

    def predict(
        self, image: np.ndarray, conf: float, **prediction_settings: Any
    ) -> list[Results]:
        """
        :return: The predefined detections at least as confident as the threshold.
        """
        self.received_confidence = conf
        self.received_image_shape = image.shape
        self.received_prediction_settings = prediction_settings
        reported = [
            detection for detection in self.detections if detection.confidence >= conf
        ]
        boxes = torch.tensor(
            [
                [
                    *detection.box.corners_in(image),
                    detection.confidence,
                    detection.class_id,
                ]
                for detection in reported
            ],
            dtype=torch.float32,
        ).reshape(-1, 6)
        keypoints = (
            torch.tensor(
                [
                    [
                        [pixel.x, pixel.y, pixel.confidence]
                        for pixel in (
                            keypoint.in_pixels_of(image)
                            for keypoint in detection.keypoints
                        )
                    ]
                    for detection in reported
                ],
                dtype=torch.float32,
            )
            if any(detection.keypoints for detection in reported)
            else None
        )
        return [
            Results(image, path="", names=self.names, boxes=boxes, keypoints=keypoints)
        ]


# %% segmentation model


@dataclass
class TopHalfSegmentationModel:
    """
    SAM mock segmenting the upper half of every prompted bounding box.
    """

    def segment(self, image: np.ndarray, corners: list[float]) -> np.ndarray:
        """
        :return: A mask on the image covering the upper half of the box.
        """
        x1, y1, x2, y2 = (int(corner) for corner in corners)
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        mask[y1 : (y1 + y2) // 2, x1:x2] = 1.0
        return mask

    def predict(self, image: np.ndarray, bboxes: list[list[float]]) -> list[Results]:
        """
        :return: The segmentation of the first prompted box.
        """
        mask = self.segment(image, bboxes[0])
        return [Results(image, path="", names={}, masks=torch.tensor(mask[None]))]


# %% model loading


@dataclass
class MockModelLoader:
    """
    The models handed out in place of the ones ultralytics would load.
    """

    model: MockDetectionModel
    """
    The detection model handed out for every model location.
    """

    segmentation_model: TopHalfSegmentationModel = field(
        default_factory=TopHalfSegmentationModel
    )
    """
    The segmentation model handed out for every SAM model location.
    """

    loaded_model_locations: list[str] = field(default_factory=list)
    """
    The locations the detection model was loaded from.
    """

    loaded_segmentation_model_locations: list[str] = field(default_factory=list)
    """
    The locations the segmentation model was loaded from.
    """

    def load_model(self, location: str) -> MockDetectionModel:
        """
        Mock of constructing a YOLO model from the given location.
        """
        self.loaded_model_locations.append(location)
        return self.model

    def load_segmentation_model(self, location: str) -> TopHalfSegmentationModel:
        """
        Mock of constructing a SAM model from the given location.
        """
        self.loaded_segmentation_model_locations.append(location)
        return self.segmentation_model
