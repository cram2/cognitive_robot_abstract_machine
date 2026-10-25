"""
Exceptions raised by RoboKudo annotators.
"""

from __future__ import annotations

from dataclasses import dataclass

from robokudo.exceptions import RoboKudoError

# %% model configuration


@dataclass
class NotAPoseModelException(RoboKudoError, ValueError):
    """
    Raised when an annotator that reads body keypoints loaded a model reporting none.
    """

    annotator_name: str
    """
    Name of the annotator whose model reports no keypoints.
    """

    model: str
    """
    The model the annotator loaded.
    """

    def error_message(self) -> str:
        return f"{self.annotator_name} loaded {self.model}, which reports no body keypoints."

    def suggest_correction(self) -> str:
        return "set 'pretrained_model' to a YOLO pose model, e.g. 'yolov8n-pose.pt'."
