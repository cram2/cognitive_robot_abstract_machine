"""
Exceptions raised by RoboKudo annotators.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass

from semantic_digital_twin.datastructures.camera_resolution import CameraResolution

from robokudo.exceptions import RoboKudoError


# %% Collection reader placement
@dataclass(kw_only=True)
class InvalidCollectionReaderParent(RoboKudoError):
    """
    A collection reader has no parent sequence to initialize.
    """

    def error_message(self) -> str:
        """
        Identify the invalid reader placement.
        """
        return "The collection reader's parent is not a behavior-tree sequence."

    def suggest_correction(self) -> str:
        """
        Describe valid placement of the reader.
        """
        return "Add the collection reader to a pipeline or sequence before ticking it."


# %% Analysis contracts
@dataclass(kw_only=True)
class InvalidAnalysisData(RoboKudoError):
    """
    Analysis data does not satisfy its required type.
    """

    expected_type: type
    """
    Type required by the analysis operation.
    """

    actual_type: type
    """
    Type of the supplied data.
    """

    def error_message(self) -> str:
        return f"Expected {self.expected_type.__name__}, received {self.actual_type.__name__}."

    def suggest_correction(self) -> str:
        return "Provide analysis data of the required type."


@dataclass(kw_only=True)
class EmptyAnalysisScope(RoboKudoError, ValueError):
    """
    An analysis scope selects no data.
    """

    analysis_scope: object
    """
    Scope that selected no analysis data.
    """

    def error_message(self) -> str:
        return f"Analysis scope selected no data: {self.analysis_scope!r}."

    def suggest_correction(self) -> str:
        return "Select an available CAS view or annotation type."


# %% model configuration


@dataclass(kw_only=True)
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


# %% object pose estimation


@dataclass(kw_only=True)
class DuplicateObjectClassMesh(RoboKudoError, ValueError):
    """
    Raised when more than one mesh is configured for an object class.
    """

    class_name: str
    """
    The object class with more than one mesh.
    """

    def error_message(self) -> str:
        return f"More than one mesh is configured for the class '{self.class_name}'."

    def suggest_correction(self) -> str:
        return "Configure exactly one mesh per class name."


@dataclass(kw_only=True)
class ImageResolutionMismatch(RoboKudoError, ValueError):
    """
    Raised when the color and the depth image of a frame differ in resolution.
    """

    color_image_resolution: CameraResolution
    """
    Resolution of the color image.
    """

    depth_image_resolution: CameraResolution
    """
    Resolution of the depth image.
    """

    def error_message(self) -> str:
        return (
            f"The color image has the resolution {self.color_image_resolution}, the "
            f"depth image {self.depth_image_resolution}."
        )

    def suggest_correction(self) -> str:
        return "Scale the color image to the resolution of the depth image."


# %% World updating


@dataclass(kw_only=True)
class CameraPoseOutsideWorldError(RoboKudoError, LookupError):
    """
    Raised when the camera pose of a pass is measured against a body that is not part of
    the world RoboKudo perceives into.
    """

    reference_frame_name: str
    """
    The name of the body the camera pose is measured against.
    """

    def error_message(self) -> str:
        return (
            f"The camera pose of the pass is measured against '{self.reference_frame_name}', "
            f"which is not a body of the world RoboKudo perceives into, so the objects "
            f"of the pass cannot be placed in it."
        )

    def suggest_correction(self) -> str:
        return (
            "let the pipeline perceive into the world of its RoboKudo, which "
            "RoboKudo.start gives every pipeline of the analysis engine."
        )


@dataclass(kw_only=True)
class QueryServerNotStarted(RoboKudoError, RuntimeError):
    """
    Raised when a query annotator is updated before its initialisation connected it to
    the query action server.
    """

    annotator_name: str
    """
    The name of the annotator that was updated too early.
    """

    def error_message(self) -> str:
        return (
            f"{self.annotator_name} was updated before it was connected to the query "
            f"action server."
        )

    def suggest_correction(self) -> str:
        return (
            "set the annotator up or initialise it before ticking it, as a behaviour "
            "tree does when it runs the annotator."
        )
