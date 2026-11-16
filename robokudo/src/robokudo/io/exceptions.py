"""
Errors raised by RoboKudo camera interfaces.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from robokudo.exceptions import InvalidCameraObservation, RoboKudoError
from robokudo.types.image import ImageRotation
from semantic_digital_twin.datastructures.camera_model import CameraDistortionModel


# %% Image rotation
@dataclass
class UnsupportedImageRotation(RoboKudoError, ValueError):
    """
    A requested image rotation is outside the supported orientations.
    """

    rotation: str | None
    """
    Unsupported rotation supplied to the camera interface.
    """

    def error_message(self) -> str:
        """
        Identify the unsupported rotation.
        """
        return f"Unsupported image rotation: {self.rotation!r}."

    def suggest_correction(self) -> str:
        """
        List the supported rotation choices.
        """
        choices = ", ".join(rotation.value for rotation in ImageRotation)
        return f"Use one of the supported image rotations: {choices}."


# %% Camera distortion metadata
@dataclass
class CameraDistortionModelMissing(InvalidCameraObservation):
    """
    Nonzero distortion coefficients have no declared distortion model.
    """

    reason: str = field(init=False)
    """
    Description of the missing model and its coefficients.
    """

    coefficients: tuple[float, ...]
    """
    Distortion coefficients supplied without a model.
    """

    def __post_init__(self) -> None:
        """
        Describe the coefficients that require a distortion model.
        """
        self.reason = f"nonzero distortion coefficients {self.coefficients!r} have no distortion model"

    def suggest_correction(self) -> str:
        """
        Identify the missing calibration metadata.
        """
        return "Declare the distortion model matching the supplied camera coefficients."


@dataclass
class UnsupportedCameraDistortionModel(InvalidCameraObservation):
    """
    Camera calibration declares a distortion model the adapter cannot interpret.
    """

    reason: str = field(init=False)
    """
    Description of the unsupported model.
    """

    distortion_model: str
    """
    Unrecognized distortion model supplied by the camera.
    """

    def __post_init__(self) -> None:
        """
        Describe the unrecognized calibration model.
        """
        self.reason = f"unsupported camera distortion model: {self.distortion_model!r}"

    def suggest_correction(self) -> str:
        """
        List the models supported at the camera-info boundary.
        """
        choices = ", ".join(
            model.value
            for model in CameraDistortionModel
            if model is not CameraDistortionModel.NONE
        )
        return f"Supply calibration using one of the supported distortion models: {choices}."


# %% Semantic robot selection


@dataclass
class RobotAnnotationMissing(RoboKudoError):
    """
    Raised when no semantic robot matches the runtime-world selection.
    """

    robot_name: str | None = None
    """
    Requested robot name, or ``None`` when no name was configured.
    """

    def error_message(self) -> str:
        """
        Describe the missing robot selection.
        """
        if self.robot_name is None:
            return (
                "The runtime Semantic Digital Twin world contains no robot annotation."
            )
        return f"No robot annotation named '{self.robot_name}' exists in the runtime world."

    def suggest_correction(self) -> str:
        """
        Describe how to make a runtime robot selectable.
        """
        return "synchronize a robot into the runtime world or select an existing robot."


@dataclass
class RobotAnnotationAmbiguous(RoboKudoError):
    """
    Raised when runtime-world robot selection has multiple valid results.
    """

    robot_names: tuple[str, ...]
    """
    Names of robot annotations that satisfy the selection.
    """

    def error_message(self) -> str:
        """
        Describe the ambiguous robot selection.
        """
        return f"Robot selection is ambiguous between {self.robot_names}."

    def suggest_correction(self) -> str:
        """
        Describe how to select one runtime robot.
        """
        return "configure the name of the robot whose default camera should be used."


# %% Semantic camera selection


@dataclass
class CameraAnnotationMissing(RoboKudoError):
    """
    Raised when no semantic camera matches the configured selection.
    """

    camera_name: str | None = None
    """
    Requested camera name, or ``None`` when no name was configured.
    """

    def error_message(self) -> str:
        if self.camera_name is None:
            return "The Semantic Digital Twin world contains no camera annotation."
        return f"No camera annotation named '{self.camera_name}' exists in the world."

    def suggest_correction(self) -> str:
        return "add a Camera annotation to the world or select an existing camera."


@dataclass
class CameraAnnotationAmbiguous(RoboKudoError):
    """
    Raised when camera selection has more than one valid result.
    """

    camera_names: tuple[str, ...]
    """
    Names of camera annotations that satisfy the selection.
    """

    def error_message(self) -> str:
        return f"Camera selection is ambiguous between {self.camera_names}."

    def suggest_correction(self) -> str:
        return "configure the name of the camera that should provide perception data."


@dataclass
class CameraPoseOverrideUnavailable(RoboKudoError):
    """
    Raised when the selected camera attachment cannot accept a pose override.
    """

    camera_name: str
    """
    Name of the semantic camera whose pose cannot be changed.
    """

    connection_type: str | None
    """
    Camera-root connection type, or ``None`` when the camera is the world root.
    """

    def error_message(self) -> str:
        """
        Describe why the configured pose cannot be applied.
        """
        if self.connection_type is None:
            return f"Camera '{self.camera_name}' has no parent connection."
        return (
            f"Camera '{self.camera_name}' is attached through immutable connection "
            f"type '{self.connection_type}'."
        )

    def suggest_correction(self) -> str:
        """
        Describe the attachment required for configurable camera placement.
        """
        return "attach the camera root through a Connection6DoF or omit camera_pose."


# %% Recorded storage
@dataclass
class RecordedViewMissing(RoboKudoError):
    """
    A recorded CAS references a view absent from storage.
    """

    view_name: str
    """
    Name of the missing CAS view.
    """

    view_id: object
    """
    Identifier referenced by the recorded CAS.
    """

    def error_message(self) -> str:
        return f"Could not find view {self.view_name!r} with id={self.view_id}."

    def suggest_correction(self) -> str:
        return "Restore the referenced view or select a complete recording."


@dataclass
class CASViewCodecMissing(RoboKudoError, TypeError):
    """
    A CAS view has no registered codec.
    """

    view_name: str
    """
    Name of the view to encode.
    """

    value_type: type
    """
    Type of the view value.
    """

    def error_message(self) -> str:
        return f"No codec registered for view {self.view_name!r} with type {self.value_type}."

    def suggest_correction(self) -> str:
        return "Register a codec for this view type."


@dataclass
class UnknownViewSerializer(RoboKudoError, ValueError):
    """
    A recorded serializer identifier has no registered decoder.
    """

    serializer_id: str
    """
    Recorded serializer identifier.
    """

    def error_message(self) -> str:
        return f"Unknown CAS view serializer {self.serializer_id!r}."

    def suggest_correction(self) -> str:
        return "Register the recorded serializer before playback."


@dataclass
class ViewStorageFailed(RoboKudoError, RuntimeError):
    """
    Storage did not acknowledge a CAS view write.
    """

    view_name: str
    """
    Name of the view being stored.
    """

    def error_message(self) -> str:
        return f"Storage did not acknowledge view {self.view_name!r}."

    def suggest_correction(self) -> str:
        return "Check the storage connection and write acknowledgement settings."


@dataclass
class RecordedViewNameMismatch(RoboKudoError, RuntimeError):
    """
    A stored view has a different name than its reference.
    """

    expected_name: str
    """
    View name referenced by the CAS.
    """

    recorded_name: str
    """
    Name decoded from the stored view.
    """

    def error_message(self) -> str:
        return f"Decoded view {self.recorded_name!r} does not match {self.expected_name!r}."

    def suggest_correction(self) -> str:
        return "Select a recording with consistent CAS view references."


@dataclass
class InvalidFileSequence(RoboKudoError, ValueError):
    """
    File playback keys do not match its data.
    """

    sequence_length: int
    """
    Number of requested playback entries.
    """

    data_length: int
    """
    Number of available data entries.
    """

    def error_message(self) -> str:
        return (
            f"Playback has {self.sequence_length} keys for {self.data_length} entries."
        )

    def suggest_correction(self) -> str:
        return "Supply one playback key for each data entry."


@dataclass
class InvalidFileSource(RoboKudoError, ValueError):
    """
    The configured file source cannot be read.
    """

    reason: str
    """
    Problem with the configured source.
    """

    def error_message(self) -> str:
        return self.reason

    def suggest_correction(self) -> str:
        return "Configure an existing readable directory containing valid image files."
