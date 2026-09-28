"""
Validated recording-save requests shared by the viewer and live bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from cramera.live.frame_range import FrameRange, InvalidFrameRange
from cramera.live.recording_storage import SceneDestination, save_recording_bundle


# %% request fields


class RecordingSaveField(StrEnum):
    """
    Fields accepted by the browser's recording-save action.
    """

    NAME = "name"
    """
    Permanent scene name.
    """

    DESTINATION = "destination"
    """
    Local or shared scene storage.
    """

    FIRST_FRAME = "firstFrame"
    """
    Inclusive first frame of an optional trim.
    """

    LAST_FRAME = "lastFrame"
    """
    Inclusive last frame of an optional trim.
    """

    ROBOT = "robot"
    """
    Optional robot display name.
    """

    ENVIRONMENT = "environment"
    """
    Optional environment display name.
    """

    TASK = "task"
    """
    Optional description of the recorded task.
    """


class InvalidRecordingSaveRequest(ValueError):
    """
    A save request contains a field of the wrong type or an unknown destination.
    """


# %% validated recording selection


@dataclass(frozen=True)
class RecordingSaveRequest:
    """
    A named recording selection ready to save in local or shared storage.
    """

    name: str
    """
    Requested permanent scene name.
    """

    destination: SceneDestination
    """
    Storage root receiving the saved recording.
    """

    frame_range: FrameRange | None
    """
    Inclusive range to save, or the complete recording when absent.
    """

    robot: str | None
    """
    Optional robot display name.
    """

    environment: str | None
    """
    Optional environment display name.
    """

    task: str | None
    """
    Optional description of the run.
    """

    @classmethod
    def from_json(cls, payload: object) -> RecordingSaveRequest:
        """
        Validate the browser's JSON body before selecting or changing any frames.

        :param payload: Decoded JSON request body.
        :raises InvalidRecordingSaveRequest: If the body or a field is malformed.
        :raises InvalidFrameRange: If the requested range is not integral or complete.
        """
        if not isinstance(payload, dict):
            raise InvalidRecordingSaveRequest("body must be a JSON object")
        destination = payload.get(
            RecordingSaveField.DESTINATION, SceneDestination.LOCAL
        )
        if not isinstance(destination, str) or destination not in tuple(
            SceneDestination
        ):
            raise InvalidRecordingSaveRequest("destination must be local or shared")
        first = payload.get(RecordingSaveField.FIRST_FRAME)
        last = payload.get(RecordingSaveField.LAST_FRAME)
        frame_range = None
        if first is not None or last is not None:
            if type(first) is not int or type(last) is not int:
                raise InvalidFrameRange("both frame bounds must be integers")
            frame_range = FrameRange(first=first, last=last)
        return cls(
            name=cls.optional_text(payload, RecordingSaveField.NAME) or "",
            destination=SceneDestination(destination),
            frame_range=frame_range,
            robot=cls.optional_text(payload, RecordingSaveField.ROBOT),
            environment=cls.optional_text(payload, RecordingSaveField.ENVIRONMENT),
            task=cls.optional_text(payload, RecordingSaveField.TASK),
        )

    @staticmethod
    def optional_text(
        payload: dict[str, object], field: RecordingSaveField
    ) -> str | None:
        """
        Read a text field without coercing arbitrary JSON values into scene names.

        :param payload: Decoded request object.
        :param field: Text field to validate.
        """
        value = payload.get(field)
        if value is not None and not isinstance(value, str):
            raise InvalidRecordingSaveRequest(f"{field} must be text")
        return value

    def save(self) -> str:
        """
        Save the selected frames and metadata, returning the permanent scene name.
        """
        return save_recording_bundle(
            self.name,
            self.destination,
            robot=self.robot,
            environment=self.environment,
            task=self.task,
            frame_range=self.frame_range,
        )
