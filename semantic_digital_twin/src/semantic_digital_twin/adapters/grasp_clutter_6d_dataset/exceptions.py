from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from krrood.exceptions import DataclassException


@dataclass
class GraspClutter6DSceneFilesMissingError(DataclassException, LookupError):
    """
    Raised when a scene's directory does not contain both `scene_camera.json` and
    `scene_gt.json`.
    """

    scene_id: str
    """The scene id that was being loaded."""

    directory: Path
    """The directory that was searched."""

    def error_message(self) -> str:
        return (
            f"GraspClutter6D scene '{self.scene_id}' at '{self.directory}' is missing "
            f"'scene_camera.json' and/or 'scene_gt.json'."
        )

    def suggest_correction(self) -> str:
        return (
            "Make sure the scenes archive was fully extracted and that this directory "
            "is the scene's own folder, not one of its parents."
        )


@dataclass
class GraspClutter6DImageNotFoundError(DataclassException, LookupError):
    """
    Raised when a requested frame id is not part of a scene.
    """

    scene_id: str
    """The scene that was searched."""

    image_id: str
    """The frame id that could not be found."""

    def error_message(self) -> str:
        return f"Frame '{self.image_id}' not found in GraspClutter6D scene '{self.scene_id}'."

    def suggest_correction(self) -> str:
        return (
            "Check the image_id of each frame in `scene.frames`, or call "
            "`scene.frame(image_id)` to look one up."
        )


@dataclass
class GraspClutter6DObjectModelNotFoundError(DataclassException, LookupError):
    """
    Raised when an object's mesh file is not found in the given models directory.
    """

    object_id: int
    """The object id that was searched for."""

    models_directory: Path
    """The directory that was searched."""

    def error_message(self) -> str:
        return (
            f"No mesh file for object id {self.object_id} found in "
            f"'{self.models_directory}' (expected 'obj_{self.object_id:06d}.ply')."
        )

    def suggest_correction(self) -> str:
        return (
            "Download and extract one of the dataset's model archives with "
            "GraspClutter6DDatasetLoader.download_models()."
        )


@dataclass
class GraspClutter6DArchiveLayoutError(DataclassException, LookupError):
    """
    Raised when extracting one of the dataset's archives does not produce the top-level
    folder its filename is expected to create.
    """

    archive_path: Path
    """The archive that was extracted."""

    expected_directory: Path
    """The folder that was expected to exist afterward, but does not."""

    def error_message(self) -> str:
        return (
            f"Extracting '{self.archive_path}' did not create the expected "
            f"'{self.expected_directory}'."
        )

    def suggest_correction(self) -> str:
        return (
            "Inspect the archive's actual top-level contents and adjust the extraction "
            "logic - this dataset's archive-name/folder-name convention may not hold for "
            "this archive."
        )


@dataclass
class GraspClutter6DSceneNotFoundError(DataclassException, LookupError):
    """
    Raised when a scene id does not resolve to exactly one folder under the extracted
    scenes directory.
    """

    scene_id: str
    """The scene id that was searched for."""

    scenes_directory: Path
    """The extracted scenes root that was searched."""

    candidates: Tuple[Path, ...] = field(default_factory=tuple)
    """Every folder that matched, if more than one did."""

    def error_message(self) -> str:
        if self.candidates:
            return (
                f"Scene id '{self.scene_id}' matched {len(self.candidates)} folders "
                f"under '{self.scenes_directory}', not exactly one: {self.candidates}."
            )
        return f"Scene id '{self.scene_id}' not found under '{self.scenes_directory}'."

    def suggest_correction(self) -> str:
        return "Call GraspClutter6DDatasetLoader.available_scene_ids() for valid scene ids."
