from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from typing_extensions import List, Tuple

from krrood.exceptions import DataclassException
from semantic_digital_twin.adapters.robocasa_dataset.semantics import (
    RoboCasaKitchenApplianceCategory,
    RoboCasaObjectCategory,
)


@dataclass
class RoboCasaApplianceNotFoundError(DataclassException, LookupError):
    """
    Raised when no configured RoboCasa fixture matching a requested appliance category can be found
    in any kitchen layout.
    """

    category: RoboCasaKitchenApplianceCategory
    """
    The appliance category that could not be found in any layout.
    """

    def error_message(self) -> str:
        return (
            f"No RoboCasa fixture for appliance category '{self.category}' was found in any kitchen "
            "layout."
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class RoboCasaTaskNotFoundError(DataclassException, LookupError):
    """
    Raised when a requested task name does not correspond to a registered RoboCasa kitchen task
    environment.
    """

    task_name: str
    """
    The task name that could not be resolved to a registered RoboCasa kitchen task.
    """

    available_task_names: List[str]
    """
    The names of the registered RoboCasa kitchen tasks.
    """

    def error_message(self) -> str:
        return f"No RoboCasa kitchen task named '{self.task_name}' is registered."

    def suggest_correction(self) -> str:
        return f"Choose one of the registered tasks: {', '.join(self.available_task_names)}."


@dataclass
class RoboCasaObjectAssetsNotFoundError(DataclassException, FileNotFoundError):
    """
    Raised when no downloaded asset files can be found for a requested RoboCasa object category.
    """

    category: RoboCasaObjectCategory
    """
    The object category whose assets could not be found.
    """

    searched_object_groups: Tuple[str, ...]
    """
    The object asset groups that were searched.
    """

    objects_directory: Path
    """
    The directory the object asset groups were searched under.
    """

    def error_message(self) -> str:
        return (
            f"No downloaded assets found for object category '{self.category}' in groups "
            f"{list(self.searched_object_groups)} under {self.objects_directory}."
        )

    def suggest_correction(self) -> str:
        return "Run 'python -m robocasa.scripts.download_kitchen_assets' first."


@dataclass
class RoboCasaObjectInstanceIndexError(DataclassException, IndexError):
    """
    Raised when a requested object instance index exceeds the number of downloaded instances of a
    RoboCasa object category.
    """

    category: RoboCasaObjectCategory
    """
    The object category the instance was requested for.
    """

    requested_instance_index: int
    """
    The out-of-range instance index that was requested.
    """

    available_instance_count: int
    """
    The number of downloaded instances that were found.
    """

    searched_object_groups: Tuple[str, ...]
    """
    The object asset groups that were searched.
    """

    objects_directory: Path
    """
    The directory the object asset groups were searched under.
    """

    def error_message(self) -> str:
        return (
            f"Requested instance index {self.requested_instance_index} for object category "
            f"'{self.category}', but only {self.available_instance_count} downloaded instance(s) "
            f"were found in groups {list(self.searched_object_groups)} under {self.objects_directory}."
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class RoboCasaObjectHasNoCollisionError(DataclassException, ValueError):
    """
    Raised when a loaded RoboCasa object world contains no body with collision geometry to annotate.
    """

    category: RoboCasaObjectCategory
    """
    The object category whose loaded world lacked collision geometry.
    """

    def error_message(self) -> str:
        return f"No body with collision geometry found for object category '{self.category}'."

    def suggest_correction(self) -> str:
        return ""
