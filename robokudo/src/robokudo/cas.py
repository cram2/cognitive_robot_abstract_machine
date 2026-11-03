"""
Common Analysis Structure (CAS) for RoboKudo.

This module provides the core data structure used throughout RoboKudo for storing
and managing data during pipeline execution. The CAS (Common Analysis Structure)
holds sensor data, annotations, and other information that is shared between
annotators.

The module provides:

* Standard view definitions for common data types
* Methods for storing and retrieving data
* Support for annotations and filtering
* Timestamp management
"""

# %% Imports
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum

import numpy as np
import open3d
from typing_extensions import TYPE_CHECKING, Any, TypeVar

from robokudo.exceptions import (
    CameraObservationMissing,
    ColorToDepthRatioMissing,
    PointCloudCameraModelMissing,
)

if TYPE_CHECKING:
    from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
    from semantic_digital_twin.world import World

    from robokudo.types.core import Annotation
    from robokudo.types.camera import CameraObservation

    from robokudo.types.cv import ColorToDepthRatio


# %% View definitions
class CASViews(StrEnum):
    """
    Standard view definitions for the Common Analysis Structure.

    This class defines the standard keys used to store and access different types
    of data in the CAS. These keys ensure consistent access to common data types
    like images, point clouds, and camera observations.
    """

    COLOR_IMAGE = "color_image"
    """
    RGB image data
    """

    DEPTH_IMAGE = "depth_image"
    """
    Depth image data
    """

    COLOR_TO_DEPTH_RATIO = "color_to_depth_ratio"
    """
    Size of the :attr:`DEPTH_IMAGE` relative to the :attr:`COLOR_IMAGE`, as a
    :class:`~robokudo.types.cv.ColorToDepthRatio`.
    """

    CAMERA_OBSERVATION = "camera_observation"
    """
    Semantic camera and effective projection metadata for the current frame.
    """

    POINTCLOUD_CAMERA_MODEL = "pointcloud_camera_model"
    """
    Effective pinhole model used to generate the current point cloud.
    """

    CLOUD = "cloud"
    """
    Point cloud data
    """

    QUERY = "query"
    """
    Query information
    """

    CAS_ID = "cas_id"
    """
    Monotonic ID of the CAS instance within a single pipeline run.
    type: Int
    """

    OBJECT_IMAGE = "object_image"
    """
    Object image data. This view is used in imagistic reasoning pipelines where a
    rendered scene can be fully segmented per object.
    """

    BODY_COLORS = "body_colors"
    """
    Object color mapping data which assigns objects visible in OBJECT_IMAGE to entity names.
    """

    GROUND_TRUTH_WORLD_REFERENCE = "ground_truth_world_ref"
    """
    Read-only reference to the SemDT ground-truth world used for this CAS frame.

    This view is intended for in-process consumers only and should be written
    by producers via `CAS.set_ref(...)` to avoid deep-copying the world object.
    Consumers must treat this object as strictly read-only.
    """


# %% Common analysis structure
@dataclass
class CAS:
    """
    The main data representation in RoboKudo.

    This class provides the central data structure used by annotators to store and
    retrieve information. Each pipeline has its own CAS instance that maintains
    views (singular data like sensor readings) and annotations (multiple descriptors
    of the same data).

    Views can be extended during runtime as they are supposed to be flexibly extended by different Annotators.
    However, some views are prevalent in most pipelines and can be accessed directly via @property.
    """

    timestamp: float = field(default_factory=time.time_ns)
    """
    Unix timestamp when this CAS was created. In Nanoseconds since Epoch.
    """

    timestamp_readable: str = field(init=False)
    """
    Human readable timestamp string.
    """

    views: dict[str, Any] = field(default_factory=dict)
    """
    Dictionary storing view data, each view stores data that is typically singular for a single CAS.

    Example: Sensor data, a camera observation, and a point cloud.
    """

    annotations: list[Annotation] = field(default_factory=list)
    """
    List of annotations, each annotation describes a certain part of the acquired sensor data. In contrast to views
    there can be multiple annotations for the same 'thing' in the data.
    """

    def __post_init__(self) -> None:
        dt_timestamp = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
            microseconds=self.timestamp // 1000
        )
        self.timestamp_readable = dt_timestamp.strftime("%Y-%m-%d %H:%M:%S")

    @property
    def color_image(self) -> np.ndarray | None:
        return self.views.get(CASViews.COLOR_IMAGE)

    @color_image.setter
    def color_image(self, value: np.ndarray) -> None:
        self.views[CASViews.COLOR_IMAGE] = value

    @property
    def depth_image(self) -> np.ndarray | None:
        return self.views.get(CASViews.DEPTH_IMAGE)

    @depth_image.setter
    def depth_image(self, value: np.ndarray) -> None:
        self.views[CASViews.DEPTH_IMAGE] = value

    @property
    def color_to_depth_ratio(self) -> ColorToDepthRatio:
        """
        Return required color-to-depth scaling.

        :raises ColorToDepthRatioMissing: If scaling is absent or unset.
        """
        ratio = self.views.get(CASViews.COLOR_TO_DEPTH_RATIO)
        if ratio is None:
            raise ColorToDepthRatioMissing()
        return ratio

    @color_to_depth_ratio.setter
    def color_to_depth_ratio(self, value: ColorToDepthRatio | None) -> None:
        self.views[CASViews.COLOR_TO_DEPTH_RATIO] = value

    @property
    def camera_observation(self) -> CameraObservation | None:
        """
        Return the semantic camera metadata for the current frame.
        """
        return self.views.get(CASViews.CAMERA_OBSERVATION)

    @camera_observation.setter
    def camera_observation(self, value: CameraObservation) -> None:
        """
        Store camera metadata by reference to preserve semantic identity.
        """
        self.views[CASViews.CAMERA_OBSERVATION] = value

    def require_camera_observation(self) -> CameraObservation:
        """
        Return camera metadata required by frame-dependent processing.

        :return: Camera observation for the current frame.
        :raises CameraObservationMissing: If no camera interface supplied metadata.
        """
        observation = self.camera_observation
        if observation is None:
            raise CameraObservationMissing()
        return observation

    @property
    def pointcloud_camera_model(self) -> PinholeCameraModel:
        """
        Return the calibration required for the current point cloud.

        :raises PointCloudCameraModelMissing: If point cloud calibration is absent.
        """
        camera_model = self.views.get(CASViews.POINTCLOUD_CAMERA_MODEL)
        if camera_model is None:
            raise PointCloudCameraModelMissing()
        return camera_model

    @pointcloud_camera_model.setter
    def pointcloud_camera_model(self, value: PinholeCameraModel) -> None:
        """
        Store the calibration used to generate the current point cloud.
        """
        self.views[CASViews.POINTCLOUD_CAMERA_MODEL] = value

    @property
    def cloud(self) -> open3d.geometry.PointCloud | None:
        return self.views.get(CASViews.CLOUD)

    @cloud.setter
    def cloud(self, value: open3d.geometry.PointCloud) -> None:
        self.views[CASViews.CLOUD] = value

    @property
    def query(self) -> Any | None:
        return self.views.get(CASViews.QUERY)

    @query.setter
    def query(self, value: Any) -> None:
        self.views[CASViews.QUERY] = value

    @property
    def cas_id(self) -> int | None:
        return self.views.get(CASViews.CAS_ID)

    @cas_id.setter
    def cas_id(self, value: int) -> None:
        self.views[CASViews.CAS_ID] = value

    @property
    def ground_truth_world_ref(self) -> World | None:
        return self.views.get(CASViews.GROUND_TRUTH_WORLD_REFERENCE)

    @ground_truth_world_ref.setter
    def ground_truth_world_ref(self, value: World) -> None:
        self.views[CASViews.GROUND_TRUTH_WORLD_REFERENCE] = value

    def get(self, view_name: str) -> Any:
        """
        Get a view by name.

        :param view_name: Name of the view to retrieve
        :return: The view data
        :raises KeyError: If the view does not exist
        """
        return self.views[view_name]

    def contains(self, view_name: str) -> bool:
        """
        Check if a view exists.

        :param view_name: Name of the view to check
        :return: True if the view exists
        """
        return view_name in self.views

    def get_copy(self, view_name: str) -> Any:
        """
        Get a deep copy of a view.

        :param view_name: Name of the view to copy
        :return: Deep copy of the view data
        :raises KeyError: If the view does not exist
        """
        return copy.deepcopy(self.views[view_name])

    def set(self, view_name: str, value: Any) -> None:
        """
        Put data in the CAS index by a given view name. This method will make a deepcopy of value.

        :param view_name: The name of the view which should be selected from constants in the CASView class.
        :param value: The value that will be placed in the CAS under view_name by making a deepcopy of it.
        """
        self.views[view_name] = copy.deepcopy(value)

    def set_ref(self, view_name: str, value: Any) -> None:
        """
        Put data in the CAS index by a given view name. In contrast to set(), this will not make a copy but just
        does an assignment.

        :param view_name: The name of the view which should be selected from constants in the CASView class.
        :param value: The value that will be placed in the CAS under view_name by making assigning it.
        """
        self.views[view_name] = value

    T = TypeVar("T")

    @staticmethod
    def filter_by_type(type_to_include: type[T], input_list: list[Any]) -> list[T]:
        """
        Filter a list to include only objects of a specific type.

        :param type_to_include: Type to filter for
        :param input_list: List to filter
        :return: Filtered list containing only objects of the specified type
        """
        return [
            element for element in input_list if isinstance(element, type_to_include)
        ]

    def filter_annotations_by_type(self, type_to_include: type[T]) -> list[T]:
        """
        Filter annotations to include only those of a specific type.

        :param type_to_include: Type to filter for
        :return: A filtered list of annotations in CAS
        """
        return CAS.filter_by_type(type_to_include, self.annotations)

    @staticmethod
    def _filter_objects(
        objects: list[Any], criteria: dict[str, tuple[str, Any]]
    ) -> list[Any]:
        """
        Filters a list of objects based on specified criteria.

        :param objects: List of objects to be filtered.
        :param criteria: A dictionary where keys are attribute names and values are tuples containing
                         the comparison operator as a string ("==", ">", "<", ">=", "<=") and the value to compare against.
        :return: Filtered list of objects matching all criteria
        """

        def matches_criteria(obj):
            for attr, (op, value) in criteria.items():
                attr_value = getattr(obj, attr, None)
                if not compare(attr_value, op, value):
                    return False
            return True

        def compare(attr_value, op, value):
            if op == "==":
                return attr_value == value
            elif op == ">":
                return attr_value > value
            elif op == "<":
                return attr_value < value
            elif op == ">=":
                return attr_value >= value
            elif op == "<=":
                return attr_value <= value
            else:
                raise ValueError(f"Unsupported operator: {op}")

        return [obj for obj in objects if matches_criteria(obj)]

    def filter_by_type_and_criteria(
        self,
        type_to_include: type,
        input_list: list[Any],
        criteria: dict[str, tuple[str, Any]],
    ) -> list[Any]:
        """
        Filters a list of objects based on specified criteria. Objects must be of type 'type_to_include'

        :param type_to_include: All the returned objects must be of this type.
        :param input_list: List of objects to be filtered.
        :param criteria: A dictionary where keys are attribute names and values are tuples containing
                         the comparison operator as a string ("==", ">", "<", ">=", "<=") and the value to compare against.
        :return: A list of objects of type 'type_to_include' that match all specified attribute values.
        """
        annotations = self.filter_by_type(type_to_include, input_list)
        return self._filter_objects(annotations, criteria)
