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

from __future__ import annotations

import copy
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum

import numpy as np
import open3d as o3d
from sensor_msgs.msg import CameraInfo
from typing_extensions import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

from robokudo.types.tf import StampedTransform

if TYPE_CHECKING:
    from semantic_digital_twin.spatial_types.spatial_types import (
        HomogeneousTransformationMatrix,
    )
    from semantic_digital_twin.world import World

    from robokudo.types.core import Annotation


class CASViews(StrEnum):
    """Standard view definitions for the Common Analysis Structure.

    This class defines the standard keys used to store and access different types
    of data in the CAS. These keys ensure consistent access to common data types
    like images, point clouds, and camera information.
    """

    COLOR_IMAGE = "color_image"
    """RGB image data"""

    DEPTH_IMAGE = "depth_image"
    """Depth image data"""

    COLOR2DEPTH_RATIO = "color2depth_ratio"
    """Scale factor to scale the `COLOR_IMAGE` to the resolution of the `DEPTH_IMAGE` in x, y format.
    
    Example: 1280x960 RGB, 640x480 DEPTH -> 0.5 along X and Y
    """

    CAMERA_INFO = "cam_info"
    """ROS camera info message coming from ROS"""

    CAMERA_INTRINSIC = "cam_intrinsic"
    """Open3D pinhole camera intrinsic model for RGB to be set by the camera driver."""

    POINTCLOUD_CAMERA_INTRINSIC = "pc_cam_intrinsic"
    """Camera intrinsic that has been used for point cloud generation. This can be different, 
    because depth and RGB resolutions might mismatch."""

    CLOUD = "cloud"
    """Point cloud data"""

    QUERY = "query"
    """Query information"""

    WORLD_FRAME = "world_frame"
    """Name of the world frame."""

    CAMERA_FRAME = "camera_frame"
    """Name of the camera frame."""

    VIEWPOINT_CAMERA_TO_WORLD = "viewpoint_cam_to_world"
    """Deprecated: Use CAMERA_TO_WORLD_TRANSFORM instead.
    Camera to world transform.
    Type: robokudo.types.tf.StampedTransform"""

    CAMERA_TO_WORLD_TRANSFORM = "cam_to_world_transform"
    """Camera to world. 
    Type: semantic_digital_twin.spatial_types.spatial_types.HomogeneousTransformationMatrix"""

    DATA_TIMESTAMP = "data_timestamp"
    """Nanoseconds since epoch at which the sensor data has been received.
    type: Int
    """

    CAS_ID = "cas_id"
    """Monotonic ID of the CAS instance within a single pipeline run.
    type: Int
    """

    OBJECT_IMAGE = "object_image"
    """Object image data. This view is used in imagistic reasoning pipelines where a 
    rendered scene can be fully segmented per object."""

    OBJECT_COLOR_MAP = "object_color_map"
    """Object color mapping data which assigns objects visible in OBJECT_IMAGE to entity names."""

    GROUND_TRUTH_WORLD_REFERENCE = "ground_truth_world_ref"
    """Read-only reference to the SemDT ground-truth world used for this CAS frame.

    This view is intended for in-process consumers only and should be written
    by producers via `CAS.set_ref(...)` to avoid deep-copying the world object.
    Consumers must treat this object as strictly read-only.
    """


@dataclass
class CAS:
    """The main data representation in RoboKudo.

    This class provides the central data structure used by annotators to store and
    retrieve information. Each pipeline has its own CAS instance that maintains
    views (singular data like sensor readings) and annotations (multiple descriptors
    of the same data).

    Views can be extended during runtime as they are supposed to be flexibly extended by different Annotators.
    However, some views are prevalent in most pipelines and can be accessed directly via @property.
    """

    timestamp: float = field(default_factory=time.time_ns)
    """Unix timestamp when this CAS was created. In Nanoseconds since Epoch."""

    timestamp_readable: str = field(init=False)
    """Human readable timestamp string."""

    views: Dict[str, Any] = field(default_factory=dict)
    """Dictionary storing view data, each view stores data that is typically singular for a single CAS.
    
    Example: Sensor data, camera info and cloud which are read from the sensors.
    """

    annotations: List[Annotation] = field(default_factory=list)
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
    def color_image(self) -> Optional[np.ndarray]:
        return self.views.get(CASViews.COLOR_IMAGE)

    @color_image.setter
    def color_image(self, value: np.ndarray) -> None:
        self.views[CASViews.COLOR_IMAGE] = value

    @property
    def depth_image(self) -> Optional[np.ndarray]:
        return self.views.get(CASViews.DEPTH_IMAGE)

    @depth_image.setter
    def depth_image(self, value: np.ndarray) -> None:
        self.views[CASViews.DEPTH_IMAGE] = value

    @property
    def color2depth_ratio(self) -> Optional[Tuple[float, float]]:
        return self.views.get(CASViews.COLOR2DEPTH_RATIO)

    @color2depth_ratio.setter
    def color2depth_ratio(self, value: Tuple[float, float]) -> None:
        self.views[CASViews.COLOR2DEPTH_RATIO] = value

    @property
    def camera_info(self) -> Optional[CameraInfo]:
        return self.views.get(CASViews.CAMERA_INFO)

    @camera_info.setter
    def camera_info(self, value: CameraInfo) -> None:
        self.views[CASViews.CAMERA_INFO] = value

    @property
    def camera_intrinsic(self) -> Optional[o3d.camera.PinholeCameraIntrinsic]:
        return self.views.get(CASViews.CAMERA_INTRINSIC)

    @camera_intrinsic.setter
    def camera_intrinsic(self, value: o3d.camera.PinholeCameraIntrinsic) -> None:
        self.views[CASViews.CAMERA_INTRINSIC] = value

    @property
    def pointcloud_camera_intrinsic(
        self,
    ) -> Optional[o3d.camera.PinholeCameraIntrinsic]:
        return self.views.get(CASViews.POINTCLOUD_CAMERA_INTRINSIC)

    @pointcloud_camera_intrinsic.setter
    def pointcloud_camera_intrinsic(
        self, value: o3d.camera.PinholeCameraIntrinsic
    ) -> None:
        self.views[CASViews.POINTCLOUD_CAMERA_INTRINSIC] = value

    @property
    def cloud(self) -> Optional[o3d.geometry.PointCloud]:
        return self.views.get(CASViews.CLOUD)

    @cloud.setter
    def cloud(self, value: o3d.geometry.PointCloud) -> None:
        self.views[CASViews.CLOUD] = value

    @property
    def world_frame(self) -> Optional[str]:
        """Name of the world frame."""
        return self.views.get(CASViews.WORLD_FRAME)

    @world_frame.setter
    def world_frame(self, value: str) -> None:
        self.views[CASViews.WORLD_FRAME] = value

    @property
    def camera_frame(self) -> Optional[str]:
        """Name of the camera frame."""
        return self.views.get(CASViews.CAMERA_FRAME)

    @camera_frame.setter
    def camera_frame(self, value: str) -> None:
        self.views[CASViews.CAMERA_FRAME] = value

    @property
    def viewpoint_camera_to_world(self) -> Optional[StampedTransform]:
        warnings.warn(
            "CAS.viewpoint_camera_to_world is deprecated. "
            "Use CAS.camera_to_world_transform instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.views.get(CASViews.VIEWPOINT_CAMERA_TO_WORLD)

    @viewpoint_camera_to_world.setter
    def viewpoint_camera_to_world(self, value: StampedTransform) -> None:
        warnings.warn(
            "CAS.viewpoint_camera_to_world is deprecated. "
            "Use CAS.camera_to_world_transform instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.views[CASViews.VIEWPOINT_CAMERA_TO_WORLD] = value

    @property
    def camera_to_world_transform(self) -> Optional[HomogeneousTransformationMatrix]:
        return self.views.get(CASViews.CAMERA_TO_WORLD_TRANSFORM)

    @camera_to_world_transform.setter
    def camera_to_world_transform(self, value: HomogeneousTransformationMatrix) -> None:
        self.views[CASViews.CAMERA_TO_WORLD_TRANSFORM] = value

    @property
    def data_timestamp(self) -> Optional[int]:
        return self.views.get(CASViews.DATA_TIMESTAMP)

    @data_timestamp.setter
    def data_timestamp(self, value: int) -> None:
        self.views[CASViews.DATA_TIMESTAMP] = value

    @property
    def query(self) -> Optional[Any]:
        return self.views.get(CASViews.QUERY)

    @query.setter
    def query(self, value: Any) -> None:
        self.views[CASViews.QUERY] = value

    @property
    def cas_id(self) -> Optional[int]:
        return self.views.get(CASViews.CAS_ID)

    @cas_id.setter
    def cas_id(self, value: int) -> None:
        self.views[CASViews.CAS_ID] = value

    @property
    def ground_truth_world_ref(self) -> Optional[World]:
        return self.views.get(CASViews.GROUND_TRUTH_WORLD_REFERENCE)

    @ground_truth_world_ref.setter
    def ground_truth_world_ref(self, value: World) -> None:
        self.views[CASViews.GROUND_TRUTH_WORLD_REFERENCE] = value

    def get(self, view_name: str) -> Any:
        """Get a view by name.

        :param view_name: Name of the view to retrieve
        :return: The view data
        :raises KeyError: If the view does not exist
        """
        return self.views[view_name]

    def contains(self, view_name: str) -> bool:
        """Check if a view exists.

        :param view_name: Name of the view to check
        :return: True if the view exists
        """
        return view_name in self.views

    def get_copy(self, view_name: str) -> Any:
        """Get a deep copy of a view.

        :param view_name: Name of the view to copy
        :return: Deep copy of the view data
        :raises KeyError: If the view does not exist
        """
        return copy.deepcopy(self.views[view_name])

    def set(self, view_name: str, value: Any) -> None:
        """Put data in the CAS index by a given view name. This method will make a deepcopy of value.

        :param view_name: The name of the view which should be selected from constants in the CASView class.
        :param value: The value that will be placed in the CAS under view_name by making a deepcopy of it.
        """
        self.views[view_name] = copy.deepcopy(value)

    def set_ref(self, view_name: str, value: Any) -> None:
        """Put data in the CAS index by a given view name. In contrast to set(), this will not make a copy but just
        does an assignment.

        :param view_name: The name of the view which should be selected from constants in the CASView class.
        :param value: The value that will be placed in the CAS under view_name by making assigning it.
        """
        self.views[view_name] = value

    T = TypeVar("T")

    @staticmethod
    def filter_by_type(type_to_include: Type[T], input_list: List[Any]) -> List[T]:
        """Filter a list to include only objects of a specific type.

        :param type_to_include: Type to filter for
        :param input_list: List to filter
        :return: Filtered list containing only objects of the specified type
        """
        return [
            element for element in input_list if isinstance(element, type_to_include)
        ]

    def filter_annotations_by_type(self, type_to_include: Type[T]) -> List[T]:
        """
        Filter annotations to include only those of a specific type.

        :param type_to_include: Type to filter for
        :return: A filtered list of annotations in CAS
        """
        return CAS.filter_by_type(type_to_include, self.annotations)

    @staticmethod
    def _filter_objects(
        objects: List[Any], criteria: Dict[str, Tuple[str, Any]]
    ) -> List[Any]:
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
        type_to_include: Type,
        input_list: List[Any],
        criteria: Dict[str, Tuple[str, Any]],
    ) -> List[Any]:
        """Filters a list of objects based on specified criteria. Objects must be of type 'type_to_include'

        :param type_to_include: All the returned objects must be of this type.
        :param input_list: List of objects to be filtered.
        :param criteria: A dictionary where keys are attribute names and values are tuples containing
                         the comparison operator as a string ("==", ">", "<", ">=", "<=") and the value to compare against.
        :return: A list of objects of type 'type_to_include' that match all specified attribute values.
        """
        annotations = self.filter_by_type(type_to_include, input_list)
        return self._filter_objects(annotations, criteria)
