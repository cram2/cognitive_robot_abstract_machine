"""
MongoDB storage interface for RoboKudo.

This module provides the core interface between RoboKudo and MongoDB for storing
and retrieving sensor data and annotations. It supports:

* Serialization of complex data types
* Conversion between ROS and MongoDB formats
* Efficient binary storage of numpy arrays
* Configurable view storage and restoration
* Thread-safe database operations

The module handles:

* RGB-D camera data
* Camera calibration information
* ROS messages and transforms
* Open3D data structures
* Custom RoboKudo types
"""

from __future__ import annotations
import array
import io
import os
import sys
from typing import Optional, Dict, Union

import builtin_interfaces.msg
import numpy as np
import open3d as o3d
from pymongo import MongoClient
from sensor_msgs.msg import CameraInfo
from typing_extensions import Any, List, Type, TYPE_CHECKING, Tuple

import robokudo.analysis_engine
import robokudo.analysis_engine
import robokudo.cas
import robokudo.types.tf
import robokudo.utils.serialization as serializer
import robokudo.world
from robokudo.cas import CAS, CASViews
from robokudo.types.tf import StampedTransform
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

if TYPE_CHECKING:
    import numpy.typing as npt
    from pymongo.results import InsertOneResult
    from pymongo.synchronous.database import Database


def recursive_convert(value: Any) -> Union[Dict[Any, Any], List[Any]]:
    """
    Recursively converts non-BSON types into Python-native types.
    In particular, converts array.array and numpy.ndarray objects into lists.
    """
    # Convert Python array.array to a list
    if isinstance(value, array.array):
        return list(value)
    # Convert numpy arrays to lists
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, list):
        return [recursive_convert(item) for item in value]
    elif isinstance(value, dict):
        return {k: recursive_convert(v) for k, v in value.items()}
    else:
        return value


def ros_message_to_dict(msg: Any) -> Dict[str, Any]:
    """
    Recursively converts a ROS2 message into a dictionary using introspection.
    Ensures that any array.array or numpy.ndarray objects are converted to lists.
    """
    result: Dict[str, Any] = {}
    for field, field_type in msg.get_fields_and_field_types().items():
        value = getattr(msg, field)
        if hasattr(value, "get_fields_and_field_types"):
            result[field] = ros_message_to_dict(value)
        elif isinstance(value, list):
            result[field] = [recursive_convert(item) for item in value]
        else:
            result[field] = recursive_convert(value)
    return result


def dict_to_ros_message(message_type: Type, data_dict: Dict[Any, Any]) -> Any:
    """
    Recursively converts a dictionary into a ROS2 message of the specified type.
    This example assumes that the dictionary values are basic types.
    """
    msg = message_type()
    for field, field_type in msg.get_fields_and_field_types().items():
        if field in data_dict:
            value = data_dict[field]
            current_field = getattr(msg, field)
            if hasattr(current_field, "get_fields_and_field_types"):
                setattr(msg, field, dict_to_ros_message(type(current_field), value))
            elif isinstance(current_field, list):
                if current_field and hasattr(
                    current_field[0], "get_fields_and_field_types"
                ):
                    new_list = [
                        dict_to_ros_message(type(current_field[0]), item)
                        for item in value
                    ]
                else:
                    new_list = value
                setattr(msg, field, new_list)
            else:
                setattr(msg, field, value)
    return msg


class Storage:
    """Main interface between RoboKudo and MongoDB.

    This class holds the main interface code between RoboKudo and the MongoDB database.
    It stores sensor data and CAS views by converting specialized types (NumPy arrays,
    ROS messages, etc.) into a BSON‑encodable dictionary format.
    """

    BLACKLISTED_TYPES: Tuple[Type] = (type(lambda x: x),)  # Example: functions

    @staticmethod
    def is_blacklisted(obj: Any) -> bool:
        """
        Check if an object is of a blacklisted type.

        :param obj: Object to check
        :return: True if object type is blacklisted, False otherwise
        """
        return isinstance(obj, Storage.BLACKLISTED_TYPES)

    @staticmethod
    def instantiate_mongo_client() -> MongoClient:
        """
        Create a MongoDB client instance.

        Uses environment variables RK_MONGO_HOST and RK_MONGO_PORT if set,
        otherwise defaults to localhost:27017.

        :return: MongoDB client instance
        """
        # Fetch environment variables which might have been used to configure
        # another MongoDB Host and Port
        # This was originally introduced to support unit tests.
        mongo_host = os.getenv("RK_MONGO_HOST", "localhost")
        mongo_port = int(os.getenv("RK_MONGO_PORT", 27017))

        return MongoClient(host=mongo_host, port=mongo_port)

    @DeprecationWarning  # Use ListReader instead for better pymongo compatibility across versions
    class Reader:
        """
        Deprecated cursor-based MongoDB reader.

        .. deprecated:: Use ListReader instead for better pymongo compatibility
        """

        def __init__(self, db_name: str) -> None:
            """Initialize the reader.

            :param db_name: Name of the MongoDB database
            """
            client = Storage.instantiate_mongo_client()

            self.db_reader = client[db_name]
            self.reset_cursor()

        def reset_cursor(self) -> None:
            """Reset the cursor to the start of the collection."""
            self.cursor = self.db_reader.cas.find()

        def collection_has_frames(self) -> bool:
            """Check if collection has any frames.

            :return: True if frames exist, False otherwise
            """
            return self.db_reader.cas.find().alive

        def cursor_has_frames(self) -> bool:
            """Check if cursor has more frames.

            :return: True if more frames exist, False otherwise
            """
            return self.cursor.alive

        def get_next_frame(self) -> Optional[dict]:
            """
            Get the next frame from the cursor.

            :return: Next frame data or None if no more frames
            """
            if not self.cursor_has_frames():
                return None
            return self.cursor.next()

    class ListReader:
        """List-based MongoDB reader.

        This class reads all matching records into a list for iteration,
        providing better compatibility across pymongo versions and simpler
        cursor management.
        """

        def __init__(self, db_name: str) -> None:
            """Initialize the list reader.

            :param db_name: Name of the MongoDB database
            """
            client = Storage.instantiate_mongo_client()

            self.db_reader = client[db_name]
            """MongoDB database instance"""

            self.index: Optional[int] = None
            """Current position in data list"""

            self.data: List[Dict[Any, Any]] = []
            """List of loaded documents"""

            self.reset_cursor()

        def reset_cursor(self) -> None:
            """Reset the reader state.

            Clears and reloads all documents from the database.
            """
            self.data.clear()
            for data in self.db_reader.cas.find():
                self.data.append(data)
            self.index = 0 if self.data else None

        def cursor_has_frames(self) -> bool:
            """Check if more frames are available.

            :return: True if more frames exist, False otherwise
            """
            return self.index is not None and self.index < len(self.data)

        def get_next_frame(self) -> Optional[Dict[Any, Any]]:
            """Get the next frame from the data list.

            :return: Next frame data or None if no more frames
            """
            if not self.cursor_has_frames():
                return None
            data = self.data[self.index]
            self.index += 1
            return data

    def __init__(self, db_name: str) -> None:
        """Initialize the storage interface.

        :param db_name: Name of the MongoDB database
        """

        self.db_name = db_name
        """Name of the MongoDB database"""

        self.client: MongoClient = Storage.instantiate_mongo_client()
        """MongoDB client connection"""

        self.db: Database = self.client[db_name]
        """MongoDB database instance"""

    def drop_database(self) -> None:
        """Drop the entire database."""
        self.client.drop_database(self.db_name)

    @staticmethod
    def nd_array_to_numpy_binary(arr: npt.NDArray) -> bytes:
        """Convert numpy array to binary format for storage.

        :param arr: Numpy array to convert
        :return: Binary representation of array
        """
        memfile = io.BytesIO()
        np.save(memfile, arr)
        return memfile.getvalue()

    @staticmethod
    def numpy_binary_to_nd_array(binary: bytes) -> npt.NDArray:
        """Convert binary data back to numpy array.

        :param bin: Binary data to convert
        :return: Reconstructed numpy array
        """
        memfile = io.BytesIO()
        memfile.write(binary)
        memfile.seek(0)
        return np.load(memfile)

    # Conversion functions for ROS messages and transforms

    @staticmethod
    def ros_cam_info_to_mongo(cam_info: CameraInfo) -> Dict[Any, Any]:
        """Convert ROS camera info to MongoDB format.

        Convert a ROS2 CameraInfo message into a dictionary using our custom introspection,
        ensuring that all non-BSON types (numpy.ndarray, array.array) are converted.

        :param cam_info: ROS camera info message
        :return: Dictionary representation of camera info
        """
        return ros_message_to_dict(cam_info)

    @staticmethod
    def ros_cam_info_from_mongo(mongo_cam_info: Dict[Any, Any]) -> CameraInfo:
        """Convert a dictionary from MongoDB back into a ROS2 CameraInfo message.

        :param mongo_cam_info: Dictionary representation of camera info
        :return: ROS camera info message
        """
        return dict_to_ros_message(CameraInfo, mongo_cam_info)

    @staticmethod
    def camera_intrinsic_to_mongo(
        camera_intrinsic: o3d.camera.PinholeCameraIntrinsic,
    ) -> Dict[str, Union[int, float]]:
        intrinsic_matrix = camera_intrinsic.intrinsic_matrix  # 3 x 3
        result = {
            "width": camera_intrinsic.width,
            "height": camera_intrinsic.height,
            "fx": intrinsic_matrix[0, 0],
            "fy": intrinsic_matrix[1, 1],
            "cx": intrinsic_matrix[0, 2],
            "cy": intrinsic_matrix[1, 2],
        }
        return result

    @staticmethod
    def camera_intrinsic_from_mongo(
        camera_intrinsic_dict: Dict[str, Union[int, float]],
    ) -> o3d.camera.PinholeCameraIntrinsic:
        return o3d.camera.PinholeCameraIntrinsic(**camera_intrinsic_dict)

    @staticmethod
    def rk_stamped_transform_to_mongo(
        stamped_transform: StampedTransform,
    ) -> Dict[str, Any]:
        """Convert RoboKudo stamped transform to MongoDB format.

        :param stamped_transform: RoboKudo stamped transform
        :return: Dictionary representation of transform
        """
        result = stamped_transform.__dict__
        time = result["timestamp"]  # this is a ros type that has no __dict__
        result["timestamp"] = {
            "secs": time.sec,
            "nsecs": time.nanosec,
        }
        return result

    @staticmethod
    def rk_stamped_transform_from_mongo(
        stamped_transform_dict: Dict[str, Any],
    ) -> StampedTransform:
        """Convert MongoDB transform back to RoboKudo format.

        :param stamped_transform_dict: Dictionary representation of transform
        :return: RoboKudo stamped transform
        """
        st = robokudo.types.tf.StampedTransform()
        st.rotation = stamped_transform_dict["rotation"]
        st.translation = stamped_transform_dict["translation"]
        st.frame = stamped_transform_dict["frame"]
        st.child_frame = stamped_transform_dict["child_frame"]
        st.timestamp = builtin_interfaces.msg.Time(
            sec=stamped_transform_dict["timestamp"]["secs"],
            nanosec=stamped_transform_dict["timestamp"]["nsecs"],
        )
        return st

    @staticmethod
    def homogeneous_transform_matrix_to_mongo(
        matrix: HomogeneousTransformationMatrix,
    ) -> Dict[str, Any]:
        """Convert a stored HomogeneousTransformationMatrix to a dict.

        :param matrix: The transform matrix
        :return: Dictionary representation of transform
        """
        return matrix.to_json()

    @staticmethod
    def homogeneous_transform_matrix_from_mongo(
        matrix_dict: Dict[str, Any],
    ) -> HomogeneousTransformationMatrix:
        """Convert a stored HomogeneousTransformationMatrix back.

        To reconstruct the references to the KinematicStructureEntities,
        we need access to a WorldEntityTracker that has tracked the restoration of the World that was
        also stored in the Mongo DB.

        :param matrix_dict: The transform matrix
        :return: The restored HomogeneousTransformationMatrix from matrix_dict
        """
        tracker = robokudo.world.get_world_entity_tracker()
        kwargs = tracker.create_kwargs()
        return HomogeneousTransformationMatrix.from_json(matrix_dict, **kwargs)

    # TODO Have a separate module for conversion handling OR ormatic handles everything anyway in the future

    # TODO This should be overwritable in the Descriptor/Config. Atleast the Views you want to save/restore.
    view_configuration = {
        CASViews.COLOR_IMAGE: {
            "to_mongo": lambda x: Storage.nd_array_to_numpy_binary(x),
            "from_mongo": lambda x: Storage.numpy_binary_to_nd_array(x),
        },
        CASViews.DEPTH_IMAGE: {
            "to_mongo": lambda x: Storage.nd_array_to_numpy_binary(x),
            "from_mongo": lambda x: Storage.numpy_binary_to_nd_array(x),
        },
        CASViews.COLOR2DEPTH_RATIO: {
            "from_mongo": lambda x: tuple(
                x
            ),  # mongodb will serialize and load to lists
        },
        CASViews.CAM_INFO: {
            "to_mongo": lambda x: Storage.ros_cam_info_to_mongo(x),
            "from_mongo": lambda x: Storage.ros_cam_info_from_mongo(x),
        },
        CASViews.CAM_INTRINSIC: {
            "to_mongo": lambda x: Storage.camera_intrinsic_to_mongo(x),
            "from_mongo": lambda x: Storage.camera_intrinsic_from_mongo(x),
        },
        CASViews.VIEWPOINT_CAM_TO_WORLD: {
            "to_mongo": lambda x: Storage.rk_stamped_transform_to_mongo(x),
            "from_mongo": lambda x: Storage.rk_stamped_transform_from_mongo(x),
        },
        CASViews.CAM_TO_WORLD_TRANSFORM: {
            "to_mongo": lambda x: Storage.homogeneous_transform_matrix_to_mongo(x),
            "from_mongo": lambda x: Storage.homogeneous_transform_matrix_from_mongo(x),
        },
        CASViews.DATA_TIMESTAMP: {
            "to_mongo": lambda x: x,
            "from_mongo": lambda x: int(x),
        },
    }

    def store_views_in_mongo(self, cas_dict: Dict[str, Any]) -> None:
        """Store CAS views in MongoDB.

        Store the views present in cas.views.
        Please note that this method will change cas. So make a deepcopy if you want to leave your
        true CAS untouched!

        :param cas_dict: CAS as a dictionary to persist
        """
        for view in Storage.view_configuration:
            if view not in cas_dict["views"]:
                print(
                    f'You want to store view "{view}" but it is not in the CAS. Ignoring...'
                )
                continue

            mongo_view_data = cas_dict["views"][view]
            if mongo_view_data is not None:
                # get a transform function or get identity function if no special function is set
                to_mongo_fun = Storage.view_configuration[view].get(
                    "to_mongo", lambda x: x
                )
                mongo_view_data = to_mongo_fun(mongo_view_data)

            # Construct the document (record) we want to save in the DB
            mongo_view_document = {"data": mongo_view_data}
            result = self.db[view].insert_one(mongo_view_document)
            if not result.acknowledged:
                print(f'Mongo DB Error when trying to store view "{view}"')
                sys.exit(1)

            cas_dict["view_ids"][view] = result.inserted_id

    def load_views_from_mongo_in_cas(self, cas_document: Dict[str, Any]) -> None:
        """Load views from MongoDB into a CAS document.

        Retrieves and converts each view referenced in the CAS document from its
        MongoDB representation back to its original format.

        :param cas_document: CAS document to update with loaded views
        """
        for view_name, view_id in cas_document["view_ids"].items():
            view_document = self.db[view_name].find_one({"_id": view_id})
            if not view_document:
                print(f"Couldn't find View '{view_name}' with id={view_id}")
                sys.exit(1)
            from_mongo_fun = Storage.view_configuration[view_name].get(
                "from_mongo", lambda x: x
            )
            view_data = from_mongo_fun(view_document["data"])

            view_data = view_document["data"]
            if view_data is not None:
                # Put the view document into the CAS we are restoring - Respect transform method if present
                from_mongo_fun = Storage.view_configuration[view_name].get(
                    "from_mongo", lambda x: x
                )
                view_data = from_mongo_fun(view_data)

            cas_document["views"][view_name] = view_data

    @staticmethod
    def load_annotations_from_mongo_in_cas(
        cas_document: Dict[str, Any], cas: CAS
    ) -> None:
        """Load annotations from MongoDB into a CAS.

        Restore the annotations from the database and insert them into the CAS.
        If no (pickled) annotations are available, cas will be untouched.

        :param cas_document: A dict representing a frame of a CAS in the database
        :param cas: The 'robokudo.cas.CAS' instance where the annotations shall be inserted to
        """
        if cas_document["annotations"]:
            cas.annotations = serializer.unflatten(cas_document["annotations"])

    @staticmethod
    def generate_dict_from_real_cas(cas: CAS) -> Dict[str, Any]:
        """Convert a CAS instance to a MongoDB-compatible dictionary.

        Generate dict that we can put in mongo from CAS

        :param cas: Input CAS that should be used to create a dict-representation of it.
        :return: A dict with references to parts of the input CAS.
        """
        serialized_annotations = serializer.flatten(cas.annotations)

        result = {
            "timestamp": cas.timestamp,
            "timestamp_readable": cas.timestamp_readable,
            "annotations": serialized_annotations,
            "views": cas.views,
        }

        return result

    def store_cas_dict(self, cas_dict: Dict[str, Any]) -> InsertOneResult:
        """
        Store a CAS dictionary in MongoDB.

        Stores the views and creates a CAS document in MongoDB that references
        them.

        :param cas_dict: Dictionary representation of a CAS
        :return: MongoDB ObjectId of the stored CAS document
        """
        return self.db.cas.insert_one(cas_dict)
