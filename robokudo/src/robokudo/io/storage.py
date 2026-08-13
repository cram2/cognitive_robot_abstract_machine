"""
MongoDB storage interface for RoboKudo.

This module provides the core interface between RoboKudo and MongoDB for storing
and retrieving sensor data and annotations. It supports:

* Codec-driven serialization of CAS views
* KRROOD-backed annotation serialization
* Configurable view persistence boundaries
* Thread-safe database operations

The module handles:

* RGB-D camera data
* Camera calibration and transform objects
* Custom RoboKudo types
"""

from __future__ import annotations
import os

from pymongo import MongoClient
from typing_extensions import Any, Dict, List, Optional, TYPE_CHECKING, Tuple
from robokudo import world

from robokudo.cas import CAS
from robokudo.io.cas_annotation_codecs import (
    serialize_annotations,
    deserialize_annotations,
)
from robokudo.io.cas_view_codecs import CASViewCodecRegistry

if TYPE_CHECKING:
    from pymongo.results import InsertOneResult
    from pymongo.synchronous.database import Database


class Storage:
    """Main interface between RoboKudo and MongoDB.

    This class holds the main interface code between RoboKudo and the MongoDB database.
    It stores sensor data and CAS views by converting specialized types (NumPy arrays,
    ROS messages, etc.) into a BSON‑encodable dictionary format.
    """

    BLACKLISTED_TYPES: Tuple[type, ...] = (type(lambda x: x),)
    VIEW_COLLECTION_NAME: str = "cas_views"
    cas_view_codecs = CASViewCodecRegistry()

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
    def encode_view_document(view_name: str, view_value: Any) -> Dict[str, Any]:
        """Encode one CAS view value into a serializable view document."""
        document = Storage.cas_view_codecs.encode_view(
            view_name=view_name,
            value=view_value,
            strict=True,
        )
        if document is None:
            raise TypeError(
                f"No codec registered for view '{view_name}' with type '{type(view_value)}'."
            )
        return document

    @staticmethod
    def decode_view_document(view_document: Dict[str, Any]) -> tuple[str, Any]:
        """Decode one serialized view document into a CAS view value."""
        return Storage.cas_view_codecs.decode_view(view_document)

    def store_views_in_mongo(self, cas_dict: Dict[str, Any]) -> None:
        """Store CAS views in MongoDB.

        Persist CAS views in a dedicated view collection and store references
        to those view records in ``cas_dict["view_ids"]``.

        :param cas_dict: CAS as a dictionary to persist
        """
        for view_name, view_value in cas_dict["views"].items():
            view_document = Storage.encode_view_document(
                view_name=view_name,
                view_value=view_value,
            )
            result = self.db[Storage.VIEW_COLLECTION_NAME].insert_one(view_document)
            if not result.acknowledged:
                raise RuntimeError(
                    f'Mongo DB error when trying to store view "{view_name}".'
                )

            cas_dict["view_ids"][view_name] = result.inserted_id

    def load_views_from_mongo_in_cas(self, cas_document: Dict[str, Any]) -> None:
        """Load views from MongoDB into a CAS document.

        Retrieve and decode each persisted view referenced in ``view_ids``.

        :param cas_document: CAS document to update with loaded views
        """
        for expected_view_name, view_id in cas_document["view_ids"].items():
            view_document = self.db[Storage.VIEW_COLLECTION_NAME].find_one(
                {"_id": view_id}
            )
            if not view_document:
                raise RuntimeError(
                    f"Couldn't find view '{expected_view_name}' with id={view_id}."
                )
            decoded_view_name, decoded_view_value = Storage.decode_view_document(
                view_document
            )
            if decoded_view_name != expected_view_name:
                raise RuntimeError(
                    f"Decoded view name '{decoded_view_name}' does not match expected "
                    f"view name '{expected_view_name}' for id={view_id}."
                )
            cas_document["views"][decoded_view_name] = decoded_view_value

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
            tracker = world.get_world_entity_tracker()
            kwargs = tracker.create_kwargs() if tracker is not None else {}
            cas.annotations = deserialize_annotations(
                cas_document["annotations"], **kwargs
            )

    @staticmethod
    def generate_dict_from_real_cas(cas: CAS) -> Dict[str, Any]:
        """Convert a CAS instance to a MongoDB-compatible dictionary.

        Generate a dictionary representation from CAS that is ready for storage.
        All views are validated through the active view codec registry.

        :param cas: Input CAS that should be used to create a dict-representation of it.
        :return: A dict with references to parts of the input CAS.
        """
        serialized_annotations = serialize_annotations(cas.annotations)
        for view_name, view_value in cas.views.items():
            Storage.encode_view_document(view_name=view_name, view_value=view_value)

        result = {
            "timestamp": cas.timestamp,
            "timestamp_readable": cas.timestamp_readable,
            "annotations": serialized_annotations,
            "annotations_format": "krrood_v1",
            "views": dict(cas.views),
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
