"""
Storage writer annotator for RoboKudo.

This module provides an annotator for storing sensor data in MongoDB.
It supports:

* MongoDB integration
* CAS data persistence
* View data storage
* Database management
* Configurable database settings

The module is used for:

* Data recording
* Offline processing
* Dataset creation
* Experiment logging
"""

import copy
import json
import robokudo.world
from timeit import default_timer

from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.io.storage import Storage
from semantic_digital_twin.adapters.ros.messages import WorldModelSnapshot


class StorageWriter(BaseAnnotator):
    """Annotator for storing sensor data in MongoDB.

    This annotator provides methods to store sensor data in a MongoDB database,
    allowing for data recording and offline processing without using ROS bag files.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for storage writer."""

        class Parameters:
            """Parameter container for storage configuration."""

            def __init__(self) -> None:
                self.db_name: str = "rk_scenes"
                """Database name"""

                self.drop_database_on_storage: bool = True
                """Whether to clear database before recording"""

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "StorageWriter",
        descriptor: "StorageWriter.Descriptor" = Descriptor(),
    ) -> None:
        """Initialize the storage writer. Minimal one-time init!

        :param name: Annotator name, defaults to "StorageWriter"
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)
        self.storage = Storage(self.descriptor.parameters.db_name)

        # Wipe the database completely before recording data
        if self.descriptor.parameters.drop_database_on_storage:
            self.storage.drop_database()

    def update(self) -> Status:
        """Store current CAS data in MongoDB.

        Creates a deep copy of the CAS, flattens it into a dictionary,
        and stores both views and CAS data in the database.

        :return: SUCCESS if storage successful, FAILURE otherwise
        """
        start_timer = default_timer()

        persist_cas = copy.deepcopy(self.get_cas())

        flat_cas = self.storage.generate_dict_from_real_cas(persist_cas)
        flat_cas["view_ids"] = {}

        world = robokudo.world.world_instance()
        snapshot = WorldModelSnapshot(
            modifications=list(
                world.get_world_model_manager().model_modification_blocks
            ),
            ids=list(world.state.keys()),
            states=list(world.state.positions),
        )

        payload = snapshot.to_json()
        json_str = json.dumps(payload)
        flat_cas["world"] = json_str

        # step 1: persist each view
        self.storage.store_views_in_mongo(flat_cas)

        # step 2: persist cas and pointers to each view
        del flat_cas["views"]
        result = self.storage.store_cas_dict(flat_cas)
        if not result.acknowledged:
            self.rk_logger.error(f"mongo db error when trying to store cas")
            return Status.FAILURE

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS
