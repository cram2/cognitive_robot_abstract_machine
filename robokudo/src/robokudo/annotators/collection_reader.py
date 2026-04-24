"""Sensor data collection and CAS initialization.

This module provides an annotator for:

* Reading sensor data from multiple interfaces
* Initializing the CAS with sensor data
* Managing multiple collection readers
* Handling data synchronization

The module uses:

* Camera interface abstraction
* Pipeline CAS management
* Data availability monitoring
* Query preservation

.. note::
   This is typically the first annotator in a pipeline.
"""

from timeit import default_timer as timer
from typing_extensions import Optional, List

from py_trees.common import Status
from py_trees.composites import Sequence
from py_trees.display import unicode_tree

from robokudo.annotators.core import BaseAnnotator
from robokudo.io.camera_interface import CameraInterface
from robokudo.cas import CASViews
from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig
import robokudo.world


class CollectionReaderAnnotator(BaseAnnotator):
    """Sensor data collection and CAS initialization.

    This annotator:

    * Reads sensor data from camera interfaces
    * Initializes new CAS instances
    * Manages multiple collection readers
    * Monitors data availability
    * Preserves existing queries

    .. note::
       Uses separate camera interfaces for different sensor types.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for collection reader."""

        def __init__(
            self,
            camera_config: BaseCameraConfig,
            camera_interface: CameraInterface,
        ) -> None:
            """Initialize descriptor with camera configuration.

            :param camera_config: Configuration for camera interface
            :param camera_interface: Interface for reading camera data
            """

            class Parameters:
                """Parameters for configuring collection reader."""

                def __init__(self) -> None:
                    self.camera_config = camera_config
                    """Configuration for camera interface"""

                    self.camera_interface = camera_interface
                    """Interface for reading camera data"""

            self.parameters = Parameters()

    def __init__(
        self,
        descriptor: "CollectionReaderAnnotator.Descriptor",
        name: str = "CollectionReader",
    ) -> None:
        """Initialize the collection reader.

        :param descriptor: Configuration descriptor
        :param name: Name of this annotator instance, defaults to "CollectionReader"
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

        self.collection_readers: List["CollectionReaderAnnotator.Descriptor"] = [
            descriptor
        ]
        self.start_timer: Optional[float] = None
        self.iterations_since_last_data: int = 0
        self.iterations_since_last_data_warn_threshold: int = 40

    def initialise(self) -> None:
        """Initialize the collection reader.

        Called when:

        * First tick is received
        * Status changes from non-running

        Clears feedback messages for all children.
        """
        self.rk_logger.debug("%s.initialise()" % self.__class__.__name__)

        # Clear all feedback messages when Collection Reader starts over
        assert isinstance(self.parent, Sequence)
        for child in self.parent.children:
            child.feedback_message = ""

    def add_collection_reader(
        self, descriptor: "CollectionReaderAnnotator.Descriptor"
    ) -> None:
        """Add another collection reader descriptor.

        :param descriptor: Configuration descriptor for additional reader
        """
        interface_type = descriptor.parameters.camera_interface.interface_type
        self.collection_readers.append(descriptor)
        self.logger.debug(
            f"added descriptor to collection readers with camera interface '{interface_type}'"
        )

    def update(self) -> Status:
        """Process sensor data and update CAS.

        The method:

        * Checks for new data from all readers
        * Creates new CAS when data available
        * Preserves existing queries
        * Updates feedback messages
        * Monitors data availability

        :return: SUCCESS if data processed, RUNNING if waiting
        """
        if self.start_timer is None:
            self.start_timer = timer()

        all_crs_have_data = True
        for collection_reader in self.collection_readers:
            all_crs_have_data = (
                all_crs_have_data
                & collection_reader.parameters.camera_interface.has_new_data()
            )

        if all_crs_have_data:
            self.iterations_since_last_data = 0
            self.rk_logger.debug(
                "%s.update(): All CRs have new data! Creating a new CAS."
                % self.__class__.__name__
            )
            end_timer = timer()
            time_difference = end_timer - self.start_timer
            self.rk_logger.debug("Waited for %f seconds \n" % time_difference)
            self.start_timer = None
            pipeline = self.get_parent_pipeline()

            # Special case: If there is already a query present in this CAS, we need to temporarily save it and set
            # it in the new CAS
            query = None
            if pipeline.cas.contains(CASViews.QUERY):
                query = pipeline.cas.get(CASViews.QUERY)

            # Create a fresh CAS for the pipeline
            pipeline.create_new_cas()
            self.rk_logger.debug(
                f"{self.__class__.__name__}.update(): New CAS id={pipeline.cas.cas_id}"
            )

            # Create a fresh world
            robokudo.world.clear_world()

            # Restore any existing queries
            if query:
                pipeline.cas.set(robokudo.cas.CASViews.QUERY, query)

            for collection_reader in self.collection_readers:
                collection_reader.parameters.camera_interface.set_data(self.get_cas())

            self.feedback_message = (
                f"Got sensor data... Waited for {time_difference} seconds"
            )
            if self.parent:
                self.rk_logger.info("Current Behavior Tree:")
                print(unicode_tree(self.parent))

            return Status.SUCCESS

        else:
            self.iterations_since_last_data += 1
            self.feedback_message = "Waiting for incoming sensor data ..."
            if (
                self.iterations_since_last_data
                >= self.iterations_since_last_data_warn_threshold
            ):
                self.rk_logger.warning(
                    f"Didn't receive sensor data since {self.iterations_since_last_data} iterations."
                )
                for collection_reader in self.collection_readers:
                    self.rk_logger.warning(
                        f"  CR data available?: {collection_reader.parameters.camera_interface.has_new_data()}"
                    )
                self.iterations_since_last_data = 0
            return Status.RUNNING

    def terminate(self, new_status: Status) -> None:
        """Handle behavior termination.

        :param new_status: New status (SUCCESS, FAILURE or INVALID)
        """
        self.rk_logger.debug(
            "%s.terminate()[%s->%s]"
            % (self.__class__.__name__, self.status, new_status)
        )


"""ROS1 TO ROS2 
The setup method in the ROS2 version has three parameters (timeout, node, visitor) 
n the ROS2 version, STATUS are referenced as py_trees.common.Status.RUNNING and py_trees.common.Status.SUCCESS
"""
