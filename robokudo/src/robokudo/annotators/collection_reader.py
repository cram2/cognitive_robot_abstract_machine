"""
Acquire sensor data from one camera interface and initialize the pipeline data structures.
"""

from __future__ import annotations

# %% Imports
from timeit import default_timer
from py_trees.common import Status
from py_trees.composites import Sequence
from py_trees.display import unicode_tree
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.exceptions import InvalidCollectionReaderParent

if TYPE_CHECKING:
    from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig
    from robokudo.io.camera_interface import CameraInterface


class CollectionReaderAnnotator(BaseAnnotator):
    """
    Acquire sensor data and share it with other annotators.
    It is typically the first annotator in the pipeline.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Configuration descriptor for collection reader.
        """

        def __init__(
            self,
            camera_config: BaseCameraConfig,
            camera_interface: CameraInterface,
        ) -> None:
            """
            Initialize descriptor with camera configuration.

            :param camera_config: Configuration for camera interface
            :param camera_interface: Interface for reading camera data
            """

            class Parameters:
                """
                Parameters for configuring collection reader.
                """

                def __init__(self) -> None:
                    self.camera_config: BaseCameraConfig = camera_config
                    """
                    Configuration for camera interface
                    """

                    self.camera_interface: CameraInterface = camera_interface
                    """
                    Interface for reading camera data
                    """

            self.parameters = Parameters()

    def __init__(
        self,
        descriptor: CollectionReaderAnnotator.Descriptor,
        name: str = "CollectionReader",
    ) -> None:
        """
        Initialize the configured camera reader and its acquisition state.
        """
        super().__init__(name, descriptor)
        self.start_timer: float | None = None
        """
        Start of the current wait for camera data.
        """
        self.iterations_since_last_data: int = 0
        """
        Number of consecutive ticks waiting for camera data.
        """
        self.iterations_since_last_data_warn_threshold: int = 40
        """
        Number of waiting ticks between camera availability warnings.
        """

    def initialise(self) -> None:
        """
        Clear feedback on the parent sequence's children before acquisition.

        :raises InvalidCollectionReaderParent: If the parent is not a sequence.
        """
        if not isinstance(self.parent, Sequence):
            raise InvalidCollectionReaderParent()
        for child in self.parent.children:
            child.feedback_message = ""

    def update(self) -> Status:
        """
        Main method that acquires sensor data and places it into the central pipeline data structures.

        Return failure when available data has no owning pipeline.
        """
        if self.start_timer is None:
            self.start_timer = default_timer()

        camera_interface = self.descriptor.parameters.camera_interface
        if not camera_interface.has_new_data():
            self.iterations_since_last_data += 1
            self.feedback_message = "Waiting for incoming sensor data ..."
            if (
                self.iterations_since_last_data
                >= self.iterations_since_last_data_warn_threshold
            ):
                self.rk_logger.warning(
                    "Didn't receive sensor data since %s iterations.",
                    self.iterations_since_last_data,
                )
                self.rk_logger.warning("Camera data available?: %s", False)
                self.iterations_since_last_data = 0
            return Status.RUNNING

        self.iterations_since_last_data = 0
        time_difference = default_timer() - self.start_timer
        self.start_timer = None
        pipeline = self.get_parent_pipeline()
        if pipeline is None:
            self.feedback_message = "Pipeline is not set!"
            return Status.FAILURE

        query = pipeline.cas.query
        pipeline.create_new_cas()
        if query is not None:
            pipeline.cas.query = query
        camera_interface.set_data(pipeline.cas)

        self.feedback_message = (
            f"Got sensor data... Waited for {time_difference} seconds"
        )
        self.rk_logger.debug(
            "New CAS id=%s after waiting %f seconds",
            pipeline.cas.cas_id,
            time_difference,
        )
        if self.parent is not None:
            self.rk_logger.info("Current Behavior Tree:\n%s", unicode_tree(self.parent))
        return Status.SUCCESS
