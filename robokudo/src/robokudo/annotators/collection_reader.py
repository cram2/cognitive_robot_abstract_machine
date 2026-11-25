"""
Acquire sensor data from one camera interface and initialize the pipeline data
structures.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from timeit import default_timer

from py_trees.common import Status
from py_trees.composites import Sequence
from py_trees.display import unicode_tree

from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.exceptions import InvalidCollectionReaderParent
from robokudo.datastructures.annotator_descriptor import AnnotatorDescriptor
from robokudo.datastructures.annotator_parameters import AnnotatorParameters
from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig
from robokudo.io.camera_interface import CameraInterface


# %% Collection reader parameters
@dataclass(kw_only=True)
class CollectionReaderParameters(AnnotatorParameters):
    """
    Camera configuration independent of subscriptions and acquisition resources.
    """

    camera_configuration: BaseCameraConfig
    """
    Calibration, input source, and acquisition settings for the camera.
    """

    camera_interface_type: type[CameraInterface]
    """
    Acquisition implementation constructed from the camera configuration.
    """


# %% Collection reader descriptor
@dataclass(kw_only=True)
class CollectionReaderDescriptor(AnnotatorDescriptor[CollectionReaderParameters]):
    """
    Declarative camera selection for a collection reader.
    """


# %% Collection reader annotator
@dataclass(kw_only=True, eq=False)
class CollectionReaderAnnotator(BaseAnnotator[CollectionReaderDescriptor]):
    """
    Acquire sensor data and share it with other annotators.

    It is typically the first annotator in the pipeline.
    """

    descriptor: CollectionReaderDescriptor = field(repr=False)
    """
    Required camera selection and acquisition configuration.
    """

    camera_interface: CameraInterface | None = field(default=None, kw_only=True)
    """
    Acquisition interface owned by this reader.
    """

    start_timer: float | None = field(default=None, init=False, repr=False)
    """
    Start of the current wait for camera data.
    """

    iterations_since_last_data: int = field(default=0, init=False, repr=False)
    """
    Number of consecutive ticks waiting for camera data.
    """

    iterations_since_last_data_warning_threshold: int = field(
        default=40, init=False, repr=False
    )
    """
    Number of waiting ticks between camera availability warnings.
    """

    def __post_init__(self) -> None:
        """
        Specialize :meth:`BaseAnnotator.__post_init__` with the configured acquisition
        interface.
        """
        super().__post_init__()
        if self.camera_interface is None:
            self.camera_interface = self.descriptor.parameters.camera_interface_type(
                self.descriptor.parameters.camera_configuration
            )

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
        Main method that acquires sensor data and places it into the central pipeline
        data structures.

        Return failure when available data has no owning pipeline.
        """
        if self.start_timer is None:
            self.start_timer = default_timer()

        camera_interface = self.camera_interface
        if not camera_interface.has_new_data():
            self.iterations_since_last_data += 1
            self.feedback_message = "Waiting for incoming sensor data ..."
            if (
                self.iterations_since_last_data
                >= self.iterations_since_last_data_warning_threshold
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
