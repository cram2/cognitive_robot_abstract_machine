"""
Single-reader acquisition and CAS lifecycle.
"""

from __future__ import annotations

# %% Imports
import logging
from dataclasses import dataclass
from unittest.mock import Mock
import pytest
from py_trees.common import Status
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.exceptions import InvalidCollectionReaderParent
from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig
from robokudo.io.camera_interface import CameraInterface
from robokudo.pipeline import Pipeline
from robokudo.types.core import Query


# %% Acquisition fixture
@dataclass
class ReaderPipeline:
    """
    A pipeline with a controlled source of sensor data.
    """

    reader: CollectionReaderAnnotator
    """
    Annotator receiving the controlled camera data.
    """
    pipeline: Pipeline
    """
    Pipeline owning the CAS.
    """

    camera_interface: Mock
    """
    Camera interface with controlled readiness and acquisition.
    """


@pytest.fixture
def reader_pipeline() -> ReaderPipeline:
    """
    Construct a single-reader pipeline without external camera services.
    """
    camera_interface = Mock(spec=CameraInterface)
    descriptor = CollectionReaderAnnotator.Descriptor(
        camera_config=BaseCameraConfig(interface_type="test"),
        camera_interface=camera_interface,
    )
    reader = CollectionReaderAnnotator(descriptor)
    pipeline = Pipeline(children=[reader])
    return ReaderPipeline(reader, pipeline, camera_interface)


# %% Single-reader configuration


def test_collection_reader_descriptor_uses_configured_camera_interface(
    reader_pipeline: ReaderPipeline,
) -> None:
    """
    The reader parameters retain the configured camera interface.
    """
    parameters = reader_pipeline.reader.descriptor.parameters
    assert parameters.camera_interface is reader_pipeline.camera_interface


# %% Acquisition lifecycle


def test_collection_reader_waits_after_dropped_frame(
    reader_pipeline: ReaderPipeline,
) -> None:
    """
    A rejected acquisition cannot advance downstream annotators.
    """
    reader_pipeline.camera_interface.has_new_data.return_value = True
    reader_pipeline.camera_interface.set_data.return_value = False
    original_cas = reader_pipeline.pipeline.cas
    assert reader_pipeline.reader.update() == Status.RUNNING
    assert reader_pipeline.pipeline.cas is original_cas


def test_collection_reader_preserves_timers_after_dropped_frame(
    reader_pipeline: ReaderPipeline,
) -> None:
    """
    A dropped acquisition preserves processing and acquisition wait timers.
    """
    reader_pipeline.camera_interface.has_new_data.return_value = True
    reader_pipeline.camera_interface.set_data.return_value = False
    original_timer = reader_pipeline.pipeline.cas_start_timer
    reader_pipeline.reader.start_timer = original_timer

    assert reader_pipeline.reader.update() == Status.RUNNING

    assert reader_pipeline.pipeline.cas_start_timer == original_timer
    assert reader_pipeline.reader.start_timer == original_timer


@pytest.mark.parametrize("delivery_result", [None, True])
def test_collection_reader_commits_only_successful_acquisition(
    reader_pipeline: ReaderPipeline, delivery_result: bool | None
) -> None:
    """
    Retrying a dropped frame publishes only the accepted CAS with the next identifier.
    """
    reader_pipeline.camera_interface.has_new_data.return_value = True
    reader_pipeline.camera_interface.set_data.side_effect = [False, delivery_result]
    original_cas = reader_pipeline.pipeline.cas

    assert reader_pipeline.reader.update() == Status.RUNNING
    assert reader_pipeline.reader.update() == Status.SUCCESS

    dropped_cas = reader_pipeline.camera_interface.set_data.call_args_list[0].args[0]
    accepted_cas = reader_pipeline.camera_interface.set_data.call_args_list[1].args[0]
    assert dropped_cas is not original_cas
    assert accepted_cas is not dropped_cas
    assert reader_pipeline.pipeline.cas is accepted_cas
    assert accepted_cas.cas_id == original_cas.cas_id + 1
    assert reader_pipeline.reader.start_timer is None


def test_collection_reader_waits_without_replacing_cas(
    reader_pipeline: ReaderPipeline,
) -> None:
    """
    Waiting for sensor data preserves the existing CAS.
    """
    reader_pipeline.camera_interface.has_new_data.return_value = False
    original_cas = reader_pipeline.pipeline.cas
    assert reader_pipeline.reader.update() == Status.RUNNING
    assert reader_pipeline.pipeline.cas is original_cas
    reader_pipeline.camera_interface.has_new_data.assert_called_once_with()
    reader_pipeline.camera_interface.set_data.assert_not_called()


def test_collection_reader_populates_fresh_cas(reader_pipeline: ReaderPipeline) -> None:
    """
    Available sensor data is written once to a new pipeline CAS.
    """
    reader_pipeline.camera_interface.has_new_data.return_value = True
    original_cas = reader_pipeline.pipeline.cas
    assert reader_pipeline.reader.update() == Status.SUCCESS
    assert reader_pipeline.pipeline.cas.cas_id == original_cas.cas_id + 1
    reader_pipeline.camera_interface.has_new_data.assert_called_once_with()
    reader_pipeline.camera_interface.set_data.assert_called_once_with(
        reader_pipeline.pipeline.cas
    )


def test_collection_reader_preserves_query(reader_pipeline: ReaderPipeline) -> None:
    """
    A fresh sensor CAS retains the existing query object.
    """
    reader_pipeline.camera_interface.has_new_data.return_value = True
    query = Query()
    reader_pipeline.pipeline.cas.query = query
    assert reader_pipeline.reader.update() == Status.SUCCESS
    assert reader_pipeline.pipeline.cas.query is query


def test_collection_reader_reports_wait_threshold(
    reader_pipeline: ReaderPipeline,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Reaching the wait threshold emits a warning and resets the counter.
    """
    reader_pipeline.camera_interface.has_new_data.return_value = False
    reader_pipeline.reader.iterations_since_last_data_warn_threshold = 1
    with caplog.at_level(logging.WARNING):
        assert reader_pipeline.reader.update() == Status.RUNNING
    assert reader_pipeline.reader.iterations_since_last_data == 0
    assert [record.levelno for record in caplog.records] == [
        logging.WARNING,
        logging.WARNING,
    ]


def test_collection_reader_requires_pipeline(reader_pipeline: ReaderPipeline) -> None:
    """
    Acquisition without an owning pipeline fails before writing sensor data.
    """
    reader_pipeline.camera_interface.has_new_data.return_value = True
    reader_pipeline.pipeline.remove_child(reader_pipeline.reader)
    assert reader_pipeline.reader.update() == Status.FAILURE
    reader_pipeline.camera_interface.set_data.assert_not_called()


# %% Behavior-tree initialization


def test_collection_reader_tick_waits_for_data(reader_pipeline: ReaderPipeline) -> None:
    """
    Ticking the reader initializes its behavior-tree state and waits.
    """
    reader_pipeline.camera_interface.has_new_data.return_value = False
    reader_pipeline.reader.feedback_message = "previous feedback"
    reader_pipeline.reader.tick_once()
    assert reader_pipeline.reader.status == Status.RUNNING
    reader_pipeline.camera_interface.set_data.assert_not_called()


def test_collection_reader_initialization_requires_sequence(
    reader_pipeline: ReaderPipeline,
) -> None:
    """
    Invalid parent placement raises the specific initialization error.
    """
    reader_pipeline.pipeline.remove_child(reader_pipeline.reader)
    with pytest.raises(InvalidCollectionReaderParent):
        reader_pipeline.reader.initialise()
