"""
The RoboKudo perception source against a pipeline running in its own process.

Everything else about perception is covered in ``test_perception.py`` against a stand-in
node inside the test's own interpreter, which shares an rclpy context with the client.
This is the only test where the pipeline is discovered across a real process boundary,
and so the only one that exercises what happens on the robot: cross-context discovery,
and the ``server_timeout`` budget that a locally created server never makes the client
spend.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
from rclpy.action import get_action_names_and_types

from coraplex.exceptions import PerceptionSourceUnavailable
from coraplex.perception import (
    ROBOKUDO_QUERY_ACTION_NAME,
    PerceptionQuery,
    RoboKudoPerception,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk

from coraplex.testing import StandaloneProcess

REPORTED_CLASS_LABEL = "Milk"
"""
Label the pipeline in the other process is told to report.
"""

REPORTED_POSITION = (2.6, 2.2, 1.05)
"""
Position the pipeline in the other process is told to report.
"""

MISSING_SOURCE_TIMEOUT = timedelta(seconds=2)
"""
How long to look for a pipeline that is not running.

Short enough to keep the test quick, long enough that a slow machine does not mistake
discovery latency for an absent server.
"""


@pytest.fixture
def pipeline_process_reporting(rclpy_node):
    """
    Start a perception pipeline in its own process, reporting the given label.

    Pass an empty label for a pipeline that localizes without recognizing, which is what
    the plane-and-cluster annotators of the real Stretch engine produce.
    """
    pytest.importorskip("robokudo_msgs")

    def is_serving_queries() -> bool:
        return any(
            name.lstrip("/") == ROBOKUDO_QUERY_ACTION_NAME
            for name, _ in get_action_names_and_types(rclpy_node)
        )

    started = []

    def start(class_label: str) -> StandaloneProcess:
        process = StandaloneProcess(
            launcher_path=Path(__file__).parent.parent
            / "dataset"
            / "perception_pipeline_stand_in.py",
            is_ready=is_serving_queries,
            arguments=[
                "--class-label",
                class_label,
                "--position",
                *(str(coordinate) for coordinate in REPORTED_POSITION),
            ],
        )
        process.start()
        started.append(process)
        return process

    yield start
    for process in started:
        process.stop()


@pytest.fixture
def perception_pipeline_process(pipeline_process_reporting):
    """
    A perception pipeline serving the query action from its own process.
    """
    return pipeline_process_reporting(REPORTED_CLASS_LABEL)


def test_detection_crosses_a_process_boundary(
    immutable_model_world, whole_scene_region, rclpy_node, perception_pipeline_process
):
    """
    A pipeline that shares no interpreter state with the test is discovered through the
    middleware alone, and what it reports arrives intact.
    """
    world, view, context = immutable_model_world
    query = PerceptionQuery(Milk, whole_scene_region, view, world)

    detections = RoboKudoPerception(ros_node=rclpy_node).detect(query)

    assert [detection.semantic_annotation for detection in detections] == [Milk]
    np.testing.assert_allclose(
        detections[0].pose.to_position().to_np().flatten()[:3],
        REPORTED_POSITION,
        atol=1e-9,
    )


def test_absent_perception_source_is_reported(
    immutable_model_world, whole_scene_region, rclpy_node
):
    """
    With nothing serving the action, the source has to give up and say so rather than
    block forever or return an empty result that reads like "saw nothing".
    """
    pytest.importorskip("robokudo_msgs")
    world, view, context = immutable_model_world
    query = PerceptionQuery(Milk, whole_scene_region, view, world)

    with pytest.raises(PerceptionSourceUnavailable):
        RoboKudoPerception(
            ros_node=rclpy_node, server_timeout=MISSING_SOURCE_TIMEOUT
        ).detect(query)


def test_untyped_detection_crosses_a_process_boundary(
    immutable_model_world, whole_scene_region, rclpy_node, pipeline_process_reporting
):
    """
    The shape the real Stretch engine produces: a pipeline in another process that
    localizes without recognizing, so the annotation has to come from what was asked for.
    """
    world, view, context = immutable_model_world
    pipeline_process_reporting("")
    query = PerceptionQuery(Milk, whole_scene_region, view, world)

    detections = RoboKudoPerception(ros_node=rclpy_node).detect(query)

    assert [detection.semantic_annotation for detection in detections] == [Milk]
    np.testing.assert_allclose(
        detections[0].pose.to_position().to_np().flatten()[:3],
        REPORTED_POSITION,
        atol=1e-9,
    )
