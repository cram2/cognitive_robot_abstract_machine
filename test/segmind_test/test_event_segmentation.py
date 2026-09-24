"""
Tests for a run stating what it wants watched, what it wants detected and whether it
shows the events while it goes on.
"""

from __future__ import annotations

import threading
from collections import Counter

import pytest

from segmind.datastructures.events import TranslationEvent
from segmind.detector_selection import DetectorSelection
from segmind.detectors.coarse_event_detector_nodes import PickUpDetector
from segmind.event_segmentation import Segmind
from segmind.exceptions import NoSemanticAnnotationToWatch
from semantic_digital_twin.semantic_annotations.semantic_annotations import Food, Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

from .conftest import RESTING_ON_THE_TABLE

TICK_TIMEOUT = 10.0
"""
Seconds a test waits for the watching thread to have detected something.
"""

MOVED_ALONG_X = 0.2
"""
How far a test moves the milk while the run is watched.
"""


def _stand_the_milk_on_the_table(milk) -> None:
    """
    Put the milk where it rests on the table, so a move of it is a move from rest.
    """
    rest_x, rest_y, rest_z = RESTING_ON_THE_TABLE
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        rest_x, rest_y, rest_z, reference_frame=milk.parent_connection.parent
    )


def test_a_run_watches_the_bodies_of_every_annotation_of_a_type_it_names(
    milk_annotated_in_the_apartment,
):
    world, milk, _ = milk_annotated_in_the_apartment

    segmentation = Segmind.create_for_semantic_annotation_types(world, [Milk])

    assert segmentation.bodies == milk.bodies


def test_a_body_annotated_by_two_of_the_types_named_is_watched_once(
    milk_annotated_in_the_apartment,
):
    world, milk, _ = milk_annotated_in_the_apartment

    segmentation = Segmind.create_for_semantic_annotation_types(world, [Milk, Food])

    assert segmentation.bodies == milk.bodies


def test_a_run_cannot_watch_a_type_the_world_holds_no_annotation_of(
    milk_in_the_apartment,
):
    world, _, _ = milk_in_the_apartment

    with pytest.raises(NoSemanticAnnotationToWatch) as raised:
        Segmind.create_for_semantic_annotation_types(world, [Milk])

    assert raised.value.semantic_annotation_type is Milk


def test_a_run_is_given_every_detector_what_it_asks_for_is_read_from(
    milk_in_the_apartment,
):
    """
    A run says what it wants detected; the detectors that is concluded from come with
    it, and the kinds ticked are named once each.
    """
    world, milk, _ = milk_in_the_apartment

    segmentation = Segmind(world=world, bodies=[milk], detectors=[PickUpDetector])

    assert Counter(segmentation.detector_names) == Counter(
        detector_type.__name__
        for detector_type in DetectorSelection.of(PickUpDetector).detector_types
    )


def test_what_happens_while_a_run_is_watched_is_detected(milk_in_the_apartment):
    world, milk, _ = milk_in_the_apartment
    _stand_the_milk_on_the_table(milk)
    segmentation = Segmind(world=world, bodies=[milk])
    translated = threading.Event()
    segmentation.segmenter.event_logger.add_callback(
        TranslationEvent, lambda event: translated.set()
    )

    with segmentation:
        rest_x, rest_y, rest_z = RESTING_ON_THE_TABLE
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            rest_x + MOVED_ALONG_X,
            rest_y,
            rest_z,
            reference_frame=milk.parent_connection.parent,
        )
        seen = translated.wait(TICK_TIMEOUT)

    assert seen
    [translation] = [
        event
        for event in segmentation.segmenter.event_logger.get_events()
        if isinstance(event, TranslationEvent)
    ]
    assert translation.tracked_object is milk
