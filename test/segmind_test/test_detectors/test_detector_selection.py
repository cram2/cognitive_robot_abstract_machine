"""
Tests for choosing the detectors a run uses: asking for what is to be detected brings
along every detector it is read from, and every detector brings along the one reporting
that what it reports has ended.
"""

from __future__ import annotations

from dataclasses import dataclass

from segmind.detectors.base import AbstractDetector
from segmind.detector_selection import DetectorSelection
from segmind.detectors.atomic_event_detectors_nodes import (
    ContactDetector,
    LossOfContactDetector,
    StopTranslationDetector,
    TranslationDetector,
)
from segmind.detectors.coarse_event_detector_nodes import (
    PickUpDetector,
    PlacingDetector,
)
from segmind.detectors.agent_event_detector_nodes import (
    AbstractGraspDetector,
    GraspDetector,
    LossOfGraspDetector,
)
from segmind.detectors.spatial_relation_detector_nodes import (
    ContainmentDetector,
    InsertionDetector,
    LossOfContainmentDetector,
    LossOfSupportDetector,
    SupportDetector,
)


@dataclass(eq=False, repr=False)
class DetectorDefinedOutsideSegMind(AbstractDetector):
    """
    A kind of detector some other package defines.
    """

    def update_context_and_events(self, context, segmind_context, tracked_objects):
        return []


PICK_UP_AND_WHAT_IT_IS_READ_FROM = {
    PickUpDetector,
    SupportDetector,
    LossOfSupportDetector,
    TranslationDetector,
    StopTranslationDetector,
}
"""
The detectors a run asking only for pick-ups uses.
"""


# %% what is brought along


def test_asking_for_pick_ups_brings_what_they_are_read_from():
    selection = DetectorSelection.of(PickUpDetector)

    assert set(selection.detector_types) == PICK_UP_AND_WHAT_IT_IS_READ_FROM


def test_grasping_is_brought_along_only_when_asked_for():
    """
    A pick-up can be read from a grasp, but a run that did not ask to watch for grasps
    reads it from the object's own motion.
    """
    selection = DetectorSelection.of(PickUpDetector, GraspDetector)

    assert set(selection.detector_types) == PICK_UP_AND_WHAT_IT_IS_READ_FROM | {
        GraspDetector,
        LossOfGraspDetector,
    }


def test_insertions_bring_contact_and_containment():
    selection = DetectorSelection.of(InsertionDetector)

    assert set(selection.detector_types) == {
        InsertionDetector,
        ContactDetector,
        LossOfContactDetector,
        ContainmentDetector,
        LossOfContainmentDetector,
    }


# %% counterparts


def test_a_detector_brings_the_one_reporting_its_end():
    assert set(DetectorSelection.of(ContactDetector).detector_types) == {
        ContactDetector,
        LossOfContactDetector,
    }


def test_a_detector_reporting_an_end_brings_the_one_reporting_the_beginning():
    assert set(DetectorSelection.of(LossOfContactDetector).detector_types) == {
        ContactDetector,
        LossOfContactDetector,
    }


# %% the order they tick in


def test_a_detector_ticks_after_what_it_is_read_from():
    order = DetectorSelection.of(PickUpDetector).detector_types

    assert order.index(PickUpDetector) > order.index(SupportDetector)
    assert order.index(PickUpDetector) > order.index(TranslationDetector)


def test_each_kind_of_detector_is_chosen_once():
    order = DetectorSelection.of(PickUpDetector, PlacingDetector).detector_types

    assert len(order) == len(set(order))


# %% everything SegMind can detect


def test_everything_holds_every_kind_of_detector_segmind_defines():
    kinds = set(DetectorSelection.of_every_kind().detector_types)

    assert {
        PickUpDetector,
        PlacingDetector,
        InsertionDetector,
        GraspDetector,
        LossOfGraspDetector,
        ContactDetector,
        LossOfContactDetector,
        TranslationDetector,
        StopTranslationDetector,
    } <= kinds


def test_everything_holds_no_kind_that_is_abstract_or_defined_elsewhere():
    kinds = set(DetectorSelection.of_every_kind().detector_types)

    assert AbstractDetector not in kinds
    assert AbstractGraspDetector not in kinds
    assert DetectorDefinedOutsideSegMind not in kinds


def test_everything_ticks_each_kind_after_what_it_is_read_from():
    order = DetectorSelection.of_every_kind().detector_types

    assert order.index(PickUpDetector) > order.index(SupportDetector)
    assert order.index(InsertionDetector) > order.index(ContainmentDetector)
    assert len(order) == len(set(order))
