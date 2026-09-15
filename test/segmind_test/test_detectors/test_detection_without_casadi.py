"""
Tests that detecting events reads the world as plain numbers and calls no CasADi, so
detection can run on a thread of its own beside a plan that uses CasADi.

Only the detectors' ticks are recorded: the tests move the world through symbolic
transforms, the way a plan or a simulation would on its own thread. Detectors that only
combine events already logged, such as picking up and placing, read no geometry and are
left out.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from giskardpy.motion_statechart.context import MotionStatechartContext
from typing_extensions import Set, Type

from segmind.datastructures.events import (
    ContactEvent,
    ContainmentEvent,
    DetectionEvent,
    LossOfContactEvent,
    LossOfContainmentEvent,
    LossOfSupportEvent,
    RotationEvent,
    StopRotationEvent,
    StopTranslationEvent,
    SupportEvent,
    TranslationEvent,
)
from segmind.detectors.atomic_event_detectors_nodes import (
    ContactDetector,
    LossOfContactDetector,
    RotationDetector,
    StopRotationDetector,
    StopTranslationDetector,
    TranslationDetector,
)
from segmind.detectors.base import SegmindContext
from segmind.detectors.spatial_relation_detector_nodes import (
    ContainmentDetector,
    LossOfContainmentDetector,
    LossOfSupportDetector,
    SupportDetector,
)
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.statecharts.segmind_statechart import SegmindStatechart
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import Body

from ...casadi_calls import CasadiCalls

CARRY_STEP = 0.1
"""
How far a test carries the milk along x and z per tick.
"""

TURN_STEP = 0.3
"""
How far a test turns the milk per tick while carrying it.
"""

TICKS_OF_CARRYING = 6
"""
How many ticks a test carries the milk for; more than a motion window spans.
"""

TICKS_AT_REST = 6
"""
How many ticks a test leaves the milk where it is; more than a motion window spans.
"""

RESTING_ON_THE_TABLE = (-1.7, 0.0, 0.93)
"""
Where the milk stands resting on the apartment's second box.
"""

HANDLING_EVENTS = {
    ContactEvent,
    LossOfContactEvent,
    SupportEvent,
    LossOfSupportEvent,
    ContainmentEvent,
    LossOfContainmentEvent,
    TranslationEvent,
    StopTranslationEvent,
    RotationEvent,
    StopRotationEvent,
}
"""
The events lifting, carrying, boxing and putting back the milk gives rise to.
"""


# %% the watched milk


@pytest.fixture
def milk_in_the_apartment(_simple_apartment_setup):
    """
    The apartment with its milk and boxes, the milk put back where it stood afterwards.
    """
    world = _simple_apartment_setup
    milk = world.get_body_by_name("milk.stl")
    yield world, milk, world.get_body_by_name("box")
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        -1.7, 0, 1.07, yaw=np.pi, reference_frame=milk.parent_connection.parent
    )


def _place(body: Body, x: float, y: float, z: float, yaw: float = 0.0) -> None:
    """
    Put ``body`` at a place relative to its parent.
    """
    body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x, y, z, yaw=yaw, reference_frame=body.parent_connection.parent
    )


def _tick_recorded(
    executor: EpisodeSegmenterExecutor, casadi_calls: CasadiCalls
) -> None:
    """
    Tick the detectors once, recording the calls into CasADi they make.
    """
    with casadi_calls:
        executor.tick()


def _detected_event_types(
    executor: EpisodeSegmenterExecutor,
) -> Set[Type[DetectionEvent]]:
    segmind_context = executor.context.require_extension(SegmindContext)
    return {type(event) for event in segmind_context.logger.get_events()}


# %% detection calls no CasADi


def test_detecting_the_milk_being_handled_calls_no_casadi(milk_in_the_apartment):
    world, milk, box = milk_in_the_apartment
    executor = EpisodeSegmenterExecutor(context=MotionStatechartContext(world=world))
    detector_types = [
        ContactDetector,
        LossOfContactDetector,
        SupportDetector,
        LossOfSupportDetector,
        ContainmentDetector,
        LossOfContainmentDetector,
        TranslationDetector,
        StopTranslationDetector,
        RotationDetector,
        StopRotationDetector,
    ]
    _place(milk, *RESTING_ON_THE_TABLE)
    executor.compile(
        SegmindStatechart().build_statechart(
            [detector_type(tracked_object=milk) for detector_type in detector_types]
        )
    )
    casadi_calls = CasadiCalls()

    _tick_recorded(executor, casadi_calls)
    rest_x, rest_y, rest_z = RESTING_ON_THE_TABLE
    for step in range(1, TICKS_OF_CARRYING + 1):
        _place(
            milk,
            rest_x + CARRY_STEP * step,
            rest_y,
            rest_z + CARRY_STEP * step,
            yaw=TURN_STEP * step,
        )
        _tick_recorded(executor, casadi_calls)
    for _ in range(TICKS_AT_REST):
        _tick_recorded(executor, casadi_calls)
    _place(milk, *box.numeric_global_pose.position)
    _tick_recorded(executor, casadi_calls)
    _place(milk, *RESTING_ON_THE_TABLE)
    _tick_recorded(executor, casadi_calls)

    assert HANDLING_EVENTS - _detected_event_types(executor) == set()
    assert casadi_calls.calls_by_caller == Counter()
