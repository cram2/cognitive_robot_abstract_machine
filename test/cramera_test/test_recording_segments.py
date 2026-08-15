"""
Unit tests for the stretches a live recording is divided into.

A recording carries no action list — only what each tick captured — so the segments its
replay timeline is marked from have to be read back out of the poses and the reported
actions. What is covered here is that reading: which objects count as carried, where
their carry windows begin and end, and which action names the stretch around them.
"""

from __future__ import annotations

import pytest
from typing_extensions import Dict, List, Optional, Sequence

from cramera.live.recording_segments import (
    MOTION_TOLERANCE,
    ObjectWindow,
    RecordedSegment,
    TRANSPORT_TOLERANCE,
    derive_segments,
    object_windows,
)
from cramera.live.recording import RecordedFrame

RESTING = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
"""
Where an object sits before anything touches it.
"""


def at(x: float) -> List[float]:
    """
    A pose ``x`` metres along the world's x axis.

    :param x: Distance from the origin.
    """
    return [x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def recording(
    poses: Sequence[Dict[str, List[float]]],
    steps: Optional[Sequence[Optional[str]]] = None,
) -> List[RecordedFrame]:
    """
    A recording of ``poses``, one tick per entry, with the actions reported alongside.

    :param poses: Each tick's loose-object poses by key.
    :param steps: The action reported on each tick, or None throughout.
    """
    return [
        RecordedFrame(
            frames={},
            base=None,
            objects=objects,
            step=steps[index] if steps else None,
        )
        for index, objects in enumerate(poses)
    ]


def carrying_milk() -> List[RecordedFrame]:
    """
    Six ticks in which the milk is picked up, carried, and set down elsewhere.
    """
    return recording(
        [
            {"milk.stl": at(0.0)},
            {"milk.stl": at(0.0)},
            {"milk.stl": at(0.3)},
            {"milk.stl": at(0.6)},
            {"milk.stl": at(1.0)},
            {"milk.stl": at(1.0)},
        ]
    )


# %% which objects count as carried
class TestObjectWindows:
    def test_a_recording_with_no_ticks_has_no_windows(self):
        assert object_windows([]) == []

    def test_an_object_that_never_moved_is_not_carried(self):
        frames = recording([{"milk.stl": RESTING}] * 4)

        assert object_windows(frames) == []

    def test_an_object_nudged_within_the_tolerance_is_not_carried(self):
        frames = recording(
            [{"milk.stl": RESTING}, {"milk.stl": at(TRANSPORT_TOLERANCE / 2)}]
        )

        assert object_windows(frames) == []

    def test_a_carried_object_yields_the_frames_it_left_and_settled(self):
        assert object_windows(carrying_milk()) == [
            ObjectWindow(object_key="milk.stl", attach=2, detach=4)
        ]

    def test_an_object_that_returns_to_where_it_started_is_not_carried(self):
        frames = recording(
            [
                {"milk.stl": RESTING},
                {"milk.stl": at(0.5)},
                {"milk.stl": RESTING},
            ]
        )

        assert object_windows(frames) == []

    def test_windows_come_out_in_the_order_the_objects_were_picked_up(self):
        frames = recording(
            [
                {"milk.stl": at(0.0), "bowl.stl": at(0.0)},
                {"milk.stl": at(0.0), "bowl.stl": at(0.4)},
                {"milk.stl": at(0.4), "bowl.stl": at(0.8)},
                {"milk.stl": at(0.8), "bowl.stl": at(0.8)},
            ]
        )

        assert [window.object_key for window in object_windows(frames)] == [
            "bowl.stl",
            "milk.stl",
        ]

    def test_an_object_that_only_appears_later_is_not_carried(self):
        """
        A window's start pose is the object's first recorded one; an object that was
        spawned mid-run has none, so there is nothing to compare against.
        """
        frames = recording([{}, {"milk.stl": at(0.0)}, {"milk.stl": at(1.0)}])

        assert object_windows(frames) == []


# %% how the recording is divided
class TestDeriveSegments:
    def test_a_recording_with_no_ticks_has_no_segments(self):
        assert derive_segments([]) == []

    def test_a_carried_object_gets_a_stretch_carrying_its_window(self):
        segments = derive_segments(carrying_milk())

        assert segments[-1].window == ObjectWindow(
            object_key="milk.stl", attach=2, detach=4
        )

    def test_what_precedes_the_first_pick_is_its_own_stretch(self):
        segments = derive_segments(carrying_milk())

        assert (segments[0].start, segments[0].end) == (0, 2)

    def test_the_stretches_cover_the_recording_without_gaps(self):
        segments = derive_segments(carrying_milk())

        assert segments[0].start == 0
        assert segments[-1].end == len(carrying_milk()) - 1
        for earlier, later in zip(segments, segments[1:]):
            assert earlier.end == later.start

    def test_a_stretch_is_named_after_the_action_running_when_the_pick_happened(self):
        frames = recording(
            [frame.objects for frame in carrying_milk()],
            [
                "ParkArms",
                "ParkArms",
                "Transport",
                "Transport",
                "Transport",
                "Transport",
            ],
        )

        segments = derive_segments(frames)

        assert [segment.step for segment in segments] == ["ParkArms", "Transport"]

    def test_a_pick_between_two_actions_takes_the_one_before_it(self):
        """
        The plan is regularly between motion nodes on the very tick an object leaves its
        place; the stretch is still the action that was running up to then.
        """
        frames = recording(
            [frame.objects for frame in carrying_milk()],
            ["Transport", "Transport", None, None, None, None],
        )

        assert derive_segments(frames)[-1].step == "Transport"

    def test_a_manipulation_no_action_was_reported_for_is_named_after_its_object(self):
        assert derive_segments(carrying_milk())[-1].step == "move_milk"

    def test_each_carried_object_gets_its_own_stretch(self):
        frames = recording(
            [
                {"milk.stl": at(0.0), "bowl.stl": at(0.0)},
                {"milk.stl": at(0.5), "bowl.stl": at(0.0)},
                {"milk.stl": at(1.0), "bowl.stl": at(0.0)},
                {"milk.stl": at(1.0), "bowl.stl": at(0.5)},
                {"milk.stl": at(1.0), "bowl.stl": at(1.0)},
            ]
        )

        segments = derive_segments(frames)

        assert [
            segment.window.object_key for segment in segments if segment.window
        ] == ["milk.stl", "bowl.stl"]

    def test_a_run_that_carried_nothing_is_divided_by_the_reported_actions(self):
        frames = recording(
            [{} for _ in range(6)],
            ["Navigate", "Navigate", "Navigate", "LookAt", "LookAt", "LookAt"],
        )

        segments = derive_segments(frames)

        assert [(s.step, s.start) for s in segments] == [("Navigate", 0), ("LookAt", 3)]

    def test_a_run_with_neither_manipulation_nor_actions_is_one_stretch(self):
        segments = derive_segments(recording([{} for _ in range(4)]))

        assert segments == [
            RecordedSegment(step=RecordedSegment.UNLABELLED_STEP, start=0, end=3)
        ]

    def test_an_object_jittering_below_the_tolerance_has_not_left_its_place(self):
        """
        An object shifts a little while it is being grasped; only leaving its resting
        pose by more than the tolerance counts as the pick.
        """
        frames = recording(
            [
                {"milk.stl": at(0.0)},
                {"milk.stl": at(MOTION_TOLERANCE / 2)},
                {"milk.stl": at(1.0)},
                {"milk.stl": at(1.0)},
            ]
        )

        assert object_windows(frames)[0].attach == 2


# %% what the bundle writes
class TestSegmentPayload:
    def test_a_plain_stretch_carries_no_object(self):
        payload = RecordedSegment(step="Navigate", start=0, end=9).to_payload()

        assert payload == {
            "step": "Navigate",
            "action": None,
            "arm": None,
            "start": 0,
            "end": 9,
        }

    def test_an_object_set_down_the_frame_it_arrives_is_still_carried(self):
        """
        A carry that resolves within one tick would otherwise leave the move unmarked.
        """
        frames = recording(
            [{"milk.stl": at(0.0)}, {"milk.stl": at(1.0)}, {"milk.stl": at(1.0)}]
        )

        assert object_windows(frames) == [
            ObjectWindow(object_key="milk.stl", attach=1, detach=1)
        ]

    def test_a_manipulation_stretch_names_its_object_and_both_frames(self):
        segment = RecordedSegment(
            step="Transport",
            start=0,
            end=9,
            window=ObjectWindow(object_key="milk.stl", attach=2, detach=7),
        )

        payload = segment.to_payload()

        assert payload["picks"] == "milk.stl"
        assert (payload["attach"], payload["detach"]) == (2, 7)
