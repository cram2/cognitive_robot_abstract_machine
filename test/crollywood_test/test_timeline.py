"""
Tests for :mod:`crollywood.timeline`: scenes laid end to end, dissolving into one
another, and read out as a stream of frames at a fixed rate.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from crollywood.exceptions import FramesOfDifferentSizes, MomentOutsideScene
from crollywood.frame import Frame, Resolution
from crollywood.slides import SingleFrameSlide
from crollywood.timeline import (
    FrameFade,
    TimelineScene,
    TimelineStyle,
    Transition,
    VideoTimeline,
)

from .marks import ProgressStamp, stamped_value

FRAME_SIZE = Resolution(width=64, height=32)
NO_TIME = timedelta(0)


def seconds(amount: float) -> timedelta:
    """
    :return: That many seconds.
    """
    return timedelta(seconds=amount)


def flat(grey: int) -> Frame:
    """
    :return: A frame of one grey value.
    """
    return FRAME_SIZE.blank(grey)


def first_grey(frame: Frame) -> int:
    """
    :return: The grey of a frame's first pixel.
    """
    return int(frame.pixels[0, 0, 0])


# %% one scene


def test_a_still_shows_the_same_frame_for_its_whole_duration() -> None:
    held = seconds(1.5)
    still = SingleFrameSlide(picture=flat(200), held_for=held)
    assert still.duration == held
    assert first_grey(still.frame_at(NO_TIME)) == 200
    assert first_grey(still.frame_at(held - seconds(0.1))) == 200


def test_a_scene_refuses_a_moment_outside_its_video() -> None:
    held = seconds(1)
    scene = TimelineScene(role_taker=SingleFrameSlide(picture=flat(0), held_for=held))
    with pytest.raises(MomentOutsideScene):
        scene.frame_at(held)


def test_a_scene_shows_what_its_video_shows() -> None:
    scene = TimelineScene(
        role_taker=SingleFrameSlide(picture=flat(200), held_for=seconds(1))
    )
    assert first_grey(scene.frame_at(NO_TIME)) == 200
    assert scene.duration == scene.role_taker.duration


def test_progress_through_a_scene_is_the_moment_over_the_duration() -> None:
    scene = TimelineScene(
        role_taker=SingleFrameSlide(picture=flat(0), held_for=seconds(4))
    )
    assert scene.progress_at(seconds(1)) == 0.25


# %% the timeline


def test_the_timeline_lays_scenes_end_to_end_without_a_dissolve() -> None:
    first_hold, second_hold = seconds(0.5), seconds(0.25)
    timeline = VideoTimeline(
        scenes=[
            TimelineScene(role_taker=SingleFrameSlide(flat(10), held_for=first_hold)),
            TimelineScene(role_taker=SingleFrameSlide(flat(20), held_for=second_hold)),
        ],
        style=TimelineStyle(frames_per_second=4, dissolve=NO_TIME),
    )
    frames = list(timeline.frames())
    assert timeline.duration == first_hold + second_hold
    assert [first_grey(frame) for frame in frames] == [10, 10, 20]


def test_a_dissolve_blends_the_end_of_one_scene_into_the_start_of_the_next() -> None:
    hold, dissolve = seconds(1), seconds(0.5)
    first_grey_level, second_grey_level = 0, 100
    timeline = VideoTimeline(
        scenes=[
            TimelineScene(
                role_taker=SingleFrameSlide(flat(first_grey_level), held_for=hold)
            ),
            TimelineScene(
                role_taker=SingleFrameSlide(flat(second_grey_level), held_for=hold)
            ),
        ],
        style=TimelineStyle(frames_per_second=4, dissolve=dissolve),
    )
    assert timeline.duration == 2 * hold - dissolve
    values = [first_grey(frame) for frame in timeline.frames()]
    assert len(values) == timeline.frame_count
    assert values[:2] == [first_grey_level] * 2
    assert values[-2:] == [second_grey_level] * 2
    # the two frames in the dissolve climb from the first scene towards the second
    assert first_grey_level <= values[2] < values[3] < second_grey_level


def test_the_timeline_rejects_scenes_of_different_sizes() -> None:
    wider = Resolution(FRAME_SIZE.width + 1, FRAME_SIZE.height)
    hold = seconds(0.25)
    timeline = VideoTimeline(
        scenes=[
            TimelineScene(role_taker=SingleFrameSlide(flat(0), held_for=hold)),
            TimelineScene(role_taker=SingleFrameSlide(wider.blank(0), held_for=hold)),
        ],
        style=TimelineStyle(frames_per_second=4, dissolve=NO_TIME),
    )
    with pytest.raises(FramesOfDifferentSizes):
        list(timeline.frames())


def test_a_timeline_plays_as_one_scene_of_a_longer_timeline() -> None:
    hold, greys = seconds(0.5), (10, 20, 30)
    chapter = VideoTimeline(
        [
            TimelineScene(role_taker=SingleFrameSlide(flat(greys[0]), held_for=hold)),
            TimelineScene(role_taker=SingleFrameSlide(flat(greys[1]), held_for=hold)),
        ],
        style=TimelineStyle(dissolve=NO_TIME),
    )
    whole = VideoTimeline(
        [
            TimelineScene(role_taker=chapter),
            TimelineScene(role_taker=SingleFrameSlide(flat(greys[2]), held_for=hold)),
        ],
        style=TimelineStyle(dissolve=NO_TIME),
    )
    assert whole.duration == chapter.duration + hold
    assert [first_grey(whole.frame_at(seconds(t))) for t in (0.25, 0.75, 1.25)] == [
        *greys
    ]


# %% fading


@pytest.mark.parametrize("elapsed, expected", [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)])
def test_a_fade_keeps_its_ends_and_its_middle(elapsed: float, expected: float) -> None:
    assert FrameFade(seconds(1)).weight_at(seconds(elapsed)) == pytest.approx(expected)


def test_a_fade_starts_and_ends_slowly() -> None:
    fade = FrameFade(seconds(1))
    assert fade.weight_at(seconds(0.1)) < 0.1
    assert fade.weight_at(seconds(0.9)) > 0.9


def test_a_fade_stays_at_its_ends_beyond_its_duration() -> None:
    fade = FrameFade(seconds(1))
    assert fade.weight_at(seconds(-1)) == 0.0
    assert fade.weight_at(seconds(2)) == 1.0


# %% a scene that cuts in


def test_a_scene_that_cuts_in_takes_no_dissolve_from_the_scene_before() -> None:
    hold, dissolve = seconds(1), seconds(0.5)
    greys = (10, 20, 30)
    timeline = VideoTimeline(
        scenes=[
            TimelineScene(role_taker=SingleFrameSlide(flat(greys[0]), held_for=hold)),
            TimelineScene(
                role_taker=SingleFrameSlide(flat(greys[1]), held_for=hold),
                transition=Transition.CUT,
            ),
            TimelineScene(role_taker=SingleFrameSlide(flat(greys[2]), held_for=hold)),
        ],
        style=TimelineStyle(frames_per_second=10, dissolve=dissolve),
    )
    assert timeline.dissolve_into(0) == NO_TIME
    assert timeline.dissolve_into(1) == NO_TIME
    assert timeline.dissolve_into(2) == dissolve
    assert timeline.duration == 3 * hold - dissolve
    assert timeline.starts() == [NO_TIME, hold, 2 * hold - dissolve]
    assert first_grey(timeline.frame_at(hold)) == greys[1]  # no blend at the cut
    assert first_grey(timeline.frame_at(hold - seconds(0.1))) == greys[0]
    # the dissolve after it as usual
    assert greys[1] < first_grey(timeline.frame_at(seconds(1.75))) < greys[2]


# %% overlays


def test_every_overlay_is_drawn_over_each_frame_at_that_moment_of_the_video() -> None:
    hold, grey = seconds(2), 100
    timeline = VideoTimeline(
        [
            TimelineScene(role_taker=SingleFrameSlide(flat(grey), held_for=hold)),
            TimelineScene(role_taker=SingleFrameSlide(flat(grey), held_for=hold)),
        ],
        style=TimelineStyle(
            frames_per_second=10, dissolve=NO_TIME, overlays=[ProgressStamp()]
        ),
    )
    for moment in (seconds(1), seconds(3)):
        frame = timeline.frame_at(moment)
        assert (frame.pixels[0, 0] == stamped_value(moment, timeline.duration)).all()
        assert (frame.pixels[1:] == grey).all()


def test_a_timeline_without_overlays_shows_its_scenes_alone() -> None:
    grey = 100
    timeline = VideoTimeline(
        [TimelineScene(role_taker=SingleFrameSlide(flat(grey), held_for=seconds(2)))],
        style=TimelineStyle(frames_per_second=10),
    )
    assert (timeline.frame_at(seconds(1)).pixels == grey).all()
