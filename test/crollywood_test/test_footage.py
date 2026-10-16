"""
Tests for :mod:`crollywood.footage` and :mod:`crollywood.footage_playback`:

footage framed, filling the frame at a speed with a second view in its corner, and a
title over it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pytest
from semantic_digital_twin.spatial_types.spatial_types import Point2

from crollywood.footage import FootageFrame, Framing, ViewOfTheRun
from crollywood.footage_playback import (
    FootagePlayback,
    InsetStill,
    TitleOverFootage,
)
from crollywood.frame import Frame, Resolution
from crollywood.typesetting import Anchor
from crollywood.video import Video

NO_TIME = timedelta(0)
CAMERA_SIZE = Resolution(width=160, height=90)
STILL_GREY = 200


def seconds(amount: float) -> timedelta:
    """
    :return: That many seconds.
    """
    return timedelta(seconds=amount)


# %% framing


def test_a_framing_cuts_the_margins_it_names_off_a_picture() -> None:
    rows, columns = 6, 8
    pixels = np.arange(rows * columns * 3, dtype=np.uint8).reshape(rows, columns, 3)
    framing = Framing(top=1, left=2, right=1, bottom=0)
    framed = framing.of(Frame(pixels)).pixels
    assert framed.shape == (
        rows - framing.top - framing.bottom,
        columns - framing.left - framing.right,
        3,
    )
    assert (framed == pixels[1:6, 2:7]).all()


# %% a stretch of a recording, filling the frame


@dataclass
class CountingVideo(Video):
    """
    A video whose every frame says which second it was asked for, in its red byte.
    """

    resolution: Resolution = CAMERA_SIZE
    duration: timedelta = seconds(100)
    asked: list = field(default_factory=list)

    def frame_at(self, moment: timedelta) -> Frame:
        self.asked.append(moment)
        pixels = np.zeros(self.resolution.shape, dtype=np.uint8)
        pixels[..., 0] = int(moment.total_seconds())
        return Frame(pixels)


def test_a_view_reads_its_video_from_where_its_stretch_starts_through_its_framing() -> (
    None
):
    video = CountingVideo(Resolution(width=200, height=100))
    starts_at, into = seconds(15.5), seconds(2)
    framing = Framing(top=10, bottom=10)
    picture = ViewOfTheRun(video, starts_at=starts_at, framing=framing).frame_at(into)
    assert video.asked == [starts_at + into]
    assert picture.pixels.shape == (80, 200, 3)
    assert picture.pixels[0, 0, 0] == int((starts_at + into).total_seconds())


def test_the_footage_fills_the_frame_with_the_inset_in_the_corner_in_step() -> None:
    inset_start, main_start = seconds(18), seconds(2.5)
    length, speed, moment = seconds(30), 2.0, seconds(2)
    own = ViewOfTheRun(CountingVideo(), starts_at=inset_start)
    by_hand = ViewOfTheRun(CountingVideo(), starts_at=main_start)
    scene = FootagePlayback(by_hand, length=length, speed=speed, inset=own)
    assert scene.duration == length / speed
    frame = scene.frame_at(moment)
    assert frame.resolution == scene.resolution
    assert own.video.asked[-1] == inset_start + moment * speed
    assert by_hand.video.asked[-1] == main_start + moment * speed
    inset = scene.inset_panel
    assert inset.max_x - inset.min_x == scene.inset_width
    assert inset.max_x == scene.resolution.width - scene.resolution.margin
    assert inset.max_y <= scene.resolution.stage_height
    # the full view fills the frame: the footage's second is in every corner's red byte
    bottom_left = frame.pixels[scene.resolution.height - 1, 0]
    assert bottom_left[0] == int((main_start + moment * speed).total_seconds())
    assert bottom_left[1] == 0


def test_the_inset_shows_what_it_is_given_before_it_runs() -> None:
    inset_start, shown_for = seconds(18), seconds(3)
    own = ViewOfTheRun(CountingVideo(), starts_at=inset_start)
    by_hand = ViewOfTheRun(CountingVideo(), starts_at=NO_TIME)
    picture = CAMERA_SIZE.blank(STILL_GREY)
    scene = FootagePlayback(
        by_hand,
        length=seconds(10),
        speed=1.0,
        inset=own,
        inset_still=InsetStill(picture, shown_for),
    )
    assert (scene.inset_picture_at(seconds(1)).pixels == STILL_GREY).all()
    later = seconds(4)
    assert scene.inset_picture_at(later).pixels[0, 0, 0] == int(
        (inset_start + later).total_seconds()
    )


def test_the_title_stands_over_the_footage_and_fades_out_when_told() -> None:
    footage = FootagePlayback(
        ViewOfTheRun(CountingVideo(), NO_TIME), length=seconds(6), speed=1.0
    )
    title_for, fade = seconds(4), seconds(0.5)
    titled = TitleOverFootage(
        footage, "A Title", "a line", title_for=title_for, fade=fade
    )
    assert titled.duration == footage.duration
    assert titled.title_at(seconds(1)) == 1.0
    assert titled.title_at(title_for + fade) == 0.0
    with_title = titled.frame_at(seconds(1))
    after_the_fade = seconds(5)
    without = titled.frame_at(after_the_fade)
    assert not np.array_equal(with_title.pixels, without.pixels)
    assert np.array_equal(without.pixels, footage.frame_at(after_the_fade).pixels)


@pytest.mark.parametrize(
    "length, speed", [(seconds(32), 4.0), (seconds(20), 2.0)], ids=["fast", "slow"]
)
def test_the_footage_lasts_the_stretch_divided_by_the_speed(
    length: timedelta, speed: float
) -> None:
    scene = FootagePlayback(
        ViewOfTheRun(CountingVideo(), NO_TIME), length=length, speed=speed
    )
    assert scene.duration == length / speed


def test_footage_at_its_own_pace_carries_no_speed_badge() -> None:
    scene = FootagePlayback(
        ViewOfTheRun(CountingVideo(), NO_TIME), length=seconds(5), speed=1.0
    )
    full_view = scene.full_view.frame_at(NO_TIME)
    unbadged = scene.resolution.blank(0).pasted(full_view, scene.resolution.whole)
    assert np.array_equal(scene.frame_at(NO_TIME).pixels, unbadged.pixels)


# %% badges


def test_a_badge_stands_at_the_corner_it_is_anchored_to() -> None:
    resolution = Resolution(width=400, height=200)
    footage = FootageFrame(role_taker=resolution.blank(0))
    corner = Point2(300, 100)
    text = "×2"
    for anchor in (Anchor.LEFT_TOP, Anchor.RIGHT_TOP, Anchor.RIGHT_MIDDLE):
        badged = footage.badged(text, corner, anchor).pixels
        changed_rows, changed_columns = np.nonzero(badged.max(axis=2))
        if anchor is Anchor.LEFT_TOP:
            assert changed_columns.min() >= float(corner.x)
            assert changed_rows.min() >= float(corner.y)
        else:
            assert changed_columns.max() <= float(corner.x)
        if anchor is Anchor.RIGHT_MIDDLE:
            assert changed_rows.max() <= float(corner.y)


def test_the_speed_badge_stands_in_the_top_right_corner_inside_the_margin() -> None:
    resolution = Resolution(width=400, height=200)
    badged = FootageFrame(role_taker=resolution.blank(0)).speed_badged(4.0).pixels
    rows, columns = np.nonzero(badged.max(axis=2))
    assert columns.max() <= resolution.width - resolution.margin
    assert rows.min() >= resolution.margin
