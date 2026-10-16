"""
Tests for :mod:`crollywood.captions`: subtitles drawn into the picture, in the band
every scene keeps clear or on a box over the bottom of the frame.
"""

from __future__ import annotations


from datetime import timedelta

import numpy as np

from crollywood.captions import BandCaption, BoxedCaption, Subtitled
from crollywood.frame import Frame, Resolution
from crollywood.narration import NarratedLine, NarratedTimeline, Narration, SpokenLine
from crollywood.speech import SoundChannel
from crollywood.subtitles import SubtitleCue, SubtitleRoom
from crollywood.theme import VideoTheme
from crollywood.slides import SingleFrameSlide
from crollywood.timeline import TimelineScene, TimelineStyle, VideoTimeline

from .marks import ProgressStamp, stamped_value

SMALL = Resolution(width=320, height=180)
WIDE = Resolution(width=1280, height=720)
WHITE = 255
DARK = 128
CAPTION = SubtitleCue(timedelta(seconds=1), timedelta(seconds=2), "hello there")
WHILE_CAPTIONED = timedelta(seconds=1.5)
BEFORE_CAPTION = timedelta(seconds=0.5)
HOLD = timedelta(seconds=4)
NO_TIME = timedelta(0)


def timeline_of(
    resolution: Resolution, grey: int = WHITE, **changes: object
) -> VideoTimeline:
    """
    :return: A timeline of one still of a grey.
    """
    return VideoTimeline(
        [
            TimelineScene(
                role_taker=SingleFrameSlide(resolution.blank(grey), held_for=HOLD)
            )
        ],
        TimelineStyle(frames_per_second=10, dissolve=NO_TIME, **changes),
    )


def narrated_saying(
    text: str, starts: float, length: float, **changes: object
) -> NarratedTimeline:
    """
    :return: The one-still timeline with a line of steady sound said over it.
    """
    rate = 100
    line = SpokenLine(
        NarratedLine(text),
        timedelta(seconds=starts),
        SoundChannel(np.ones(int(length * rate), dtype=np.float32), rate),
    )
    return NarratedTimeline(
        role_taker=timeline_of(WIDE, **changes), narration=Narration([line])
    )


def inks_of(frame: Frame) -> set:
    """
    :return: The colours a frame holds, as bytes.
    """
    return {tuple(pixel) for pixel in frame.pixels.reshape(-1, 3)}


def stage_rows(frame: Frame) -> slice:
    """
    :return: The rows above the band the captions are read on.
    """
    return slice(0, int(frame.resolution.stage_height))


def band_rows(frame: Frame) -> slice:
    """
    :return: The rows of the band the captions are read on.
    """
    return slice(int(frame.resolution.stage_height), frame.resolution.height)


# %% a timeline with its captions


def test_a_subtitled_timeline_draws_the_cue_of_the_moment_into_the_band_and_nothing_else() -> (
    None
):
    timeline = timeline_of(SMALL)
    subtitled = Subtitled(timeline.scenes, timeline.style, cues=[CAPTION])
    assert subtitled.duration == timeline.duration
    assert subtitled.frame_count == timeline.frame_count
    assert subtitled.cue_at(WHILE_CAPTIONED) is CAPTION
    assert subtitled.cue_at(CAPTION.end) is None
    assert subtitled.frame_at(BEFORE_CAPTION).pixels.min() == WHITE
    written = subtitled.frame_at(WHILE_CAPTIONED)
    assert written.pixels[band_rows(written)].min() < DARK
    assert written.pixels[stage_rows(written)].min() == WHITE


def test_burned_in_subtitles_show_a_whole_clause_on_two_rows_of_the_band() -> None:
    text = (
        "Which objects moved is answered from the motion events and their tracked "
        "objects; which of them the robot moved, from the agent-interaction events "
        "restricted to those objects."
    )
    narrated = narrated_saying(text, 0.0, 9.0)
    room = SubtitleRoom.burned_in()
    burned_in, players = narrated.narration.cues(room), narrated.narration.cues()
    assert len(burned_in) < len(players)
    assert all(len(cue.rows(room)) <= room.rows for cue in burned_in)
    assert any(
        len(cue.text) > SubtitleRoom.of_player().characters_per_row for cue in burned_in
    )
    subtitled = Subtitled.over(narrated)
    assert subtitled.cues == burned_in
    written = subtitled.frame_at(WHILE_CAPTIONED)
    band = written.pixels[band_rows(written)]
    inked_rows = np.flatnonzero(band.min(axis=(1, 2)) < DARK)
    assert written.pixels[stage_rows(written)].min() == WHITE
    # two rows of letters, a gap between them, inside the band with air above and below
    air = 4
    assert inked_rows[0] > air and inked_rows[-1] < band.shape[0] - air - 1
    assert np.any(np.diff(inked_rows) > air)
    # one row hangs from the same baseline as the first of two
    short = Subtitled.over(narrated_saying("A short line.", 0.0, 9.0))
    short_frame = short.frame_at(WHILE_CAPTIONED)
    short_rows = np.flatnonzero(
        short_frame.pixels[band_rows(short_frame)].min(axis=(1, 2)) < DARK
    )
    assert short_rows[0] == inked_rows[0]


def test_subtitles_are_written_over_a_timeline_that_keeps_its_overlays() -> None:
    narrated = narrated_saying("hello there", 1.0, 1.0, overlays=[ProgressStamp()])
    subtitled = Subtitled.over(narrated, captions=BoxedCaption())
    assert subtitled.style.overlays == narrated.style.overlays
    assert isinstance(subtitled.captions, BoxedCaption)
    stamp = subtitled.frame_at(WHILE_CAPTIONED).pixels[0, 0]
    assert (stamp == stamped_value(WHILE_CAPTIONED, narrated.duration)).all()


# %% how a caption is drawn


def test_over_footage_the_caption_is_white_on_a_band_shaded_towards_black() -> None:
    footage_grey = 120
    subtitled = Subtitled(
        timeline_of(SMALL, footage_grey).scenes,
        TimelineStyle(frames_per_second=10, dissolve=NO_TIME),
        cues=[CAPTION],
    )
    written = subtitled.frame_at(WHILE_CAPTIONED)
    band = written.pixels[band_rows(written)]
    clear_of_the_band = int(SMALL.stage_height) - 30
    assert band.max() == WHITE  # white letters
    assert band[-1].mean() < footage_grey  # shaded at the bottom
    assert (written.pixels[:clear_of_the_band] == footage_grey).all()


def test_a_boxed_caption_is_written_light_on_a_shaded_box_over_the_bottom_of_the_picture() -> (
    None
):
    resolution = Resolution(width=640, height=360)
    style = BoxedCaption()
    subtitled = Subtitled(
        timeline_of(resolution).scenes,
        TimelineStyle(frames_per_second=10, dissolve=NO_TIME),
        cues=[CAPTION],
        captions=style,
    )
    written = subtitled.frame_at(WHILE_CAPTIONED)
    shade = round(WHITE * (1.0 - style.shade.darkness))
    box_rows, box_columns = np.nonzero((written.pixels == shade).all(axis=2))
    box = written.pixels[
        box_rows.min() : box_rows.max() + 1, box_columns.min() : box_columns.max() + 1
    ]
    assert box_rows.min() > resolution.height / 2
    assert 0 < box_columns.min() and box_columns.max() < resolution.width - 1
    assert tuple(style.theme.footage_text.to_rgb_bytes()) in inks_of(Frame(box))
    assert (written.pixels[: box_rows.min()] == WHITE).all()
    assert (written.pixels[box_rows.max() + 1 :] == WHITE).all()


def test_a_band_caption_on_the_page_of_its_theme_is_written_in_its_text_colour() -> (
    None
):
    theme = VideoTheme.dark()
    subtitled = Subtitled(
        [
            TimelineScene(
                role_taker=SingleFrameSlide(theme.page_of(SMALL), held_for=HOLD)
            )
        ],
        TimelineStyle(frames_per_second=10, dissolve=NO_TIME),
        cues=[CAPTION],
        captions=BandCaption(theme=theme),
    )
    written = subtitled.frame_at(WHILE_CAPTIONED)
    band = written.pixels[band_rows(written)]
    edge = 10
    assert tuple(theme.text.to_rgb_bytes()) in inks_of(Frame(band))
    assert (band[:, :edge] == tuple(theme.page.to_rgb_bytes())).all()  # not shaded
