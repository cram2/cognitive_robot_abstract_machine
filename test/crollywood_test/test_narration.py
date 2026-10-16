"""
Tests for :mod:`crollywood.narration`: the lines the voice says over the scenes, where
each starts, that none runs into the next, the subtitles cut from them, and the
soundtrack they are laid on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from crollywood.exceptions import NarrationOverrun
from crollywood.frame import Frame, Resolution
from crollywood.narration import (
    NarratedLine,
    NarratedScene,
    NarratedTimeline,
    Narration,
    Soundtrack,
    SpokenLine,
    Storyboard,
)
from crollywood.speech import SoundChannel
from crollywood.subtitles import SubtitleCue, SubtitleRoom
from crollywood.slides import HeldSlide, SingleFrameSlide
from crollywood.timeline import TimelineScene, TimelineStyle, VideoTimeline
from crollywood.video import Video

from .voices import EvenlyPacedTextToSpeech

SAMPLE_RATE = 100
NO_TIME = timedelta(0)
SLIDE_SIZE = Resolution(16, 9)
PLAYER_ROOM = SubtitleRoom.of_player()
EIGHT_WORDS = "one two three four five six seven eight"
ONE_WORD_TIME = timedelta(seconds=0.5)
VOICE = EvenlyPacedTextToSpeech(time_per_word=ONE_WORD_TIME)
FOUR_SECONDS = 8 * ONE_WORD_TIME


def seconds(amount: float) -> timedelta:
    """
    :return: That many seconds.
    """
    return timedelta(seconds=amount)


def spoken(text: str, starts: float, length: float) -> SpokenLine:
    """
    :return: A line starting at a moment and said for a while, as steady sound.
    """
    samples = np.ones(int(length * SAMPLE_RATE), dtype=np.float32)
    return SpokenLine(
        NarratedLine(text), seconds(starts), SoundChannel(samples, SAMPLE_RATE)
    )


# %% lines and speech


def test_a_line_is_said_as_written_unless_told_otherwise() -> None:
    written, said = "paper 3889", "paper thirty-eight eighty-nine"
    assert NarratedLine(written).said == written
    assert NarratedLine(written, spoken=said).said == said


def test_lines_said_in_turn_start_a_pause_after_the_one_before_ends() -> None:
    lines = [
        NarratedLine("one two"),
        NarratedLine("three four five six"),
        NarratedLine("seven"),
    ]
    pause = seconds(0.5)
    first, second = 2 * ONE_WORD_TIME, 4 * ONE_WORD_TIME
    assert NarratedLine.start_times(lines, VOICE, pause) == [
        NO_TIME,
        first + pause,
        first + pause + second + pause,
    ]
    assert NarratedLine.start_times([], VOICE, pause) == []


def test_lines_said_in_turn_are_each_measured_at_their_own_pace() -> None:
    half_pace = 0.5
    lines = [NarratedLine("one two", pace=half_pace), NarratedLine("three")]
    pause = seconds(0.5)
    assert NarratedLine.start_times(lines, VOICE, pause)[1] == (
        2 * ONE_WORD_TIME / half_pace + pause
    )


# %% subtitles of a line


def test_a_short_line_is_one_cue_over_the_whole_speech() -> None:
    text, starts, length = "Thank you for watching.", 10.0, 2.0
    line = spoken(text, starts, length)
    assert line.cues() == [SubtitleCue(seconds(starts), seconds(starts + length), text)]


def test_a_long_line_is_cut_at_clauses_into_cues_that_fit_the_room() -> None:
    text = (
        "The framework provides a common query interface over heterogeneous knowledge "
        "sources and computations. A query's representation and grounded meaning are "
        "separated from its computation, so that different backends can answer the same "
        "query from different sources of knowledge."
    )
    starts, length = 5.0, 20.0
    two_rows = SubtitleRoom(PLAYER_ROOM.characters_per_row, rows=2)
    line = spoken(text, starts, length)
    cues = line.cues(two_rows)
    assert len(cues) >= 3
    assert all(len(cue.text) <= PLAYER_ROOM.characters_per_row for cue in line.cues())
    assert " ".join(cue.text for cue in cues) == text
    for cue in cues:
        assert len(cue.rows(two_rows)) <= two_rows.rows
    assert cues[0].start == seconds(starts) and cues[-1].end == line.ends
    for earlier, later in zip(cues, cues[1:]):
        assert later.start == earlier.end
    # a cue's share of the time is its share of the letters
    share = (cues[0].end - cues[0].start) / seconds(length)
    assert share == pytest.approx(len(cues[0].text) / len(text), abs=0.025)


# %% one line after the other


def test_a_line_that_runs_into_the_next_is_refused() -> None:
    runs_for = seconds(10)
    Narration([spoken("a", 0.0, 4.0), spoken("b", 4.5, 2.0)]).check(runs_for)
    with pytest.raises(NarrationOverrun) as refused:
        Narration([spoken("a", 0.0, 5.0), spoken("b", 4.5, 2.0)]).check(runs_for)
    assert refused.value.line.written == "a"
    assert refused.value.by == seconds(0.5)


def test_a_line_that_runs_past_the_end_of_the_video_is_refused() -> None:
    with pytest.raises(NarrationOverrun) as refused:
        Narration([spoken("a", 0.0, 4.0), spoken("b", 9.0, 2.0)]).check(seconds(10))
    assert refused.value.line.written == "b"
    assert refused.value.by == seconds(1)


def test_the_soundtrack_lays_each_line_where_it_starts() -> None:
    narration = Narration([spoken("a", 1.0, 1.0), spoken("b", 3.0, 0.5)])
    track = narration.soundtrack(runs_for=seconds(5))
    assert isinstance(track, Soundtrack)
    assert track.sample_rate == SAMPLE_RATE and len(track.samples) == 5 * SAMPLE_RATE
    for silent_second in (0.5, 2.5, 4.0):
        assert track.samples[int(silent_second * SAMPLE_RATE)] == 0
    for spoken_second in (1.5, 3.2):
        assert track.samples[int(spoken_second * SAMPLE_RATE)] == 1


def test_a_soundtrack_is_written_as_a_wave_file(tmp_path: Path) -> None:
    runs_for = seconds(3)
    narration = Narration([spoken("a", 1.0, 1.0)])
    path = narration.soundtrack(runs_for).written_to(tmp_path / "track.wav")
    assert SoundChannel.read_from(path).duration == runs_for


def test_nothing_said_leaves_a_silent_soundtrack_as_long_as_the_video() -> None:
    runs_for = seconds(2)
    track = Narration([]).soundtrack(runs_for)
    assert track.duration == runs_for
    assert not track.samples.any()


def test_the_narration_is_written_as_a_subrip_file(tmp_path: Path) -> None:
    narration = Narration([spoken("One.", 1.0, 1.5), spoken("Two.", 61.25, 2.0)])
    path = narration.written_to(tmp_path / "video.srt")
    assert path.read_text(encoding="utf-8") == (
        f"1\n{narration.cues()[0].timestamps}\nOne.\n\n"
        f"2\n{narration.cues()[1].timestamps}\nTwo.\n\n"
    )


def test_a_report_row_says_how_much_silence_follows_each_line() -> None:
    narration = Narration([spoken("first", 0.0, 1.0), spoken("second", 3.0, 1.0)])
    report = narration.report(runs_for=seconds(10)).splitlines()
    assert len(report) == len(narration.lines)
    assert "+2.0 s free" in report[0] and "+6.0 s free" in report[1]


# %% the storyboard


@dataclass
class Page(HeldSlide):
    """
    A held slide that dissolves in, like most.
    """

    def frame_at(self, moment: timedelta) -> Frame:
        return SLIDE_SIZE.blank(255)


def slide(held_for: float) -> Page:
    """
    :return: A slide held that long.
    """
    return Page(held_for=seconds(held_for))


@dataclass
class CloseUp(Video):
    """
    A video whose length comes from what it shows, not from a hold.
    """

    shown_for: timedelta = seconds(2)

    @property
    def duration(self) -> timedelta:
        return self.shown_for

    def frame_at(self, moment: timedelta) -> Frame:
        return SLIDE_SIZE.blank(255)


def test_a_slide_grows_to_hold_its_line_and_a_close_up_does_not() -> None:
    close_up = CloseUp()
    held = 3.0
    board = Storyboard(
        [
            NarratedScene(role_taker=slide(held), lines=(NarratedLine(EIGHT_WORDS),)),
            NarratedScene(role_taker=slide(held), lines=(NarratedLine("one"),)),
            NarratedScene(role_taker=close_up, lines=(NarratedLine(EIGHT_WORDS),)),
            NarratedScene(role_taker=close_up),
        ]
    )
    board.fitted_to(VOICE)
    assert board.scenes[0].held_for == FOUR_SECONDS + board.lead + board.tail
    assert board.scenes[1].held_for == seconds(held)
    assert (
        board.scenes[2].role_taker is close_up
        and close_up.duration == close_up.shown_for
    )


def test_the_lines_start_where_their_scenes_start_plus_the_lead() -> None:
    dissolve = seconds(0.5)
    board = Storyboard(
        [
            NarratedScene(role_taker=slide(3.0), lines=(NarratedLine("one two"),)),
            NarratedScene(role_taker=slide(3.0)),
            NarratedScene(role_taker=slide(3.0), lines=(NarratedLine("three"),)),
        ]
    )
    style = TimelineStyle(dissolve=dissolve)
    narration = board.narrated_by(VOICE, style)
    starts = VideoTimeline(board.scenes, style).starts()
    assert [line.starts for line in narration.lines] == [
        starts[0] + board.lead,
        starts[2] + board.lead,
    ]
    assert narration.lines[0].speech.duration == 2 * ONE_WORD_TIME


def test_two_lines_on_one_scene_are_said_one_after_the_other_with_a_pause() -> None:
    board = Storyboard(
        [
            NarratedScene(
                role_taker=slide(1.0),
                lines=(NarratedLine("one two"), NarratedLine("three")),
            )
        ]
    )
    board.fitted_to(VOICE)
    first, second = 2 * ONE_WORD_TIME, ONE_WORD_TIME
    assert board.scenes[0].held_for == (
        board.lead + first + board.pause + second + board.tail
    )
    narration = board.narrated_by(VOICE, TimelineStyle(dissolve=seconds(0.5)))
    assert [line.starts for line in narration.lines] == [
        board.lead,
        board.lead + first + board.pause,
    ]


def test_a_line_that_runs_on_leaves_its_slide_alone_and_a_delay_moves_its_start() -> (
    None
):
    first_hold, delay, dissolve = 1.0, seconds(2.5), seconds(0.5)
    board = Storyboard(
        [
            NarratedScene(
                role_taker=slide(first_hold),
                lines=(NarratedLine(EIGHT_WORDS),),
                runs_on=True,
            ),
            NarratedScene(
                role_taker=slide(3.0), lines=(NarratedLine("five"),), delay=delay
            ),
        ]
    )
    board.fitted_to(VOICE)
    assert board.scenes[0].held_for == seconds(first_hold)
    assert board.scenes[1].held_for == (board.lead + delay + ONE_WORD_TIME + board.tail)
    narration = board.narrated_by(VOICE, TimelineStyle(dissolve=dissolve))
    second_start = seconds(first_hold) - dissolve
    assert narration.lines[1].starts == second_start + board.lead + delay
    with pytest.raises(NarrationOverrun):
        narration.check(runs_for=seconds(100))


def test_a_later_line_waits_beyond_the_pause_for_what_it_speaks_of() -> None:
    voice = EvenlyPacedTextToSpeech(time_per_word=seconds(1))
    waits = (seconds(1.5), NO_TIME)
    board = Storyboard(
        [
            NarratedScene(
                role_taker=slide(1.0),
                lines=(NarratedLine("one"), NarratedLine("two"), NarratedLine("three")),
                waits=waits,
            )
        ],
        pause=seconds(0.5),
    )
    board.fitted_to(voice)
    word = voice.time_per_word
    assert board.scenes[0].held_for == (
        board.lead + 3 * word + 2 * board.pause + sum(waits, NO_TIME) + board.tail
    )
    starts = [
        line.starts
        for line in board.narrated_by(voice, TimelineStyle(dissolve=NO_TIME)).lines
    ]
    second_start = board.lead + word + board.pause + waits[0]
    assert starts == [board.lead, second_start, second_start + word + board.pause]


# %% the narration placed on a timeline


def test_a_narrated_timeline_is_a_timeline_that_checks_its_narration_against_its_length() -> (
    None
):
    timeline = VideoTimeline(
        [
            TimelineScene(
                role_taker=SingleFrameSlide(SLIDE_SIZE.blank(0), held_for=seconds(10))
            )
        ],
        style=TimelineStyle(frames_per_second=10),
    )
    fine = NarratedTimeline(
        role_taker=timeline, narration=Narration([spoken("a", 0, 4)])
    )
    fine.check()
    assert fine.duration == timeline.duration
    assert fine.soundtrack().duration == timeline.duration
    too_long = NarratedTimeline(
        role_taker=timeline, narration=Narration([spoken("a", 9.0, 2.0)])
    )
    with pytest.raises(NarrationOverrun):
        too_long.check()
