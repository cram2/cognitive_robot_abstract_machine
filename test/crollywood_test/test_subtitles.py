"""
Tests for :mod:`crollywood.subtitles`: a line cut into pieces that fit the room a player
gives them, and the SubRip text written from them.
"""

from __future__ import annotations

from datetime import timedelta

from crollywood.subtitles import (
    EvenCuts,
    SubRipFile,
    SubtitleCue,
    SubtitleRoom,
    WordCuts,
)

PLAYER_ROOM = SubtitleRoom.of_player()
TWO_ROWS = SubtitleRoom(characters_per_row=PLAYER_ROOM.characters_per_row, rows=2)
LONG_SENTENCE = (
    "The working-memory backend evaluates the support condition again in "
    "simulation: a geometric check confirms that the cube rests on the lid."
)


def seconds(amount: float) -> timedelta:
    """
    :return: That many seconds.
    """
    return timedelta(seconds=amount)


def test_words_are_cut_into_parts_after_each_cut_word() -> None:
    words = ["one", "two", "three", "four"]
    assert WordCuts([0, 2]).parts(words) == ["one", "two three", "four"]


def test_every_even_cut_cuts_a_line_into_the_parts_asked_for() -> None:
    words = LONG_SENTENCE.split()
    parts = 3
    for cuts in EvenCuts(words, parts, TWO_ROWS):
        assert len(cuts.parts(words)) == parts


def test_the_rooms_of_a_player_and_of_the_picture_differ_in_rows_and_width() -> None:
    burned_in = SubtitleRoom.burned_in()
    assert burned_in.rows > PLAYER_ROOM.rows
    assert burned_in.characters_per_row > PLAYER_ROOM.characters_per_row


def test_a_sentence_too_long_for_a_subtitle_is_cut_evenly_at_a_clause_end_where_one_lies_near() -> (
    None
):
    parts = TWO_ROWS.evenly(LONG_SENTENCE)
    assert " ".join(parts) == LONG_SENTENCE
    assert all(TWO_ROWS.holds(part) for part in parts)
    assert parts[0].endswith("simulation:")
    assert TWO_ROWS.evenly("short") == ["short"]


def test_short_sentences_are_joined_where_they_fit_together() -> None:
    joined = "It holds. It does."
    pieces = TWO_ROWS.pieces(f"{joined} {LONG_SENTENCE}")
    assert pieces[0] == joined
    assert pieces[1] == TWO_ROWS.evenly(LONG_SENTENCE)[0]


def test_a_cue_is_written_on_two_even_rows_when_it_needs_two() -> None:
    cue = SubtitleCue(
        seconds(0),
        seconds(1),
        "The framework provides a common query interface over heterogeneous",
    )
    rows = cue.rows(PLAYER_ROOM)
    rows_allowed, balance = 2, 12
    assert len(rows) == rows_allowed
    assert abs(len(rows[0]) - len(rows[1])) < balance
    assert all(len(row) <= PLAYER_ROOM.characters_per_row for row in rows)


def test_a_cue_that_fits_one_row_is_written_on_one() -> None:
    text = "short"
    assert SubtitleCue(seconds(0), seconds(1), text).rows(PLAYER_ROOM) == [text]


def test_cues_are_written_as_subrip() -> None:
    cues = [
        SubtitleCue(seconds(1), seconds(2.5), "One."),
        SubtitleCue(seconds(61.25), seconds(63.25), "Two."),
    ]
    assert SubRipFile(cues, PLAYER_ROOM).text == (
        f"1\n{cues[0].timestamps}\nOne.\n\n2\n{cues[1].timestamps}\nTwo.\n\n"
    )


def test_a_cue_names_its_moments_as_hours_minutes_seconds_and_milliseconds() -> None:
    cue = SubtitleCue(seconds(3661.5), seconds(3662.25), "x")
    assert cue.timestamps == "01:01:01,500 --> 01:01:02,250"
