"""
Tests for :mod:`crollywood.code`: a line of Python cut into pieces, compared with an
earlier line, and written in colour.
"""

from __future__ import annotations


from semantic_digital_twin.spatial_types.spatial_types import Point2

from crollywood.code import CodeLine, CodeTypesetting, LineSpan, Token
from crollywood.frame import Frame, Resolution
from crollywood.lettering import FontFace
from crollywood.theme import VideoTheme
from crollywood.typesetting import Typesetting

FRAME_SIZE = Resolution(width=600, height=100)
CODE_SIZE = 24
QUERY = 'an(entity(trial.episode).where(contains(name, "cube"), 3))'


def inks_of(frame: Frame) -> set:
    """
    :return: The colours a frame holds, as bytes.
    """
    return {tuple(pixel) for pixel in frame.pixels.reshape(-1, 3)}


def test_a_line_of_code_comes_apart_into_calls_classes_names_strings_and_numbers() -> (
    None
):
    pieces = CodeLine(QUERY).pieces
    kinds = {piece.text: piece.token for piece in pieces}
    assert kinds["an"] is Token.CALL and kinds["where"] is Token.CALL
    assert kinds["trial"] is Token.NAME and kinds["episode"] is Token.NAME
    assert kinds['"cube"'] is Token.STRING and kinds["3"] is Token.NUMBER
    assert kinds["("] is Token.PUNCTUATION
    assert CodeLine("a(MotionEvent)").pieces[2].token is Token.CLASS
    assert "".join(piece.text for piece in pieces) == QUERY


def test_keywords_and_comments_are_pieces_of_their_own() -> None:
    kinds = {
        piece.text: piece.token
        for piece in CodeLine("for event in events:  # every event").pieces
    }
    assert kinds["for"] is Token.KEYWORD and kinds["in"] is Token.KEYWORD
    assert kinds["# every event"] is Token.COMMENT


def test_a_formatted_string_is_coloured_as_a_string_around_its_names() -> None:
    kinds = {piece.text: piece.token for piece in CodeLine('f"{name}!"').pieces}
    assert kinds['f"{'] is Token.STRING and kinds['}!"'] is Token.STRING
    assert kinds["name"] is Token.NAME


def test_the_changed_spans_of_a_line_are_the_pieces_that_differ_from_the_earlier_line() -> (
    None
):
    before = CodeLine("    contains(tick.events, motion := a(MotionEvent)()),")
    after = CodeLine("    contains(tick.events, pickup := a(PickUpEvent)()),")
    spans = after.changed_spans(before)
    assert [after.text[span.first : span.after_last] for span in spans] == [
        "pickup",
        "PickUpEvent",
    ]
    assert after.changed_spans(after) == []
    new_line = CodeLine("a(Tick)()")
    assert new_line.changed_spans(CodeLine("")) == [LineSpan(0, len(new_line.text))]


def test_coloured_code_is_as_wide_as_the_same_line_in_one_colour_and_carries_its_colours() -> (
    None
):
    line = CodeLine('a(MotionEvent)("x")')
    code = CodeTypesetting(size=CODE_SIZE)
    assert code.width_of(line.text) == Typesetting(
        size=CODE_SIZE, face=FontFace.MONO
    ).width_of(line.text)
    blank = Frame.of_color(FRAME_SIZE, VideoTheme.neutral().page)
    inks = inks_of(code.written(blank, line, Point2(10, 30)))
    for token in (Token.CLASS, Token.STRING, Token.CALL):
        assert tuple(code.color_of(token).to_rgb_bytes()) in inks


def test_a_marked_stretch_of_code_is_filled_behind_in_the_marker_and_nothing_else_is() -> (
    None
):
    line = CodeLine("pickup := a(PickUpEvent)()")
    code = CodeTypesetting(size=CODE_SIZE)
    marker = tuple(code.theme.marker.to_rgb_bytes())
    blank = Frame.of_color(FRAME_SIZE, code.theme.page)
    start = Point2(10, 30)
    marked = code.written(blank, line, start, marked=[LineSpan(10, 24)])
    plain = code.written(blank, line, start)
    assert marker in inks_of(marked)
    assert marker not in inks_of(plain)
    before_the_mark = int(10 + code.width_of(line.text[:2]) + 1)
    assert tuple(marked.pixels[30, before_the_mark]) != marker


def test_code_is_coloured_in_the_colours_of_its_theme() -> None:
    dark = CodeTypesetting(size=CODE_SIZE, theme=VideoTheme.dark())
    neutral = CodeTypesetting(size=CODE_SIZE)
    line = CodeLine('Tick(name="pick")')
    written = dark.written(dark.theme.page_of(FRAME_SIZE), line, Point2(10, 30))
    inks = inks_of(written)
    assert tuple(dark.color_of(Token.CLASS).to_rgb_bytes()) in inks
    assert tuple(dark.color_of(Token.STRING).to_rgb_bytes()) in inks
    assert tuple(neutral.color_of(Token.CLASS).to_rgb_bytes()) not in inks
