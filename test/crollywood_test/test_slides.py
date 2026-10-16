"""
Tests for :mod:`crollywood.slides`: a table of data and a text card, drawn in greyscale
on white and held as long as they are told.
"""

from __future__ import annotations


from datetime import timedelta

import numpy as np
from numpy.typing import NDArray

from crollywood.frame import Frame
from crollywood.slides import TabularDataRow, TabularDataSlide, HeldSlide, TextCard
from crollywood.theme import VideoTheme

ROWS = (
    TabularDataRow("Trials that succeeded", "4 of 4"),
    TabularDataRow("Questions answered", "213 of 213"),
)
MOMENT = timedelta(seconds=1)
TABLE_HOLD = timedelta(seconds=4)
CARD_HOLD = timedelta(seconds=5)
EDGE_WIDTH = 120
CARD_TOP_BLANK_ROWS = 200
THEME_TOP_BLANK_ROWS = 120
GREY_TOLERANCE = 16
WHITE = 255
DARK = 128


def coloured_pixels(frame: Frame) -> NDArray[np.uint8]:
    """
    :return: The pixels of a frame whose channels differ: any that are not grey.
    """
    pixels = frame.pixels
    return pixels[(pixels.max(axis=2) - pixels.min(axis=2)) > GREY_TOLERANCE]


def test_the_table_is_drawn_centred_on_white_with_hairlines_only() -> None:
    table = TabularDataSlide("On the real robot", ROWS, held_for=TABLE_HOLD)
    assert isinstance(table, HeldSlide) and table.duration == TABLE_HOLD
    pixels = table.frame_at(MOMENT).pixels
    assert pixels.shape == table.resolution.shape
    assert coloured_pixels(Frame(pixels)).size == 0
    assert (pixels[:, :EDGE_WIDTH] == WHITE).all()
    assert (pixels[:, -EDGE_WIDTH:] == WHITE).all()
    assert pixels.min() < DARK


def test_a_text_card_writes_its_heading_and_line_above_the_subtitle_band() -> None:
    card = TextCard("The code", "https://example.org/code", held_for=CARD_HOLD)
    assert isinstance(card, HeldSlide) and card.duration == CARD_HOLD
    pixels = card.frame_at(MOMENT).pixels
    assert pixels.min() < DARK
    assert (pixels[int(card.resolution.stage_height) :] == WHITE).all()
    assert (pixels[:CARD_TOP_BLANK_ROWS] == WHITE).all()


def test_a_text_card_writes_both_its_texts() -> None:
    both = TextCard("The code", "a line").frame_at(timedelta(0))
    heading_alone = TextCard("The code", "").frame_at(timedelta(0))
    assert not np.array_equal(both.pixels, heading_alone.pixels)


def test_slides_are_drawn_on_the_page_of_their_theme_in_its_text_colour() -> None:
    theme = VideoTheme.dark()
    for slide in (
        TextCard("The code", "a line", theme=theme),
        TabularDataSlide("On the real robot", ROWS, theme=theme),
    ):
        pixels = slide.frame_at(timedelta(0)).pixels
        assert (pixels[:THEME_TOP_BLANK_ROWS] == tuple(theme.page.to_rgb_bytes())).all()
        inks = {tuple(pixel) for pixel in pixels.reshape(-1, 3)}
        assert tuple(theme.text.to_rgb_bytes()) in inks
