"""
Tests for :mod:`crollywood.theme`: the colours and fonts scenes draw in.
"""

from __future__ import annotations


from semantic_digital_twin.spatial_types.spatial_types import Point2

from crollywood.frame import Frame, Resolution
from crollywood.lettering import FontFace
from crollywood.theme import VideoTheme

FRAME_SIZE = Resolution(width=200, height=100)
TEXT_SIZE = 24


def test_a_theme_fills_a_page_with_its_page_colour() -> None:
    theme = VideoTheme.dark()
    page = theme.page_of(FRAME_SIZE)
    assert page.resolution == FRAME_SIZE
    assert (page.pixels == tuple(theme.page.to_rgb_bytes())).all()


def test_text_set_in_a_theme_is_written_in_its_text_colour_and_typeface() -> None:
    theme = VideoTheme.dark()
    lettering = theme.typesetting(size=TEXT_SIZE, face=FontFace.BOLD)
    assert lettering.typeface == theme.typeface
    written = lettering.written(theme.page_of(FRAME_SIZE), "VideoTheme", Point2(10, 50))
    inks = {tuple(pixel) for pixel in written.pixels.reshape(-1, 3)}
    assert tuple(theme.text.to_rgb_bytes()) in inks


def test_the_neutral_theme_is_dark_on_light_and_the_dark_theme_light_on_dark() -> None:
    neutral, dark = VideoTheme.neutral(), VideoTheme.dark()
    assert sum(neutral.page.to_rgb()) > sum(neutral.text.to_rgb())
    assert sum(dark.page.to_rgb()) < sum(dark.text.to_rgb())


def test_footage_colours_do_not_depend_on_the_theme() -> None:
    neutral, dark = VideoTheme.neutral(), VideoTheme.dark()
    assert neutral.footage_text == dark.footage_text
    assert neutral.footage_badge == dark.footage_badge
