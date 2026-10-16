"""
Tests for :mod:`crollywood.frame`: frames of a size, pictures pasted and fitted into
them, and the shapes drawn on them.
"""

from __future__ import annotations


import numpy as np
import pytest
from semantic_digital_twin.spatial_types.spatial_types import Point2
from semantic_digital_twin.world_description.geometry import Color

from crollywood.frame import Frame, Pixel, PixelBox, Resolution

FRAME_SIZE = Resolution(width=200, height=100)
PICTURE_GREY = 200

# %% the size of a frame


def test_the_standard_resolution_is_720_rows_by_1280_columns() -> None:
    standard = Resolution.standard()
    assert (standard.height, standard.width) == (720, 1280)


def test_a_scene_may_draw_down_to_the_band_left_for_subtitles() -> None:
    resolution = Resolution(width=1600, height=900)
    assert resolution.stage_height == pytest.approx(
        resolution.height * (1 - resolution.subtitle_band_share)
    )


def test_a_blank_frame_is_one_grey_and_knows_its_size() -> None:
    grey = 77
    blank = FRAME_SIZE.blank(grey)
    assert blank.pixels.shape == FRAME_SIZE.shape
    assert (blank.pixels == grey).all()
    assert blank.resolution == FRAME_SIZE


def test_a_frame_of_a_colour_holds_its_bytes() -> None:
    color = Color.from_hex("#1F2328")
    frame = Frame.of_color(FRAME_SIZE, color)
    assert (frame.pixels == tuple(color.to_rgb_bytes())).all()


def test_the_whole_frame_box_covers_every_pixel() -> None:
    box = FRAME_SIZE.whole
    assert PixelBox.nearest(box) == PixelBox.between(
        0, 0, FRAME_SIZE.width, FRAME_SIZE.height
    )


# %% boxes


def test_fitting_keeps_the_aspect_and_centres_in_the_room() -> None:
    room = FRAME_SIZE.whole
    square = Frame.fitting(room, 1.0)
    assert PixelBox.nearest(square) == PixelBox.between(50, 0, 150, 100)
    wide_aspect = 4.0
    wide = Frame.fitting(room, wide_aspect)
    assert PixelBox.nearest(wide) == PixelBox.between(0, 25, 200, 75)


def test_a_box_inside_a_frame_is_cut_to_the_frame() -> None:
    hanging_over = PixelBox.between(-10, 90, 40, 120)
    assert hanging_over.inside(FRAME_SIZE) == PixelBox.between(
        0, 90, 40, FRAME_SIZE.height
    )


def test_a_box_outside_a_frame_covers_no_pixel_of_it() -> None:
    beyond = PixelBox.between(FRAME_SIZE.width + 5, 0, FRAME_SIZE.width + 15, 10)
    assert beyond.inside(FRAME_SIZE).is_empty


def test_a_pixel_box_runs_from_its_top_left_pixel_to_its_bottom_right_one() -> None:
    left, top, width, height = 10, 20, 10, 5
    box = PixelBox.between(left, top, left + width, top + height)
    assert box.lower == Pixel(column=left, row=top)
    assert box.upper == Pixel(column=left + width - 1, row=top + height - 1)
    assert box.width == width and box.height == height


def test_a_position_falls_on_the_nearest_pixel() -> None:
    assert Pixel.nearest(Point2(3.4, 7.6)) == Pixel(column=3, row=8)


# %% pasting


def test_pasting_resizes_the_picture_into_the_box() -> None:
    frame = FRAME_SIZE.blank(0)
    picture = Resolution(10, 10).blank(PICTURE_GREY)
    left, top, width, height = 20, 30, 40, 20
    result = frame.pasted(picture, Frame.box(left, top, width, height))
    inside = result.pixels[top : top + height, left : left + width]
    assert inside.min() == PICTURE_GREY
    assert result.pixels[top - 1, left].max() == 0
    assert result.pixels[top, left - 1].max() == 0
    assert frame.pixels.max() == 0


def test_pasting_crops_what_falls_outside_the_frame() -> None:
    frame = FRAME_SIZE.blank(0)
    picture = Resolution(10, 10).blank(PICTURE_GREY)
    left, top, width, height = 180, -10, 40, 30
    result = frame.pasted(picture, Frame.box(left, top, width, height))
    assert (
        result.pixels[0 : top + height, left : FRAME_SIZE.width].min() == PICTURE_GREY
    )
    assert result.pixels.shape == frame.pixels.shape


def test_fitting_a_picture_keeps_its_aspect() -> None:
    frame = FRAME_SIZE.blank(0)
    picture = Resolution(40, 10).blank(PICTURE_GREY)
    result = frame.fitted(picture, FRAME_SIZE.whole)
    top, bottom = 25, 75
    assert result.pixels[top:bottom, :].min() == PICTURE_GREY
    assert result.pixels[top - 1, :].max() == 0
    assert result.pixels[bottom, :].max() == 0


# %% tone


def test_dimming_moves_towards_white() -> None:
    assert FRAME_SIZE.blank(0).dimmed(0.5).pixels[0, 0, 0] == 128


def test_blending_mixes_two_frames_by_the_weight() -> None:
    dark, light = FRAME_SIZE.blank(0), FRAME_SIZE.blank(100)
    assert dark.blended(light, 0.0).pixels.max() == 0
    assert dark.blended(light, 1.0).pixels.min() == 100
    assert dark.blended(light, 0.5).pixels[0, 0, 0] == 50


# %% shapes


def test_filling_a_box_paints_it_in_the_colour() -> None:
    color = Color.from_hex("#FF8800")
    filled = FRAME_SIZE.blank(0).filled(Frame.box(10, 10, 20, 20), color)
    assert tuple(filled.pixels[20, 20]) == tuple(color.to_rgb_bytes())
    assert filled.pixels[50, 50].max() == 0


def test_framing_a_box_outlines_it_and_leaves_its_inside() -> None:
    color = Color.WHITE()
    framed = FRAME_SIZE.blank(0).framed(Frame.box(20, 20, 60, 40), color)
    assert framed.pixels[20, 50].min() == 255
    assert framed.pixels[40, 50].max() == 0


def test_a_line_and_an_arrow_are_drawn_between_their_points() -> None:
    color = Color.WHITE()
    start, end = Point2(10, 50), Point2(190, 50)
    lined = FRAME_SIZE.blank(0).lined(start, end, color)
    arrowed = FRAME_SIZE.blank(0).arrowed(start, end, color)
    assert lined.pixels[50, 100].min() == 255 and lined.pixels[10, 100].max() == 0
    assert arrowed.pixels[50, 100].min() == 255
    assert np.count_nonzero(arrowed.pixels) > np.count_nonzero(lined.pixels)
