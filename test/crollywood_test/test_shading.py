"""
Tests for :mod:`crollywood.shading`: areas of a frame shaded towards black, evenly or
deepening from their top edge.
"""

from __future__ import annotations

import pytest

from crollywood.exceptions import DarknessOutOfRange
from crollywood.frame import Frame, Resolution
from crollywood.shading import EvenShade, GradientShade

FRAME_SIZE = Resolution(width=200, height=100)
GREY = 200

# %% the darkness of a shade


@pytest.mark.parametrize("shade_type", [EvenShade, GradientShade])
@pytest.mark.parametrize("darkness", [-0.1, 1.1])
def test_a_shade_darker_than_black_or_lighter_than_nothing_is_rejected(
    shade_type: type, darkness: float
) -> None:
    with pytest.raises(DarknessOutOfRange):
        shade_type(darkness)


# %% shading evenly


def test_an_even_shade_darkens_inside_its_box_and_leaves_the_rest() -> None:
    shade = EvenShade(0.5, corner_radius=5)
    result = shade.over(FRAME_SIZE.blank(GREY), Frame.box(50, 20, 100, 60))
    assert result.pixels[50, 100, 0] == pytest.approx(
        GREY * (1 - shade.darkness), abs=1
    )
    assert (result.pixels[:15] == GREY).all()


def test_an_even_shade_rounds_its_corners_by_its_radius() -> None:
    box = Frame.box(50, 20, 100, 60)
    square = EvenShade(0.5).over(FRAME_SIZE.blank(GREY), box)
    rounded = EvenShade(0.5, corner_radius=10).over(FRAME_SIZE.blank(GREY), box)
    corner = (20, 50)
    assert square.pixels[corner][0] < GREY
    assert rounded.pixels[corner][0] == GREY


# %% shading deeper towards the bottom


def test_a_gradient_shade_deepens_from_its_top_edge() -> None:
    shade = GradientShade(0.8)
    result = shade.over(FRAME_SIZE.blank(GREY), Frame.box(0, 50, 200, 50))
    assert (result.pixels[:50] == GREY).all()
    assert result.pixels[50, 0, 0] == GREY
    assert result.pixels[99, 0, 0] == pytest.approx(GREY * (1 - shade.darkness), abs=1)


def test_a_gradient_shade_is_at_its_full_darkness_from_its_share_of_the_height_down() -> (
    None
):
    shade = GradientShade(0.8, deepening_share=0.5)
    top, height = 50, 50
    result = shade.over(FRAME_SIZE.blank(GREY), Frame.box(0, top, 200, height))
    full_from = top + round(height * shade.deepening_share)
    assert result.pixels[full_from - 5, 0, 0] > result.pixels[full_from, 0, 0]
    assert (result.pixels[full_from:, 0, 0] == result.pixels[99, 0, 0]).all()
