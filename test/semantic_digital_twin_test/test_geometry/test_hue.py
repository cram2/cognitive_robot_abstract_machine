"""
Tests for a hue, where a colour lies on the colour circle, and for reading and building
a colour by its hue.
"""

import numpy as np
import pytest

from semantic_digital_twin.world_description.geometry import Color, Hue

# %% how far apart two hues lie


def test_hues_are_compared_the_short_way_round_the_colour_circle():
    assert Hue(2.0).distance(Hue.full_turn - 3.0) == pytest.approx(5.0)


def test_hues_on_the_same_side_of_red_are_compared_directly():
    assert Hue(20.0).distance(25.0) == pytest.approx(5.0)


def test_hues_are_compared_one_by_one_against_an_array_of_them():
    distances = Hue(10.0).distance(np.array([[10.0, 40.0, Hue.full_turn - 20.0]]))

    assert distances == pytest.approx(np.array([[0.0, 30.0, 30.0]]))


def test_a_hue_is_the_same_whichever_full_turn_it_is_written_in():
    assert Hue(10.0).distance(10.0 + Hue.full_turn) == pytest.approx(0.0)


# %% colours built from and read as a hue


def test_a_colour_at_hue_zero_with_full_saturation_and_value_is_pure_red():
    assert Color.from_hsv(Hue(0.0)) == Color.RED()


def test_a_colour_a_third_of_the_way_round_is_pure_green():
    assert Color.from_hsv(Hue(Hue.full_turn / 3)) == Color.GREEN()


def test_a_colour_with_no_saturation_is_grey_at_its_value():
    assert Color.from_hsv(Hue(0.0), saturation=0.0, value=0.5) == Color(0.5, 0.5, 0.5)


def test_a_colour_is_read_back_at_the_hue_it_was_built_from():
    built_from = Hue(172.0)

    assert Color.from_hsv(built_from).hue.distance(built_from.degrees) == pytest.approx(
        0.0
    )


def test_a_whole_degree_hue_is_read_back_exactly_from_the_colour_built_from_it():
    for degrees in range(0, int(Hue.full_turn)):
        assert Color.from_hsv(Hue(float(degrees))).hue == Hue(float(degrees))
