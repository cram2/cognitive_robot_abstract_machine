"""
Tests for how far hues lie from a colour's, the short way round the hue wheel.
"""

from __future__ import annotations

import numpy as np
import pytest

from semantic_digital_twin.world_description.geometry import Color

# %% how far apart two hues lie


def test_hues_are_compared_the_short_way_round_the_hue_wheel():
    color = Color(hue=0.01, saturation=1.0, value=1.0)

    assert color.hue_distance(hues=0.98) == pytest.approx(0.03)


def test_hues_on_the_same_side_of_red_are_compared_directly():
    color = Color(hue=0.2, saturation=1.0, value=1.0)

    assert color.hue_distance(hues=0.25) == pytest.approx(0.05)


def test_hues_are_compared_one_by_one_against_an_array_of_them():
    color = Color(hue=0.1, saturation=1.0, value=1.0)

    distances = color.hue_distance(hues=np.array([[0.1, 0.4, 0.9]]))

    assert distances == pytest.approx(np.array([[0.0, 0.3, 0.2]]))


def test_a_hue_is_the_same_whichever_turn_it_is_written_in():
    color = Color(hue=0.1, saturation=1.0, value=1.0)

    assert color.hue_distance(hues=1.1) == pytest.approx(0.0)
