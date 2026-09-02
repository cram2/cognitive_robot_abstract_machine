"""
Tests for reading and writing a color as a hex string.
"""

import pytest

from semantic_digital_twin.exceptions import MalformedHexColor
from semantic_digital_twin.world_description.geometry import Color

# %% reading a hex string


def test_a_hex_string_names_one_channel_per_pair_of_digits():
    color = Color.from_hex("#4080C0")
    assert color.to_rgb() == (0x40 / 255, 0x80 / 255, 0xC0 / 255)


def test_a_hex_string_without_an_alpha_pair_is_fully_opaque():
    assert Color.from_hex("#4080C0").A == Color().A


def test_a_fourth_pair_of_digits_names_the_opacity():
    assert Color.from_hex("#4080C020").A == 0x20 / 255


def test_the_leading_hash_is_optional():
    assert Color.from_hex("4080C0") == Color.from_hex("#4080C0")


def test_a_hex_string_is_read_regardless_of_the_case_it_is_written_in():
    assert Color.from_hex("#4080c0") == Color.from_hex("#4080C0")


# %% rejecting what is not a hex string


@pytest.mark.parametrize(
    "malformed", ["#4080C", "#4080C0F", "", "#", "#4080GG", "#4080C0 "]
)
def test_a_string_that_is_not_a_color_is_rejected(malformed):
    """
    A malformed literal that is read as a color silently draws the wrong thing, so it
    has to be refused where it is read.
    """
    with pytest.raises(MalformedHexColor):
        Color.from_hex(malformed)


# %% writing a hex string


def test_a_color_read_from_a_hex_string_is_written_back_as_that_string():
    literal = "#4080C0"
    assert Color.from_hex(literal).to_hex() == literal


def test_the_written_hex_string_leaves_out_the_opacity():
    """
    :meth:`Color.to_hex` is the counterpart of :meth:`Color.to_rgb`, so a translucent
    color is written as the same three channels an opaque one is.
    """
    translucent = Color.from_hex("#4080C020")
    assert translucent.to_hex() == Color.from_hex("#4080C0").to_hex()


# %% colors as set and dictionary members


def test_equal_colors_are_one_and_the_same_set_member():
    assert len({Color.from_hex("#4080C0"), Color.from_hex("#4080C0")}) == 1


def test_colors_differing_only_in_opacity_are_told_apart():
    assert len({Color.from_hex("#4080C0"), Color.from_hex("#4080C020")}) == 2
