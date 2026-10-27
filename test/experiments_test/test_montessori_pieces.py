"""
Tests for the lab's piece sets: the pieces each one holds, what is read off their
bodies, and telling a measured colour apart from theirs.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.montessori.pieces import PieceHue, PieceSet
from experiments.montessori.semantics import (
    CubeShape,
    CylinderShape,
    DiskShape,
    RectangularPrismShape,
    TriangularPrismShape,
)
from semantic_digital_twin.world_description.geometry import Hue

KINDS_IN_EACH_SET = (
    CubeShape,
    CylinderShape,
    RectangularPrismShape,
    TriangularPrismShape,
)
"""
The kinds of piece each of the lab's sets holds one of.
"""


@pytest.fixture(scope="module")
def full_size() -> PieceSet:
    return PieceSet.full_size()


@pytest.fixture(scope="module")
def smaller() -> PieceSet:
    return PieceSet.smaller()


# %% the pieces a set holds


def test_a_set_holds_one_piece_of_each_kind(full_size: PieceSet):
    kinds = [type(piece) for piece in full_size.pieces]

    assert sorted(kinds, key=lambda kind: kind.__name__) == sorted(
        KINDS_IN_EACH_SET, key=lambda kind: kind.__name__
    )


def test_a_kind_of_piece_the_set_has_none_of_is_not_found(full_size: PieceSet):
    assert full_size.piece_for(DiskShape) is None


# %% what is read off a piece's body


def test_a_cube_covers_a_square_as_wide_as_it_stands_tall(full_size: PieceSet):
    cube = full_size.piece_for(CubeShape)

    extent = PieceSet.outline_of(cube).bounding_box

    assert extent.depth == pytest.approx(PieceSet.height_of(cube))
    assert extent.width == pytest.approx(PieceSet.height_of(cube))


def test_a_piece_is_read_at_the_hue_it_was_printed_in(full_size: PieceSet):
    assert PieceSet.hue_of(full_size.piece_for(CubeShape)) == Hue(PieceHue.CYAN.degrees)


def test_two_pieces_printed_in_one_colour_wear_one_hue(full_size: PieceSet):
    cube = full_size.piece_for(CubeShape)
    cylinder = full_size.piece_for(CylinderShape)
    rectangular_prism = full_size.piece_for(RectangularPrismShape)

    assert PieceSet.hue_of(cube) == PieceSet.hue_of(cylinder)
    assert PieceSet.hue_of(cube) != PieceSet.hue_of(rectangular_prism)


def test_the_set_wears_each_of_its_colours_once(full_size: PieceSet):
    assert [hue.degrees for hue in full_size.hues] == sorted(
        [PieceHue.CYAN.degrees, PieceHue.YELLOW.degrees]
    )


# %% the smaller set


@pytest.mark.parametrize("kind", KINDS_IN_EACH_SET)
def test_the_smaller_set_is_the_full_size_set_at_four_fifths_of_its_size(
    full_size: PieceSet, smaller: PieceSet, kind
):
    large = full_size.piece_for(kind)
    small = smaller.piece_for(kind)

    assert PieceSet.outline_of(small).area == pytest.approx(
        PieceSet.outline_of(large).area * 0.8**2
    )
    assert PieceSet.height_of(small) == pytest.approx(PieceSet.height_of(large) * 0.8)


def test_the_smaller_set_wears_colours_of_its_own(smaller: PieceSet):
    assert [hue.degrees for hue in smaller.hues] == sorted(
        [PieceHue.SMALLER_SET_BLUE.degrees, PieceHue.SMALLER_SET_YELLOW.degrees]
    )


# %% telling a measured colour apart


def test_a_colour_is_taken_for_a_piece_only_within_the_tolerance(full_size: PieceSet):
    cube = full_size.piece_for(CubeShape)
    worn = PieceSet.hue_of(cube).degrees

    assert full_size.could_be(cube, Hue(worn + full_size.hue_tolerance))
    assert not full_size.could_be(cube, Hue(worn + full_size.hue_tolerance + 1))
    assert full_size.could_be(cube, None)


def test_a_pixel_wears_a_piece_colour_only_near_one_of_the_sets_hues(
    full_size: PieceSet,
):
    degrees = np.array(
        [
            [
                PieceHue.CYAN.degrees,
                PieceHue.YELLOW.degrees + full_size.hue_tolerance,
                PieceHue.CYAN.degrees + 60,
            ]
        ]
    )

    assert full_size.wears_a_piece_colour(degrees).tolist() == [[True, True, False]]
