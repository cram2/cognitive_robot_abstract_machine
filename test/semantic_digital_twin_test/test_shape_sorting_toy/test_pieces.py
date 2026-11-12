"""
Tests for the lab's piece sets: the pieces each one holds, and what is read off them.
"""

from __future__ import annotations

import pytest

from semantic_digital_twin.shape_sorting_toy.semantic_annotations import (
    CubePiece,
    CylinderPiece,
    DiskPiece,
    RectangularPrismPiece,
    TriangularPrismPiece,
)
from semantic_digital_twin.shape_sorting_toy.pieces import PieceColor, PieceSet

# %% the sets

KINDS_IN_EACH_SET = (
    CubePiece,
    CylinderPiece,
    RectangularPrismPiece,
    TriangularPrismPiece,
)
"""
The kinds of piece each of the lab's sets holds one of.
"""


@pytest.fixture(scope="module")
def full_size() -> PieceSet:
    return PieceSet.full_size(grasp_height=0.01)


@pytest.fixture(scope="module")
def smaller() -> PieceSet:
    return PieceSet.smaller(grasp_height=0.01)


def test_a_set_holds_one_piece_of_each_kind(full_size):
    kinds = [type(piece) for piece in full_size.pieces]

    assert sorted(kinds, key=lambda kind: kind.__name__) == sorted(
        KINDS_IN_EACH_SET, key=lambda kind: kind.__name__
    )


def test_a_kind_of_piece_the_set_has_none_of_is_not_found(full_size):
    assert full_size.find_piece(kind=DiskPiece) is None


def test_every_piece_is_gripped_at_the_given_height():
    pieces = PieceSet.full_size(grasp_height=0.012).pieces

    assert {piece.grasp_height for piece in pieces} == {0.012}


# %% what is read off a piece


def test_a_cube_covers_a_square_as_wide_as_it_stands_tall(full_size):
    cube = full_size.find_piece(kind=CubePiece)

    assert cube.footprint.bounding_box.dimensions == pytest.approx(
        [cube.height, cube.height]
    )


def test_a_piece_wears_the_colour_it_was_given(full_size):
    assert full_size.find_piece(kind=CubePiece).color == PieceColor.CYAN.value


def test_the_triangular_prism_is_centred_on_its_centroid(full_size):
    prism = full_size.find_piece(kind=TriangularPrismPiece)

    assert prism.footprint.centroid.to_np() == pytest.approx([0.0, 0.0], abs=1e-9)


def test_the_set_wears_each_of_its_colours_once(full_size):
    assert full_size.colors == (PieceColor.YELLOW.value, PieceColor.CYAN.value)


# %% the smaller set


@pytest.mark.parametrize("kind", KINDS_IN_EACH_SET)
def test_the_smaller_set_is_the_full_size_set_at_four_fifths_of_its_size(
    full_size, smaller, kind
):
    large = full_size.find_piece(kind=kind)
    small = smaller.find_piece(kind=kind)

    assert small.footprint.area == pytest.approx(large.footprint.area * 0.8**2)
    assert small.height == pytest.approx(large.height * 0.8)


def test_the_smaller_set_wears_colours_of_its_own(smaller):
    assert smaller.colors == (
        PieceColor.SMALLER_SET_YELLOW.value,
        PieceColor.SMALLER_SET_BLUE.value,
    )
