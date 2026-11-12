"""
Tests for telling a measured colour apart from the colours a set's pieces wear.
"""

from __future__ import annotations

import pytest

from experiments.montessori.pieces import PieceColorMatcher
from semantic_digital_twin.semantic_annotations.semantic_annotations import CubePiece
from semantic_digital_twin.shape_sorting_toy.pieces import PieceColor, PieceSet
from semantic_digital_twin.world_description.geometry import Color

# %% telling a measured colour apart


@pytest.fixture(scope="module")
def matcher() -> PieceColorMatcher:
    return PieceColorMatcher(piece_set=PieceSet.full_size(grasp_height=0.01))


def _hue_shifted(color: Color, turns: float) -> Color:
    return Color(hue=(color.hue + turns) % 1.0, saturation=1.0, value=1.0)


def test_a_colour_is_taken_for_a_piece_only_within_the_tolerance(matcher):
    cube = matcher.piece_set.find_piece(kind=CubePiece)

    assert matcher.could_be(
        piece=cube, measured=_hue_shifted(cube.color, matcher.hue_tolerance / 2)
    )
    assert not matcher.could_be(
        piece=cube, measured=_hue_shifted(cube.color, matcher.hue_tolerance * 1.5)
    )


def test_a_piece_could_be_anything_when_no_colour_was_measured(matcher):
    cube = matcher.piece_set.find_piece(kind=CubePiece)

    assert matcher.could_be(piece=cube, measured=None)


def test_a_hue_is_a_piece_colour_only_near_one_of_the_sets_colours(matcher):
    hues = [
        [
            PieceColor.CYAN.value.hue,
            PieceColor.YELLOW.value.hue + matcher.hue_tolerance / 2,
            PieceColor.CYAN.value.hue + 1 / 6,
        ]
    ]

    assert matcher.is_piece_color(hues=hues).tolist() == [[True, True, False]]
