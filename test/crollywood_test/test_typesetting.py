"""
Tests for :mod:`crollywood.typesetting`: text written where asked and broken to fit a
width.
"""

from __future__ import annotations

import numpy as np
from semantic_digital_twin.spatial_types.spatial_types import Point2

from crollywood.frame import Resolution
from crollywood.typesetting import Anchor, TextSize, Typesetting

FRAME_SIZE = Resolution(width=200, height=100)
TEXT_SIZE = 20


def test_text_is_written_where_asked() -> None:
    left, middle = 10, 50
    written = Typesetting(size=TEXT_SIZE).written(
        FRAME_SIZE.blank(255), "ab", Point2(left, middle), Anchor.LEFT_MIDDLE
    )
    rows, columns = np.nonzero(written.pixels[:, :, 0] < 128)
    assert left <= columns.min() < left + 5
    assert middle - 10 <= rows.min() < middle <= rows.max() < middle + 12


def test_several_rows_centred_on_a_point_hang_half_their_height_each_side() -> None:
    centre = Point2(100, 50)
    written = Typesetting(size=TEXT_SIZE).written(
        FRAME_SIZE.blank(255), "a\nb", centre, Anchor.CENTRE_MIDDLE
    )
    rows = np.nonzero(written.pixels[:, :, 0].min(axis=1) < 128)[0]
    assert abs((rows.min() + rows.max()) / 2 - float(centre.y)) < TEXT_SIZE / 2


def test_wrapping_breaks_at_spaces_to_fit_the_width() -> None:
    setting = Typesetting(size=TEXT_SIZE)
    text = "one two three four"
    narrow = setting.width_of(text) / 2
    wrapped = setting.wrapped(text, narrow)
    assert wrapped.count("\n") >= 1
    assert all(setting.width_of(line) <= narrow for line in wrapped.split("\n"))
    assert wrapped.replace("\n", " ") == text


def test_the_text_sizes_are_ordered_from_label_to_claim() -> None:
    assert TextSize.LABEL < TextSize.BODY < TextSize.CLAIM
