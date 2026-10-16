"""
Tests for :mod:`krrood.units`: amounts kept apart by their unit, and how fast they flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pytest

from krrood.exceptions import DifferentQuantities
from krrood.units import BitRate, DataSize, Quantity

TEN_SECONDS = timedelta(seconds=10)

# %% a quantity counted in fractions


@dataclass(order=True)
class FractionalQuantity(Quantity[float]):
    """
    A quantity counted in fractions of its unit, standing in for any such quantity.
    """


# %% amounts of data


def test_a_byte_is_eight_bits() -> None:
    assert DataSize.of_bytes(3) == DataSize(3 * DataSize.bits_per_byte)
    assert DataSize(3 * DataSize.bits_per_byte).bytes == 3


def test_amounts_of_data_add_subtract_and_compare() -> None:
    small, large = DataSize(100), DataSize(250)
    assert small + large == DataSize(350)
    assert large - small == DataSize(150)
    assert small < large


def test_an_amount_of_data_scales_to_the_nearest_bit() -> None:
    assert DataSize(10).scaled(0.25) == DataSize(2)


def test_dividing_an_amount_by_another_gives_how_often_it_fits() -> None:
    assert DataSize(300) / DataSize(100) == 3.0


# %% quantities counted in fractions


def test_a_fractional_quantity_scales_without_rounding() -> None:
    assert FractionalQuantity(1.0).scaled(0.25) == FractionalQuantity(0.25)


def test_amounts_of_different_quantities_do_not_add() -> None:
    with pytest.raises(DifferentQuantities):
        DataSize(1) + FractionalQuantity(1.0)


def test_amounts_of_different_quantities_are_not_equal() -> None:
    assert DataSize(1) != FractionalQuantity(1.0)


# %% rates


def test_a_bit_rate_over_a_duration_is_the_data_that_flows_in_it() -> None:
    assert BitRate(DataSize(8_000)).over(TEN_SECONDS) == DataSize(80_000)


def test_the_rate_filling_an_amount_in_a_duration_flows_that_amount_in_it() -> None:
    amount = DataSize.of_bytes(1_000)
    assert BitRate.filling(amount, TEN_SECONDS).over(TEN_SECONDS) == amount


def test_a_scaled_rate_is_still_a_bit_rate() -> None:
    scaled = BitRate(DataSize(1_000)).scaled(1.5)
    assert scaled == BitRate(DataSize(1_500))
    assert isinstance(scaled, BitRate)
