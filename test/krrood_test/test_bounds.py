from dataclasses import dataclass, fields

import pytest

from krrood.bounds import Bounds, BoundsInterface
from krrood.patterns.exceptions import InterfaceInstantiationError

# %% the corners of an axis-aligned region


def test_the_bounds_interface_itself_cannot_be_instantiated():
    with pytest.raises(InterfaceInstantiationError):
        BoundsInterface()


def test_bounds_stored_as_their_corners_implement_the_bounds_interface():
    assert isinstance(Bounds(0.0, 1.0), BoundsInterface)


def test_bounds_bound_to_a_type_store_corners_of_that_type():
    @dataclass
    class IntegerBounds(Bounds[int]): ...

    lower_field, upper_field = fields(IntegerBounds)
    assert (lower_field.type, upper_field.type) == (int, int)
