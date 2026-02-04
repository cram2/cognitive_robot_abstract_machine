from enum import Enum

from krrood.ormatic.type_dict import TypeDict
from ..dataset.example_classes import (
    ChildEnum2,
    Position4D,
    Position,
    Position5D,
)


def test_type_dict():
    type_dict = TypeDict({float: 1, Enum: 2, Position: 3, Position4D: 4})

    assert type_dict[float] == 1
    assert type_dict[ChildEnum2] == 2
    assert type_dict[Position5D] == 4
    assert type_dict[Position] == 3
