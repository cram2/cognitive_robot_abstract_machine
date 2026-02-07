from enum import Enum

from krrood.ormatic.type_dict import TypeDict
from ..dataset.example_classes import (
    ChildEnum2,
    KRROODPosition4D,
    KRROODPosition,
    KRROODPosition5D,
)


def test_type_dict():
    type_dict = TypeDict({float: 1, Enum: 2, KRROODPosition: 3, KRROODPosition4D: 4})

    assert type_dict[float] == 1
    assert type_dict[ChildEnum2] == 2
    assert type_dict[KRROODPosition5D] == 4
    assert type_dict[KRROODPosition] == 3
