import collections.abc
from enum import Enum

from krrood.ormatic.type_dict import TypeDict
from ..dataset.example_classes import (
    ChildEnum2,
    KRROODPosition4D,
    KRROODPosition,
    KRROODPosition5D,
)


def test_type_dict():
    type_dict = TypeDict(
        _dict={float: 1, Enum: 2, KRROODPosition: 3, KRROODPosition4D: 4}
    )

    assert type_dict[float] == 1
    assert type_dict[ChildEnum2] == 2
    assert type_dict[KRROODPosition5D] == 4
    assert type_dict[KRROODPosition] == 3


# %% virtual subclasses


def test_a_virtual_subclass_finds_the_entry_of_its_abstract_base_class():
    """
    A class an abstract base class claims without inheriting from it gets the entry of
    that abstract base class, unless an entry of its own is closer.
    """
    type_dict = TypeDict(_dict={collections.abc.Sequence: 1, tuple: 2})

    assert type_dict[list] == 1
    assert type_dict[tuple] == 2
