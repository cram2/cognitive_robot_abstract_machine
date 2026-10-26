from dataclasses import dataclass

import pytest

from krrood.patterns.exceptions import InterfaceInstantiationError
from krrood.patterns.interface import Interface

# %% an interface and the class implementing it


class Measurable(Interface):
    length: float


@dataclass
class Rod(Measurable):
    length: float


# %% instantiation


def test_an_interface_cannot_be_instantiated():
    with pytest.raises(InterfaceInstantiationError):
        Measurable()


def test_a_class_implementing_an_interface_can_be_instantiated():
    length = 2.0
    assert Rod(length).length == length
