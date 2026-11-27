from abc import abstractmethod
from dataclasses import dataclass

import pytest

from krrood.patterns.exceptions import (
    InterfaceInstantiationError,
    MissingInterfaceAttributesError,
)
from krrood.patterns.interface import Interface

# %% an interface and the classes implementing it


class Measurable(Interface):
    length: float


@dataclass(kw_only=True)
class Rod(Measurable):
    length: float


class Ruler(Measurable):
    @property
    def length(self) -> float:
        return 1.0


class MeasuringTool(Measurable):
    """
    An abstract implementation that leaves the length to its subclasses.
    """

    @abstractmethod
    def calibrate(self) -> None: ...


# %% instantiation


def test_an_interface_cannot_be_instantiated():
    with pytest.raises(InterfaceInstantiationError):
        Measurable()


def test_a_class_implementing_an_interface_can_be_instantiated():
    length = 2.0
    assert Rod(length=length).length == length


def test_a_property_provides_a_declared_attribute():
    assert Ruler().length == 1.0


# %% attributes an implementation leaves out


def test_a_class_leaving_out_a_declared_attribute_cannot_be_defined():
    with pytest.raises(MissingInterfaceAttributesError) as error:

        @dataclass(kw_only=True)
        class Unmeasured(Measurable):
            label: str = ""

    assert error.value.attributes == ("length",)


def test_an_abstract_class_may_leave_a_declared_attribute_to_its_subclasses():
    assert Interface.missing_attributes(class_=MeasuringTool) == ("length",)
