from __future__ import annotations

from dataclasses import dataclass

import pytest
from typing_extensions import Self

from krrood.class_diagrams.exceptions import FactoryMethodDecoratorError
from krrood.class_diagrams.factory_method_registry import FactoryMethodRegistry
from krrood.class_diagrams.method_classifier import (
    factory_method,
    factory_method_names,
    is_factory_method,
)


@dataclass
class ClassWithMarkedFactory:
    """
    Mimics a class whose factory is declared with the explicit ``@factory_method``
    marker.
    """

    value: str = ""

    @factory_method
    @classmethod
    def make(cls) -> ClassWithMarkedFactory:
        """
        A factory recognised only through the ``@factory_method`` marker.
        """
        return cls(value="made")

    @classmethod
    def from_value(cls, value: str) -> Self:
        """
        A factory recognised through its ``-> Self`` return annotation, without a
        marker.
        """
        return cls(value=value)

    @classmethod
    def describe(cls) -> str:
        """
        An ordinary classmethod that is not a factory.
        """
        return cls.__name__


class SubclassInheritingFactory(ClassWithMarkedFactory):
    """
    Mimics a subclass that inherits a marked factory without redeclaring it.
    """


def test_descriptor_registers_marked_factory_at_definition():
    assert FactoryMethodRegistry().is_registered(ClassWithMarkedFactory, "make")


def test_registration_is_inherited_by_subclasses():
    assert FactoryMethodRegistry().is_registered(SubclassInheritingFactory, "make")


def test_unmarked_classmethod_is_not_registered():
    assert not FactoryMethodRegistry().is_registered(
        ClassWithMarkedFactory, "from_value"
    )


def test_registry_is_a_singleton():
    assert FactoryMethodRegistry() is FactoryMethodRegistry()


def test_is_factory_method_detects_marker_and_return_annotation():
    assert is_factory_method(ClassWithMarkedFactory, "make")
    assert is_factory_method(ClassWithMarkedFactory, "from_value")
    assert not is_factory_method(ClassWithMarkedFactory, "describe")


def test_factory_method_names_includes_marked_and_annotated():
    names = factory_method_names(ClassWithMarkedFactory)
    assert "make" in names
    assert "from_value" in names
    assert "describe" not in names


def test_marked_factory_stays_callable_on_class_and_instance():
    assert ClassWithMarkedFactory.make().value == "made"
    assert ClassWithMarkedFactory(value="seed").make().value == "made"


def test_marker_applied_inside_classmethod_is_rejected():
    with pytest.raises(FactoryMethodDecoratorError):

        class WrongOrder:
            @classmethod
            @factory_method
            def make(cls) -> WrongOrder:
                return cls()
