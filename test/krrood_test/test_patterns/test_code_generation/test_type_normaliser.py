import sys
from dataclasses import dataclass
from typing import List, Optional, TypeVar
from unittest.mock import MagicMock

import pytest

from krrood.class_diagrams import ClassDiagram
from krrood.patterns.code_generation.type_normaliser import TypeNormaliser
from krrood.patterns.code_generation.import_name_resolver import ImportNameResolver


@dataclass
class _FakePerson:
    name: str


class _FakeModule:
    __name__ = "test_module"
    __dict__ = {}


@pytest.fixture
def normaliser():
    class_diagram = MagicMock(spec=ClassDiagram)
    class_diagram.wrapped_classes = []
    source_module = _FakeModule()
    resolver = ImportNameResolver(
        source_module=source_module,
        companion_modules=[],
        class_diagram=class_diagram,
    )
    return TypeNormaliser(resolver=resolver, class_diagram=class_diagram)


def test_normalise_plain_class(normaliser):
    result = normaliser.normalise(_FakePerson)
    assert result == "_FakePerson"


def test_normalise_none_type(normaliser):
    result = normaliser.normalise(type(None))
    assert result == "None"


def test_normalise_type_var(normaliser):
    T = TypeVar("T")
    result = normaliser.normalise(T)
    assert result == "T"


def test_normalise_generic_list(normaliser):
    result = normaliser.normalise(List[str])
    assert "list" in result.lower()
    assert "str" in result


def test_normalise_optional(normaliser):
    result = normaliser.normalise(Optional[str])
    assert "str" in result


def test_normalise_string_forward_ref(normaliser):
    result = normaliser.normalise("MyClass")
    assert result == "MyClass"


def test_class_name_getter_used(normaliser):
    normaliser.class_name_getter = lambda clazz: f"Custom_{clazz.__name__}"
    result = normaliser.normalise(_FakePerson)
    assert result == "Custom__FakePerson"
