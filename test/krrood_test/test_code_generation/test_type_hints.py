"""Tests for type-hint serialisation in ``krrood.code_generation.type_hints``."""

from __future__ import annotations

import enum
from typing import List, Optional, Union

from krrood.code_generation.type_hints import stringify_type_hint, value_to_source


class TestStringifyTypeHint:
    """Tests for :func:`stringify_type_hint`."""

    def test_builtin_types(self):
        assert stringify_type_hint(int) == "int"
        assert stringify_type_hint(str) == "str"
        assert stringify_type_hint(float) == "float"
        assert stringify_type_hint(bool) == "bool"

    def test_string_input(self):
        assert stringify_type_hint("MyType") == "MyType"

    def test_generic_alias(self):
        assert stringify_type_hint(List[int]) == "List[int]"
        # Optional[str] is Union[str, None] internally
        result = stringify_type_hint(Optional[str])
        assert "Union" in result
        assert "str" in result
        assert stringify_type_hint(Union[int, str]) == "Union[int, str]"

    def test_nested_generic(self):
        result = stringify_type_hint(List[Optional[int]])
        assert "List" in result
        assert "Union" in result  # Optional[int] expands to Union[int, None]

    def test_custom_type(self):
        class MyCustomType:
            pass

        result = stringify_type_hint(MyCustomType)
        # Should use the qualified name for non-builtin types
        assert "MyCustomType" in result


class TestValueToSource:
    """Tests for :func:`value_to_source`."""

    def test_none(self):
        assert value_to_source(None) == "None"

    def test_bool(self):
        assert value_to_source(True) == "True"
        assert value_to_source(False) == "False"

    def test_int_float(self):
        assert value_to_source(42) == "42"
        assert value_to_source(3.14) == "3.14"

    def test_string(self):
        assert value_to_source("hello") == "'hello'"

    def test_enum(self):
        class Color(enum.Enum):
            RED = 1
            BLUE = 2

        assert value_to_source(Color.RED) == "Color.RED"

    def test_type_object(self):
        assert value_to_source(int) == "int"
        assert value_to_source(str) == "str"

    def test_fallback(self):
        """Fallback to repr for unrecognised types."""
        result = value_to_source(complex(1, 2))
        assert "1+2j" in result or "(1+2j)" in result
