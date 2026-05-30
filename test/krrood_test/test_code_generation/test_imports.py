"""Tests for import-generation utilities in ``krrood.code_generation.utils``."""

from __future__ import annotations

from typing import List, Optional

import pytest

from krrood.code_generation.utils import get_imports_from_types


class TestGetImportsFromTypes:
    """Tests for :func:`get_imports_from_types`."""

    def test_builtin_types_excluded(self):
        """Builtin types (int, str) should not generate import lines."""
        result = get_imports_from_types([int, str, float, bool])
        assert result == [] or all("builtins" not in line for line in result)

    def test_typing_types_grouped(self):
        """Types from the same module should share one import line."""
        import typing_extensions

        result = get_imports_from_types(
            [typing_extensions.List, typing_extensions.Optional]
        )
        typing_lines = [l for l in result if "typing_extensions" in l or "typing" in l]
        if typing_lines:
            line = typing_lines[0]
            assert "List" in line
            assert "Optional" in line

    def test_custom_types(self):
        """Custom types should produce import lines with their module."""

        class MyClass:
            __module__ = "my_package.my_module"
            __qualname__ = "MyClass"

        result = get_imports_from_types([MyClass])
        assert any("my_package.my_module" in line for line in result)
        assert any("MyClass" in line for line in result)

    def test_empty_input(self):
        """Empty input should produce no import lines."""
        assert get_imports_from_types([]) == []

    def test_excluded_modules(self):
        """Excluded modules should not appear."""

        class A:
            __module__ = "keep_me"
            __qualname__ = "A"

        class B:
            __module__ = "exclude_me"
            __qualname__ = "B"

        result = get_imports_from_types([A, B], excluded_modules=["exclude_me"])
        assert any("keep_me" in line for line in result)
        assert not any("exclude_me" in line for line in result)
