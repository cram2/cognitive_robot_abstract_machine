"""Tests for import generation in ``krrood.code_generation.imports``."""

from __future__ import annotations

from typing import Callable

import pytest

from krrood.code_generation.exceptions import FunctionMissingAnnotationsError
from krrood.code_generation.imports import (
    ImportData,
    generate_import_statement_for_callable,
    get_imports_from_types,
)
from krrood.exceptions import DataclassException


def _make_module_level_distance() -> Callable:
    """Return a module-level ``distance`` function living in ``my.module``."""

    def distance(x: float, y: float) -> float:
        """Compute Euclidean distance between x and y."""
        return abs(x - y)

    distance.__module__ = "my.module"
    return distance


def _make_method_host():
    """Return a class whose ``distance`` method exercises the method path."""

    class MyClass:
        def distance(self, x: float, y: float) -> float:
            return abs(x - y)

    MyClass.__module__ = "my.module"
    MyClass.distance.__module__ = "my.module"
    return MyClass


@pytest.fixture(scope="module")
def module_level_distance() -> Callable:
    """A plain module-level function with annotations."""
    return _make_module_level_distance()


@pytest.fixture(scope="module")
def method_host_class():
    """A class whose ``distance`` method exercises the method path."""
    return _make_method_host()


class TestGenerateCallableImportModuleLevel:
    """Guarantees for ``generate_import_statement_for_callable`` when passed a plain function."""

    def test_returns_import_data(self, module_level_distance: Callable) -> None:
        """Result is an :class:`ImportData` bundling the import line and access."""
        result = generate_import_statement_for_callable(module_level_distance)
        assert isinstance(result, ImportData)
        assert isinstance(result.import_line, str)
        assert isinstance(result.access_expression, str)

    def test_import_line_is_from_import(self, module_level_distance: Callable) -> None:
        """The import line starts with ``from``."""
        import_line = generate_import_statement_for_callable(module_level_distance).import_line
        assert import_line.startswith("from ")

    def test_import_line_contains_module(self, module_level_distance: Callable) -> None:
        """The import line references the function's module."""
        import_line = generate_import_statement_for_callable(module_level_distance).import_line
        assert "my.module" in import_line

    def test_import_line_contains_function_name(
        self, module_level_distance: Callable
    ) -> None:
        """The import line names the function directly (not a wrapping class)."""
        import_line = generate_import_statement_for_callable(module_level_distance).import_line
        assert "distance" in import_line

    def test_access_expression_is_bare_name(
        self, module_level_distance: Callable
    ) -> None:
        """For a module-level function the access expression is the bare name."""
        access = generate_import_statement_for_callable(module_level_distance).access_expression
        assert access == "distance"


class TestGenerateCallableImportMethod:
    """Guarantees for ``generate_import_statement_for_callable`` when passed an unbound method."""

    def test_import_line_imports_class_not_function(self, method_host_class) -> None:
        """The import line must import ``MyClass``, not ``distance`` directly."""
        import_line = generate_import_statement_for_callable(method_host_class.distance).import_line
        assert "MyClass" in import_line
        assert import_line.split()[-1] == "MyClass"

    def test_access_expression_is_dotted(self, method_host_class) -> None:
        """The access expression for a method is ``ClassName.method_name``."""
        access = generate_import_statement_for_callable(method_host_class.distance).access_expression
        assert "." in access

    def test_access_expression_starts_with_class_name(self, method_host_class) -> None:
        """The access expression starts with the class name."""
        access = generate_import_statement_for_callable(method_host_class.distance).access_expression
        assert access.startswith("MyClass")

    def test_access_expression_ends_with_method_name(self, method_host_class) -> None:
        """The access expression ends with the method name."""
        access = generate_import_statement_for_callable(method_host_class.distance).access_expression
        assert access.endswith("distance")

    def test_import_line_references_module(self, method_host_class) -> None:
        """The import line still references the correct module."""
        import_line = generate_import_statement_for_callable(method_host_class.distance).import_line
        assert "my.module" in import_line


class TestFunctionMissingAnnotationsError:
    """Guarantees for the code-generation missing-annotation exception."""

    def test_is_subclass_of_dataclass_exception(self) -> None:
        """``FunctionMissingAnnotationsError`` is a ``DataclassException`` subclass."""
        assert issubclass(FunctionMissingAnnotationsError, DataclassException)

    def test_can_be_raised_and_caught_as_dataclass_exception(self) -> None:
        """An instance can be raised and caught as ``DataclassException``."""
        with pytest.raises(DataclassException):
            raise FunctionMissingAnnotationsError("missing annotation")

    def test_can_be_caught_by_its_own_type(self) -> None:
        """An instance is catchable by its own class."""
        with pytest.raises(FunctionMissingAnnotationsError):
            raise FunctionMissingAnnotationsError("missing annotation")


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
