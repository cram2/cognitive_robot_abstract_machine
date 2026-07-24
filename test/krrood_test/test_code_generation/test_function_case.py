"""Tests for ``FunctionCase`` source generation in ``krrood.code_generation``."""

from __future__ import annotations

from typing import Callable

import pytest

from krrood.code_generation.exceptions import FunctionMissingAnnotationsError
from krrood.code_generation.function_case import FunctionCaseGenerator


def _make_module_level_distance() -> Callable:
    """Return a fully annotated module-level ``distance`` function."""

    def distance(x: float, y: float) -> float:
        """Compute Euclidean distance between x and y."""
        return abs(x - y)

    distance.__module__ = "my.module"
    return distance


@pytest.fixture(scope="module")
def module_level_distance() -> Callable:
    return _make_module_level_distance()


@pytest.fixture(scope="module")
def distance_source(module_level_distance: Callable) -> str:
    """The generated source for the ``distance`` function."""
    return FunctionCaseGenerator().generate(module_level_distance)


class TestValidateAnnotationsMissingParameter:
    """A missing parameter annotation aborts generation."""

    def test_raises_on_unannotated_positional_parameter(self) -> None:
        def bad(x, y: float) -> float:
            return x + y

        with pytest.raises(FunctionMissingAnnotationsError):
            FunctionCaseGenerator().generate(bad)

    def test_raises_when_all_parameters_lack_annotations(self) -> None:
        def bare(a, b):
            pass

        with pytest.raises(FunctionMissingAnnotationsError):
            FunctionCaseGenerator().generate(bare)


class TestValidateAnnotationsMissingReturn:
    """A missing return annotation aborts generation."""

    def test_raises_on_missing_return_annotation(self) -> None:
        def no_return(x: float, y: float):
            return x + y

        with pytest.raises(FunctionMissingAnnotationsError):
            FunctionCaseGenerator().generate(no_return)


class TestValidateAnnotationsSelfExcluded:
    """An unannotated ``self`` parameter is tolerated and never emitted."""

    def test_self_without_annotation_does_not_raise(self) -> None:
        class MyClass:
            def method(self, x: float) -> float:
                return x

        source = FunctionCaseGenerator().generate(MyClass.method)
        assert isinstance(source, str)

    def test_self_does_not_appear_as_dataclass_field(self) -> None:
        class MyClass:
            def method(self, x: float) -> float:
                return x

        source = FunctionCaseGenerator().generate(MyClass.method)
        field_lines = [
            line.strip()
            for line in source.splitlines()
            if ":" in line and not line.strip().startswith("#")
        ]
        assert [line for line in field_lines if line.startswith("self")] == []


class TestValidateAnnotationsClsExcluded:
    """An unannotated ``cls`` parameter is tolerated and never emitted."""

    def test_cls_without_annotation_does_not_raise(self) -> None:
        class MyClass:
            @classmethod
            def create(cls, value: float) -> float:
                return value

        source = FunctionCaseGenerator().generate(MyClass.create.__func__)
        assert isinstance(source, str)

    def test_cls_does_not_appear_as_dataclass_field(self) -> None:
        class MyClass:
            @classmethod
            def create(cls, value: float) -> float:
                return value

        source = FunctionCaseGenerator().generate(MyClass.create.__func__)
        field_lines = [
            line.strip()
            for line in source.splitlines()
            if ":" in line and not line.strip().startswith("#")
        ]
        assert [line for line in field_lines if line.startswith("cls")] == []


class TestGeneratedSourceBoilerplate:
    """The emitted source contains all mandatory boilerplate lines."""

    def test_contains_future_annotations_import(self, distance_source: str) -> None:
        assert "from __future__ import annotations" in distance_source

    def test_contains_dataclass_import(self, distance_source: str) -> None:
        assert "from dataclasses import dataclass" in distance_source

    def test_contains_classvar_import(self, distance_source: str) -> None:
        assert "ClassVar" in distance_source

    def test_contains_callable_import_keyword(self, distance_source: str) -> None:
        assert "Callable" in distance_source

    def test_contains_dataclass_decorator(self, distance_source: str) -> None:
        assert "@dataclass" in distance_source

    def test_returns_str(self, distance_source: str) -> None:
        assert isinstance(distance_source, str)

    def test_source_is_nonempty(self, distance_source: str) -> None:
        assert len(distance_source.strip()) > 0


class TestGeneratedSourceCallableImport:
    """The emitted source imports the callable being wrapped."""

    def test_imports_distance_function(self, distance_source: str) -> None:
        assert "import distance" in distance_source

    def test_base_class_import_present(self, distance_source: str) -> None:
        assert "FunctionCase" in distance_source


class TestGeneratedSourceClassDeclaration:
    """The emitted source declares the expected class."""

    def test_class_name_is_camel_cased_function_name(
        self, distance_source: str
    ) -> None:
        assert "class Distance" in distance_source

    def test_class_inherits_from_base(self, distance_source: str) -> None:
        assert "class Distance(FunctionCase)" in distance_source or (
            "class Distance(" in distance_source and "FunctionCase" in distance_source
        )


class TestGeneratedSourceClassVar:
    """The emitted source contains the ``function`` ClassVar assignment."""

    def test_function_classvar_present(self, distance_source: str) -> None:
        assert "function" in distance_source
        assert "ClassVar" in distance_source

    def test_function_classvar_assigned_to_distance(self, distance_source: str) -> None:
        assert "= distance" in distance_source or "=distance" in distance_source


class TestGeneratedSourceFields:
    """The emitted source declares one field per non-self/cls parameter."""

    def test_x_field_present(self, distance_source: str) -> None:
        assert "x: float" in distance_source

    def test_y_field_present(self, distance_source: str) -> None:
        assert "y: float" in distance_source


class TestGeneratedSourceOutputField:
    """The emitted source contains a field for the return value."""

    def test_output_field_present(self, distance_source: str) -> None:
        assert "_output" in distance_source

    def test_output_field_has_return_type(self, distance_source: str) -> None:
        assert "_output: float" in distance_source
