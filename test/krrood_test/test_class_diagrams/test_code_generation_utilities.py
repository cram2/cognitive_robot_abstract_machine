"""
Phase 1 tests: code-generation helpers in
``krrood.code_generation``.

Each test class / function verifies exactly one behavioural guarantee of a
public symbol.  All tests are expected to fail at import time
(``ModuleNotFoundError``) until the implementation module is created.

Testing plan mapping
--------------------
Step 1 — ``to_camel_case``:
    ``TestToCamelCase`` – all parametrized conversion cases.

Step 2 — ``generate_callable_import`` for module-level functions:
    ``TestGenerateCallableImportModuleLevel`` – import line is a ``from … import``
    containing the function name; access expression equals the bare function name.

Step 3 — ``generate_callable_import`` for class methods:
    ``TestGenerateCallableImportMethod`` – import line imports the *class*, not the
    function directly; access expression is ``ClassName.method``.

Step 4 — ``FunctionMissingAnnotationsError``:
    ``TestFunctionMissingAnnotationsError`` – is a ``TypeError`` subclass.

Step 5 — annotation validation (missing param / missing return):
    ``TestValidateAnnotationsMissingParam``
    ``TestValidateAnnotationsMissingReturn``

Step 6 — annotation validation (``self`` / ``cls`` are silently excluded):
    ``TestValidateAnnotationsSelfExcluded``
    ``TestValidateAnnotationsClsExcluded``

Step 7 — ``function_to_dataclass_source`` structural guarantees:
    ``TestFunctionToDataclassSourceKeywords`` – required boilerplate is present.
    ``TestFunctionToDataclassSourceCallableImport`` – function import appears.
    ``TestFunctionToDataclassSourceClassDeclaration`` – class name and base present.
    ``TestFunctionToDataclassSourceClassVar`` – ClassVar assignment present.
    ``TestFunctionToDataclassSourceFields`` – parameter fields present.
    ``TestFunctionToDataclassSourceOutputField`` – output field present.
"""

from __future__ import annotations

import dataclasses
import inspect
import sys
import textwrap
import types
from typing import Callable, Tuple

import pytest

# ---------------------------------------------------------------------------
# The module under test.  The import is intentionally at module scope so that
# all tests in this file fail fast with ``ModuleNotFoundError`` when the
# implementation does not yet exist — rather than producing confusing per-test
# AttributeErrors at call time.
# ---------------------------------------------------------------------------
from krrood.code_generation import (
    FunctionMissingAnnotationsError,
    function_to_dataclass_source,
    generate_callable_import,
    to_camel_case,
)

# ===========================================================================
# Shared helpers / fixtures
# ===========================================================================


def _make_module_level_distance() -> Callable:
    """
    Return a module-level function called ``distance`` whose ``__module__``
    is set to the synthetic name ``my.module``.

    The function has full annotations so it passes validation inside
    ``function_to_dataclass_source``.
    """

    def distance(x: float, y: float) -> float:
        """Compute Euclidean distance between x and y."""
        return abs(x - y)

    # Simulate the function living in a real, importable module.
    distance.__module__ = "my.module"
    return distance


def _make_method_host():
    """
    Return a class ``MyClass`` that carries a method ``distance`` whose
    ``__qualname__`` is ``MyClass.distance`` and whose ``__module__`` is
    ``my.module``.

    The returned value is the *class itself*, not an instance, so the test can
    access both the class and its method.
    """

    class MyClass:
        def distance(self, x: float, y: float) -> float:
            return abs(x - y)

    MyClass.__module__ = "my.module"
    MyClass.distance.__module__ = "my.module"
    # Rebind qualname to match what Python already sets for methods.
    # Python sets __qualname__ = "MyClass.distance" automatically, so this is
    # just here for documentary clarity.
    return MyClass


@pytest.fixture(scope="module")
def module_level_distance() -> Callable:
    """Module-scoped fixture: a plain module-level function with annotations."""
    return _make_module_level_distance()


@pytest.fixture(scope="module")
def method_host_class():
    """Module-scoped fixture: a class whose ``distance`` method exercises the method path."""
    return _make_method_host()


# ===========================================================================
# Step 1 — to_camel_case
# ===========================================================================


@pytest.mark.parametrize(
    "name, expected",
    [
        ("distance", "Distance"),
        ("my_func", "MyFunc"),
        ("my_distance_rdr", "MyDistanceRdr"),
        ("already_camel", "AlreadyCamel"),
        ("x", "X"),
        ("a_b_c", "ABC"),
    ],
    ids=[
        "single-word",
        "two-part",
        "three-part",
        "four-part",
        "single-char",
        "all-single-chars",
    ],
)
def test_to_camel_case_converts_snake_to_camel(name: str, expected: str) -> None:
    """``to_camel_case`` must title-case every underscore-separated segment."""
    result = to_camel_case(name)
    assert result == expected


def test_to_camel_case_returns_str(module_level_distance: Callable) -> None:
    """``to_camel_case`` always returns a ``str``."""
    assert isinstance(to_camel_case("distance"), str)


def test_to_camel_case_result_is_nonempty_for_nonempty_input() -> None:
    """``to_camel_case`` must not return an empty string for a non-empty input."""
    assert to_camel_case("x") != ""


# ===========================================================================
# Step 2 — generate_callable_import: module-level function
# ===========================================================================


class TestGenerateCallableImportModuleLevel:
    """Guarantees for ``generate_callable_import`` when passed a plain function."""

    def test_returns_two_tuple(self, module_level_distance: Callable) -> None:
        """Result is a two-element tuple."""
        result = generate_callable_import(module_level_distance)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_import_line_is_from_import(self, module_level_distance: Callable) -> None:
        """The import line starts with ``from``."""
        import_line, _ = generate_callable_import(module_level_distance)
        assert import_line.startswith("from ")

    def test_import_line_contains_module(self, module_level_distance: Callable) -> None:
        """The import line references the function's module."""
        import_line, _ = generate_callable_import(module_level_distance)
        assert "my.module" in import_line

    def test_import_line_contains_function_name(
        self, module_level_distance: Callable
    ) -> None:
        """The import line names the function directly (not a class wrapping it)."""
        import_line, _ = generate_callable_import(module_level_distance)
        assert "distance" in import_line

    def test_access_expression_is_bare_name(
        self, module_level_distance: Callable
    ) -> None:
        """For a module-level function the access expression is the bare function name."""
        _, access_expr = generate_callable_import(module_level_distance)
        assert access_expr == "distance"

    def test_import_line_is_str(self, module_level_distance: Callable) -> None:
        """The import line is a ``str``."""
        import_line, _ = generate_callable_import(module_level_distance)
        assert isinstance(import_line, str)

    def test_access_expression_is_str(self, module_level_distance: Callable) -> None:
        """The access expression is a ``str``."""
        _, access_expr = generate_callable_import(module_level_distance)
        assert isinstance(access_expr, str)


# ===========================================================================
# Step 3 — generate_callable_import: class method
# ===========================================================================


class TestGenerateCallableImportMethod:
    """Guarantees for ``generate_callable_import`` when passed an unbound method."""

    def test_import_line_imports_class_not_function(self, method_host_class) -> None:
        """The import line must import ``MyClass``, not ``distance`` directly.

        When a function's ``__qualname__`` contains a dot (``MyClass.distance``),
        the import must be ``from my.module import MyClass`` so the class is
        available to form the access expression.
        """
        method = method_host_class.distance
        import_line, _ = generate_callable_import(method)
        assert "MyClass" in import_line
        # The bare function name must NOT be the imported symbol.
        # (The import is "from my.module import MyClass", not "import distance")
        parts = import_line.split()
        # parts: ["from", "my.module", "import", "MyClass"]
        imported_symbol = parts[-1]
        assert imported_symbol == "MyClass"

    def test_access_expression_is_dotted(self, method_host_class) -> None:
        """The access expression for a method is ``ClassName.method_name``."""
        method = method_host_class.distance
        _, access_expr = generate_callable_import(method)
        assert "." in access_expr

    def test_access_expression_starts_with_class_name(self, method_host_class) -> None:
        """The access expression starts with the class name."""
        method = method_host_class.distance
        _, access_expr = generate_callable_import(method)
        assert access_expr.startswith("MyClass")

    def test_access_expression_ends_with_method_name(self, method_host_class) -> None:
        """The access expression ends with the method name."""
        method = method_host_class.distance
        _, access_expr = generate_callable_import(method)
        assert access_expr.endswith("distance")

    def test_import_line_references_module(self, method_host_class) -> None:
        """The import line still references the correct module."""
        method = method_host_class.distance
        import_line, _ = generate_callable_import(method)
        assert "my.module" in import_line


# ===========================================================================
# Step 4 — FunctionMissingAnnotationsError
# ===========================================================================


class TestFunctionMissingAnnotationsError:
    def test_is_subclass_of_type_error(self) -> None:
        """``FunctionMissingAnnotationsError`` must be a ``TypeError`` subclass."""
        assert issubclass(FunctionMissingAnnotationsError, TypeError)

    def test_can_be_raised_and_caught_as_type_error(self) -> None:
        """An instance can be raised and caught as ``TypeError``."""
        with pytest.raises(TypeError):
            raise FunctionMissingAnnotationsError("missing annotation")

    def test_can_be_caught_by_its_own_type(self) -> None:
        """An instance is catchable by its own class."""
        with pytest.raises(FunctionMissingAnnotationsError):
            raise FunctionMissingAnnotationsError("missing annotation")


# ===========================================================================
# Steps 5 & 6 — annotation validation
# ===========================================================================


class TestValidateAnnotationsMissingParam:
    def test_raises_on_unannotated_positional_param(self) -> None:
        """A function with an unannotated positional parameter raises ``FunctionMissingAnnotationsError``."""

        def bad(x, y: float) -> float:
            return x + y

        with pytest.raises(FunctionMissingAnnotationsError):
            function_to_dataclass_source(bad)

    def test_raises_when_all_params_lack_annotations(self) -> None:
        """A function with no annotations at all raises ``FunctionMissingAnnotationsError``."""

        def bare(a, b):
            pass

        with pytest.raises(FunctionMissingAnnotationsError):
            function_to_dataclass_source(bare)


class TestValidateAnnotationsMissingReturn:
    def test_raises_on_missing_return_annotation(self) -> None:
        """A function with annotated parameters but no return annotation raises ``FunctionMissingAnnotationsError``."""

        def no_return(x: float, y: float):
            return x + y

        with pytest.raises(FunctionMissingAnnotationsError):
            function_to_dataclass_source(no_return)


class TestValidateAnnotationsSelfExcluded:
    def test_self_without_annotation_does_not_raise(self) -> None:
        """A method with an unannotated ``self`` parameter must not raise."""

        class MyClass:
            def method(self, x: float) -> float:
                return x

        # Should succeed (no error), not raise FunctionMissingAnnotationsError.
        source = function_to_dataclass_source(MyClass.method)
        assert isinstance(source, str)

    def test_self_does_not_appear_as_dataclass_field(self) -> None:
        """The emitted source must not contain a ``self`` field declaration."""

        class MyClass:
            def method(self, x: float) -> float:
                return x

        source = function_to_dataclass_source(MyClass.method)
        # ``self: …`` as a field line must not appear in the dataclass body.
        # We check that "self" is not introduced as a typed field annotation.
        lines = source.splitlines()
        field_lines = [
            ln.strip() for ln in lines if ":" in ln and not ln.strip().startswith("#")
        ]
        # No field line should start with "self"
        self_fields = [fl for fl in field_lines if fl.startswith("self")]
        assert self_fields == []


class TestValidateAnnotationsClsExcluded:
    def test_cls_without_annotation_does_not_raise(self) -> None:
        """A classmethod with an unannotated ``cls`` parameter must not raise."""

        class MyClass:
            @classmethod
            def create(cls, value: float) -> float:
                return value

        # Access the underlying function (not the bound classmethod descriptor).
        source = function_to_dataclass_source(MyClass.create.__func__)
        assert isinstance(source, str)

    def test_cls_does_not_appear_as_dataclass_field(self) -> None:
        """The emitted source must not contain a ``cls`` field declaration."""

        class MyClass:
            @classmethod
            def create(cls, value: float) -> float:
                return value

        source = function_to_dataclass_source(MyClass.create.__func__)
        lines = source.splitlines()
        field_lines = [
            ln.strip() for ln in lines if ":" in ln and not ln.strip().startswith("#")
        ]
        cls_fields = [fl for fl in field_lines if fl.startswith("cls")]
        assert cls_fields == []


# ===========================================================================
# Step 7 — function_to_dataclass_source: structural/textual guarantees
# ===========================================================================


@pytest.fixture(scope="module")
def distance_source(module_level_distance: Callable) -> str:
    """The generated source for the ``distance`` function (module-scoped)."""
    return function_to_dataclass_source(module_level_distance)


class TestFunctionToDataclassSourceKeywords:
    """The emitted source contains all mandatory boilerplate lines."""

    def test_contains_future_annotations_import(self, distance_source: str) -> None:
        """``from __future__ import annotations`` must be present."""
        assert "from __future__ import annotations" in distance_source

    def test_contains_dataclass_import(self, distance_source: str) -> None:
        """``from dataclasses import dataclass`` must be present."""
        assert "from dataclasses import dataclass" in distance_source

    def test_contains_classvar_import(self, distance_source: str) -> None:
        """``ClassVar`` must be imported."""
        assert "ClassVar" in distance_source

    def test_contains_callable_import_keyword(self, distance_source: str) -> None:
        """``Callable`` must be imported."""
        assert "Callable" in distance_source

    def test_contains_dataclass_decorator(self, distance_source: str) -> None:
        """``@dataclass`` decorator must appear."""
        assert "@dataclass" in distance_source

    def test_returns_str(self, distance_source: str) -> None:
        """``function_to_dataclass_source`` returns a ``str``."""
        assert isinstance(distance_source, str)

    def test_source_is_nonempty(self, distance_source: str) -> None:
        """The emitted source must not be empty."""
        assert len(distance_source.strip()) > 0


class TestFunctionToDataclassSourceCallableImport:
    """The emitted source imports the callable being wrapped."""

    def test_imports_distance_function(self, distance_source: str) -> None:
        """A ``from … import distance`` line must appear in the source."""
        assert "import distance" in distance_source

    def test_base_class_import_present(self, distance_source: str) -> None:
        """The base class (``FunctionCase`` or custom) must be imported."""
        # The default base_class_fqn ends with ``FunctionCase``; check that
        # the base class name appears in an import statement.
        # We do not assert the exact module path (it may change), only that
        # the import is present.
        assert "FunctionCase" in distance_source


class TestFunctionToDataclassSourceClassDeclaration:
    """The emitted source declares the expected class."""

    def test_class_name_is_camel_cased_function_name(
        self, distance_source: str
    ) -> None:
        """The class is named ``Distance`` (CamelCase of ``distance``)."""
        assert "class Distance" in distance_source

    def test_class_inherits_from_base(self, distance_source: str) -> None:
        """The class declaration includes ``FunctionCase`` as the base."""
        assert "class Distance(FunctionCase)" in distance_source or (
            "class Distance(" in distance_source and "FunctionCase" in distance_source
        )


class TestFunctionToDataclassSourceClassVar:
    """The emitted source contains the ``function`` ClassVar assignment."""

    def test_function_classvar_present(self, distance_source: str) -> None:
        """``function: ClassVar[Callable]`` must appear in the class body."""
        assert "function" in distance_source
        assert "ClassVar" in distance_source

    def test_function_classvar_assigned_to_distance(self, distance_source: str) -> None:
        """The ``function`` ClassVar is assigned the wrapped function name."""
        assert "= distance" in distance_source or "=distance" in distance_source


class TestFunctionToDataclassSourceFields:
    """The emitted source declares one dataclass field per non-self/cls parameter."""

    def test_x_field_present(self, distance_source: str) -> None:
        """Field ``x: float`` must appear in the class body."""
        assert "x: float" in distance_source

    def test_y_field_present(self, distance_source: str) -> None:
        """Field ``y: float`` must appear in the class body."""
        assert "y: float" in distance_source


class TestFunctionToDataclassSourceOutputField:
    """The emitted source contains a field for the return value."""

    def test_output_field_present(self, distance_source: str) -> None:
        """An ``_output`` field annotated with the return type must appear."""
        assert "_output" in distance_source

    def test_output_field_has_return_type(self, distance_source: str) -> None:
        """The ``_output`` field is annotated with the function's return type (``float``)."""
        assert "_output: float" in distance_source

