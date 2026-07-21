"""
Tests for source extraction in ``krrood.code_generation.source_extraction_utils``.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect

import pytest

from krrood.code_generation.source_extraction_utils import (
    LineSpan,
    extract_class_source,
    extract_function_source,
    extract_imports_from,
)
from krrood.exceptions import (
    ModuleNotFoundForConvertingImportsToAbsolute,
    NoSourceDataToParseImportsFrom,
    SourceDataNotProvided,
)

from test.krrood_test.dataset import department_and_employee, example_classes
from test.krrood_test.dataset.department_and_employee import Department
from test.krrood_test.dataset.example_classes import module_level_function


class TestExtractFunctionSource:
    """
    Guarantees for :func:`extract_function_source`.
    """

    def test_extracts_named_function(self):
        result = extract_function_source(
            [module_level_function.__name__], file_path=example_classes.__file__
        )
        assert [definition.name for definition in result.definitions] == [
            module_level_function.__name__
        ]
        assert f"def {module_level_function.__name__}():" in result.source_of(
            module_level_function.__name__
        )

    def test_extracts_all_when_names_empty(self):
        result = extract_function_source([], file_path=example_classes.__file__)
        assert [definition.name for definition in result.definitions] == [
            module_level_function.__name__
        ]

    def test_does_not_extract_classes(self):
        result = extract_function_source(
            [Department.__name__], file_path=department_and_employee.__file__
        )
        assert result.definitions == []

    def test_line_span_is_reported(self):
        result = extract_function_source(
            [module_level_function.__name__], file_path=example_classes.__file__
        )
        line_span = result.definitions[0].line_span
        assert isinstance(line_span, LineSpan)
        assert line_span.start_line <= line_span.end_line

    def test_exclude_signature(self):
        result = extract_function_source(
            [module_level_function.__name__],
            file_path=example_classes.__file__,
            include_signature=False,
        )
        assert f"def {module_level_function.__name__}" not in result.source_of(
            module_level_function.__name__
        )
        assert "return 1" in result.source_of(module_level_function.__name__)

    def test_lines_are_reported_individually(self):
        result = extract_function_source(
            [module_level_function.__name__], file_path=example_classes.__file__
        )
        assert result.definitions[0].lines == [
            f"def {module_level_function.__name__}():",
            "    return 1",
        ]

    def test_extracts_from_source_string(self):
        result = extract_function_source(
            [module_level_function.__name__],
            source=inspect.getsource(example_classes),
        )
        assert f"def {module_level_function.__name__}():" in result.source_of(
            module_level_function.__name__
        )

    def test_missing_source_raises(self):
        with pytest.raises(SourceDataNotProvided):
            extract_function_source([module_level_function.__name__])


class TestExtractClassSource:
    """
    Guarantees for :func:`extract_class_source`.
    """

    def test_extracts_named_class(self):
        result = extract_class_source(
            [Department.__name__], file_path=department_and_employee.__file__
        )
        assert [definition.name for definition in result.definitions] == [
            Department.__name__
        ]
        assert f"class {Department.__name__}(Symbol):" in result.source_of(
            Department.__name__
        )

    def test_does_not_extract_functions(self):
        result = extract_class_source(
            [module_level_function.__name__], file_path=example_classes.__file__
        )
        assert result.definitions == []


class TestExtractImportsFromSource:
    """
    Guarantees for :func:`extract_imports_from` given a source string.
    """

    def test_missing_source_raises(self):
        with pytest.raises(NoSourceDataToParseImportsFrom):
            extract_imports_from()

    def test_extracts_plain_import(self):
        imports = extract_imports_from(source="import os\n")
        assert imports == ["import os"]

    def test_extracts_plain_import_with_alias(self):
        imports = extract_imports_from(source="import os as o\n")
        assert imports == ["import os as o"]

    def test_extracts_from_import(self):
        imports = extract_imports_from(source="from a import b\n")
        assert imports == ["from a import b"]

    def test_extracts_from_import_with_alias(self):
        imports = extract_imports_from(source="from a import b as c\n")
        assert imports == ["from a import b as c"]

    def test_merges_multiple_names_from_the_same_module(self):
        imports = extract_imports_from(source="from a import b\nfrom a import c\n")
        assert imports == ["from a import b, c"]

    def test_result_is_sorted(self):
        imports = extract_imports_from(source="import sys\nimport os\n")
        assert imports == sorted(imports)

    def test_excludes_libraries_from_plain_import(self):
        imports = extract_imports_from(
            source="import os\nimport sys\n", exclude_libraries=["os"]
        )
        assert imports == ["import sys"]

    def test_excludes_libraries_from_from_import(self):
        imports = extract_imports_from(
            source="from os import path\n", exclude_libraries=["os"]
        )
        assert imports == []

    def test_reads_from_file(self, tmp_path):
        file_path = tmp_path / "module.py"
        file_path.write_text("import os\n")
        imports = extract_imports_from(file_path=str(file_path))
        assert imports == ["import os"]

    def test_reads_from_module(self, tmp_path):
        file_path = tmp_path / "sample_module_for_import_extraction.py"
        file_path.write_text("import os\n")
        module = _import_module_from_file(file_path)
        imports = extract_imports_from(module=module)
        assert imports == ["import os"]

    def test_reads_from_ast_tree(self):
        imports = extract_imports_from(ast_tree=ast.parse("import os\n"))
        assert imports == ["import os"]

    def test_convert_relative_to_absolute_without_module_or_file_raises(self):
        with pytest.raises(ModuleNotFoundForConvertingImportsToAbsolute):
            extract_imports_from(
                source="from . import sibling\n", convert_relative_to_absolute=True
            )

    def test_convert_relative_to_absolute_resolves_via_file_path(self, tmp_path):
        file_path = tmp_path / "sample_module_for_relative_import.py"
        file_path.write_text("from . import sibling\n")
        imports = extract_imports_from(
            file_path=str(file_path), convert_relative_to_absolute=True
        )
        assert imports == ["from sample_module_for_relative_import import sibling"]


def _import_module_from_file(file_path):
    """
    Import *file_path* as a module so :func:`inspect.getsource` can read it back.
    """
    module_name = file_path.stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
