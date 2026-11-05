"""
Tests of the checks finding dataclass fields and method parameters that can be passed
positionally.
"""

from __future__ import annotations

from krrood.development_tools.agents.checker import CodeRuleChecker
from krrood.development_tools.agents.positional_parameters import (
    PositionalFieldsChecker,
    PositionalParametersChecker,
)

from .dataset_files import PositionalParametersSnippet
from .dataset_lines import (
    PositionalDefinitionsLine,
    UnimportablePositionalDefinitionsLine,
)

# %% reported lines


def reported_first_lines(
    checker: CodeRuleChecker, snippet: PositionalParametersSnippet
) -> list[int]:
    """
    :return: The first line of each violation *checker* finds in *snippet*, top to
        bottom.
    """
    return sorted(
        violation.first_line for violation in checker.violations(path=snippet)
    )


# %% dataclasses


def test_dataclasses_without_keyword_only_fields_are_reported_and_an_enum_mixin_is_not() -> (
    None
):
    """
    Dataclasses declared without ``kw_only=True`` are reported, whether imported or
    reached through :mod:`dataclasses`, and frozen or not.
    """
    assert reported_first_lines(
        PositionalFieldsChecker(), PositionalParametersSnippet.POSITIONAL_DEFINITIONS
    ) == [
        PositionalDefinitionsLine.POSITIONAL_DATACLASS,
        PositionalDefinitionsLine.POSITIONAL_FROZEN_DATACLASS,
    ]


def test_a_dataclass_of_an_unimportable_module_is_reported() -> None:
    """
    A dataclass is reported even when its module cannot be imported to rule out an enum
    mixing it in.
    """
    assert reported_first_lines(
        PositionalFieldsChecker(), PositionalParametersSnippet.UNIMPORTABLE_DEFINITIONS
    ) == [UnimportablePositionalDefinitionsLine.POSITIONAL_DATACLASS]


# %% methods


def test_positional_methods_are_reported_and_exempt_methods_are_not() -> None:
    """
    Positional methods and static methods are reported, overrides of the repository's
    own methods too; dunder methods, setters, overrides of a standard library method
    and methods marked as called positionally are not.
    """
    assert reported_first_lines(
        PositionalParametersChecker(),
        PositionalParametersSnippet.POSITIONAL_DEFINITIONS,
    ) == [
        PositionalDefinitionsLine.POSITIONAL_METHOD,
        PositionalDefinitionsLine.POSITIONAL_STATIC_METHOD,
        PositionalDefinitionsLine.POSITIONAL_OVERRIDE,
        PositionalDefinitionsLine.POSITIONAL_METHOD_OF_LIBRARY_SUBCLASS,
    ]


def test_in_an_unimportable_module_only_methods_of_classes_without_bases_are_reported() -> (
    None
):
    """
    When a module cannot be imported, a method of a class with bases might override an
    installed one and is left out.
    """
    assert reported_first_lines(
        PositionalParametersChecker(),
        PositionalParametersSnippet.UNIMPORTABLE_DEFINITIONS,
    ) == [UnimportablePositionalDefinitionsLine.POSITIONAL_METHOD_WITHOUT_BASES]
