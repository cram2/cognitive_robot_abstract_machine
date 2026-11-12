"""
Tests of the check finding code without a docstring.
"""

from __future__ import annotations

from krrood.development_tools.agents.docstring_presence import (
    MissingDocstringChecker,
)

from .dataset_files import DocstringSnippet
from .dataset_lines import (
    ModuleLine,
    UnimportableDefinitionsLine,
    UndocumentedDefinitionsLine,
)

# %% undocumented code


def undocumented_first_lines(snippet: DocstringSnippet) -> list[int]:
    """
    :return: The first line of each undocumented piece of code in *snippet*, top to
        bottom.
    """
    return sorted(
        violation.first_line
        for violation in MissingDocstringChecker().violations(path=snippet)
    )


def test_an_undocumented_field_and_methods_are_reported_and_an_abstract_implementation_is_not() -> (
    None
):
    assert undocumented_first_lines(DocstringSnippet.UNDOCUMENTED_DEFINITIONS) == [
        UndocumentedDefinitionsLine.FIELD,
        UndocumentedDefinitionsLine.OVERRIDING_METHOD,
        UndocumentedDefinitionsLine.METHOD,
    ]


def test_an_unimportable_module_is_checked_except_for_its_methods() -> None:
    assert undocumented_first_lines(DocstringSnippet.UNIMPORTABLE_DEFINITIONS) == [
        UnimportableDefinitionsLine.FUNCTION
    ]


def test_a_module_without_a_docstring_is_reported() -> None:
    assert undocumented_first_lines(DocstringSnippet.NO_MODULE_DOCSTRING) == [
        ModuleLine.FIRST
    ]


def test_a_fully_documented_module_has_no_violation() -> None:
    assert undocumented_first_lines(DocstringSnippet.DOCUMENTED_MODULE) == []
