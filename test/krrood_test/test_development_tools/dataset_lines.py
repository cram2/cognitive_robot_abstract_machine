"""
Line numbers and line counts within the dataset sources the code rule checkers read.
"""

from __future__ import annotations

from enum import IntEnum

# %% where reported code starts


class ModuleLine(IntEnum):
    """
    Where a module, and so its docstring, starts.
    """

    FIRST = 1
    """
    The first line of any module.
    """


class UndocumentedDefinitionsLine(IntEnum):
    """
    Lines of :attr:`~test.krrood_test.test_development_tools.dataset_files.DocstringSnippet.UNDOCUMENTED_DEFINITIONS`.
    """

    FIELD = 38
    """
    The undocumented field.
    """

    OVERRIDING_METHOD = 43
    """
    The undocumented override of an implemented method.
    """

    METHOD = 46
    """
    The undocumented method that overrides nothing.
    """


class UnimportableDefinitionsLine(IntEnum):
    """
    Lines of :attr:`~test.krrood_test.test_development_tools.dataset_files.DocstringSnippet.UNIMPORTABLE_DEFINITIONS`.
    """

    FUNCTION = 19
    """
    The undocumented function.
    """


class PositionalDefinitionsLine(IntEnum):
    """
    Lines of :attr:`~test.krrood_test.test_development_tools.dataset_files.PositionalParametersSnippet.POSITIONAL_DEFINITIONS`.
    """

    POSITIONAL_DATACLASS = 16
    """
    The decorator of the dataclass declared without ``kw_only``.
    """

    POSITIONAL_FROZEN_DATACLASS = 28
    """
    The decorator of the frozen dataclass declared through the module without
    ``kw_only``.
    """

    POSITIONAL_METHOD = 81
    """
    The method taking a parameter positionally.
    """

    POSITIONAL_STATIC_METHOD = 88
    """
    The static method taking a parameter positionally.
    """

    POSITIONAL_OVERRIDE = 132
    """
    The positional override of a method the repository defines.
    """

    POSITIONAL_METHOD_OF_LIBRARY_SUBCLASS = 149
    """
    The positional method that no base class of the standard library subclass defines.
    """


class UnimportablePositionalDefinitionsLine(IntEnum):
    """
    Lines of :attr:`~test.krrood_test.test_development_tools.dataset_files.PositionalParametersSnippet.UNIMPORTABLE_DEFINITIONS`.
    """

    POSITIONAL_DATACLASS = 13
    """
    The decorator of the dataclass declared without ``kw_only``.
    """

    POSITIONAL_METHOD_WITHOUT_BASES = 30
    """
    The positional method of the class without bases.
    """


class OverlongAttributeDocstringLine(IntEnum):
    """
    Lines of :attr:`~test.krrood_test.test_development_tools.dataset_files.ProseSnippet.OVERLONG_ATTRIBUTE_DOCSTRING`.
    """

    DOCSTRING = 10
    """
    The first line of the overlong docstring.
    """


class CommentBlocksLine(IntEnum):
    """
    Lines of :attr:`~test.krrood_test.test_development_tools.dataset_files.ProseSnippet.COMMENT_BLOCKS`.
    """

    LONG_BLOCK = 10
    """
    The first line of the four-line comment block.
    """


# %% how long the prose is


class ProseLineCount(IntEnum):
    """
    The prose lines of the docstrings in the prose length dataset.
    """

    FIELDS_AND_LITERAL_BLOCKS = 2
    """
    The docstring of :attr:`~test.krrood_test.test_development_tools.dataset_files.ProseSnippet.FIELDS_AND_LITERAL_BLOCKS`.
    """

    OVERLONG_MODULE_DOCSTRING = 5
    """
    The docstring of :attr:`~test.krrood_test.test_development_tools.dataset_files.ProseSnippet.OVERLONG_MODULE_DOCSTRING`.
    """
