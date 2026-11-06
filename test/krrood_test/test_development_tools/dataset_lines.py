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


class HandWrittenAccessorsLine(IntEnum):
    """
    Lines of :attr:`~test.krrood_test.test_development_tools.dataset_files.FieldAccessorSnippet.HAND_WRITTEN_ACCESSORS`.
    """

    DATA_DESCRIPTOR_SETTER = 25
    """
    The ``__set__`` of the hand written data descriptor.
    """

    DATACLASS_PROPERTY_SETTER = 50
    """
    The property setter of the dataclass.
    """


class PrepositionalNamesLine(IntEnum):
    """
    Lines of :attr:`~test.krrood_test.test_development_tools.dataset_files.MethodNameSnippet.PREPOSITIONAL_NAMES`.
    """

    CLASSMETHOD = 14
    """
    The factory classmethod beginning with ``for``.
    """

    METHOD = 20
    """
    The method beginning with ``as``.
    """

    PRIVATE_METHOD = 26
    """
    The private method beginning with ``with`` after its underscore.
    """

    FUNCTION = 33
    """
    The function beginning with ``of``.
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
