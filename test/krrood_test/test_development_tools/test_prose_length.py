"""
Tests of the check finding docstrings and comment blocks with too much prose.
"""

from __future__ import annotations

from krrood.development_tools.agents.prose_length import (
    ProseLengthChecker,
    SourceProse,
)

from .dataset_files import ProseSnippet
from .dataset_lines import (
    CommentBlocksLine,
    ModuleLine,
    OverlongAttributeDocstringLine,
    ProseLineCount,
)

# %% measuring


def overlong_first_lines(snippet: ProseSnippet) -> list[int]:
    """
    :return: The first line of each overlong block of *snippet*.
    """
    return [
        violation.first_line
        for violation in ProseLengthChecker().violations(path=snippet)
    ]


def test_a_docstring_over_its_limit_is_reported() -> None:
    assert overlong_first_lines(ProseSnippet.OVERLONG_MODULE_DOCSTRING) == [
        ModuleLine.FIRST
    ]


def test_field_lists_and_literal_blocks_are_not_prose() -> None:
    """
    Only the prose of a docstring counts towards its limit.
    """
    (block,) = SourceProse(
        source=ProseSnippet.FIELDS_AND_LITERAL_BLOCKS.read_text()
    ).blocks

    assert block.prose_lines == ProseLineCount.FIELDS_AND_LITERAL_BLOCKS


def test_an_attribute_docstring_is_measured() -> None:
    assert overlong_first_lines(ProseSnippet.OVERLONG_ATTRIBUTE_DOCSTRING) == [
        OverlongAttributeDocstringLine.DOCSTRING
    ]


def test_a_comment_block_over_its_limit_is_reported_and_section_headers_are_not_prose() -> (
    None
):
    assert overlong_first_lines(ProseSnippet.COMMENT_BLOCKS) == [
        CommentBlocksLine.LONG_BLOCK
    ]


def test_a_raised_limit_lets_the_block_through() -> None:
    """
    A docstring within a raised limit is not reported.
    """
    checker = ProseLengthChecker(
        docstring_lines=ProseLineCount.OVERLONG_MODULE_DOCSTRING
    )

    assert list(checker.violations(path=ProseSnippet.OVERLONG_MODULE_DOCSTRING)) == []
