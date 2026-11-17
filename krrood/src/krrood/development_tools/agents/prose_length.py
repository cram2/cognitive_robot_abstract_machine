"""
Finds docstrings and comment blocks whose prose runs longer than a line limit.
"""

from __future__ import annotations

import ast
import inspect
import io
import re
import tokenize
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from typing_extensions import Iterator, override

from krrood.development_tools.agents.checker import CodeRuleChecker, CodeRuleViolation

# %% what is measured


class ProseKind(StrEnum):
    """
    The kinds of prose measured.
    """

    DOCSTRING = "docstring"
    """
    A module, class, function or attribute docstring.
    """

    COMMENT = "comment block"
    """
    A run of consecutive lines holding only a comment.
    """


class ProsePattern(StrEnum):
    """
    The patterns separating prose from the text around it, each matched at the start of
    a line.
    """

    FIELD_LIST_ITEM = r":[^:\s][^:]*:"
    """
    A reStructuredText field such as ``:param name:``, which starts a docstring's field
    list.
    """

    SECTION_HEADER = r"# %%"
    """
    The start of a section header comment, which is a label, not prose.
    """

    TOOL_DIRECTIVE = r"#\s*(noqa|type:|pragma|fmt:|ruff:|!|-\*-)"
    """
    A comment addressed to a tool rather than a reader.
    """

    def starts(self, *, line: str) -> bool:
        """
        :return: Whether *line* starts with this pattern.
        """
        return re.match(self, line) is not None


@dataclass(kw_only=True, frozen=True)
class ProseBlock:
    """
    One docstring or run of comment lines in a source file.
    """

    kind: ProseKind
    """
    Whether the block is a docstring or a comment block.
    """

    first_line: int
    """
    The block's first line in the file, counting from one.
    """

    last_line: int
    """
    The block's last line in the file, counting from one.
    """

    prose_lines: int
    """
    How many of its lines are prose: for a docstring, the non-blank lines before its
    field list that are not indented below its own margin, such as a literal block.
    """


@dataclass(kw_only=True, frozen=True)
class SourceProse:
    """
    The docstrings and comment blocks of one Python source.
    """

    source: str
    """
    The text of the Python source.
    """

    @property
    def blocks(self) -> list[ProseBlock]:
        """
        Every docstring and comment block, in no particular order.
        """
        return [*self.docstrings(), *self.comment_blocks()]

    def docstrings(self) -> Iterator[ProseBlock]:
        """
        :return: The docstring of the module and of every class and function, and every
            attribute docstring directly below an assignment in a module or class body.
        """
        for node in ast.walk(ast.parse(self.source)):
            if not isinstance(
                node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
            ):
                continue
            for index, statement in enumerate(node.body):
                if not self.is_string(statement=statement):
                    continue
                follows_assignment = index > 0 and isinstance(
                    node.body[index - 1], ast.Assign | ast.AnnAssign
                )
                if index == 0 or (
                    follows_assignment and isinstance(node, ast.Module | ast.ClassDef)
                ):
                    yield self.docstring_block(statement=statement)

    @staticmethod
    def is_string(*, statement: ast.stmt) -> bool:
        """
        :return: Whether *statement* is a bare string literal.
        """
        return (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )

    @staticmethod
    def docstring_block(*, statement: ast.Expr) -> ProseBlock:
        """
        :param statement: A bare string literal.
        :return: It, measured as a docstring.
        """
        prose_lines = 0
        for line in inspect.cleandoc(statement.value.value).splitlines():
            if ProsePattern.FIELD_LIST_ITEM.starts(line=line):
                break
            if line.strip() and not line[0].isspace():
                prose_lines += 1
        return ProseBlock(
            kind=ProseKind.DOCSTRING,
            first_line=statement.lineno,
            last_line=statement.end_lineno,
            prose_lines=prose_lines,
        )

    def comment_blocks(self) -> Iterator[ProseBlock]:
        """
        :return: Every run of consecutive lines holding only a comment, section headers
            and comments addressed to tools left out.
        """
        lines: list[int] = []
        for token in tokenize.generate_tokens(io.StringIO(self.source).readline):
            if token.type != tokenize.COMMENT or token.line.strip() != token.string:
                continue
            if ProsePattern.SECTION_HEADER.starts(
                line=token.string
            ) or ProsePattern.TOOL_DIRECTIVE.starts(line=token.string):
                continue
            if lines and token.start[0] != lines[-1] + 1:
                yield ProseBlock(
                    kind=ProseKind.COMMENT,
                    first_line=lines[0],
                    last_line=lines[-1],
                    prose_lines=len(lines),
                )
                lines = []
            lines.append(token.start[0])
        if lines:
            yield ProseBlock(
                kind=ProseKind.COMMENT,
                first_line=lines[0],
                last_line=lines[-1],
                prose_lines=len(lines),
            )


# %% the check


@dataclass(kw_only=True, frozen=True)
class ProseLengthChecker(CodeRuleChecker):
    """
    Finds the docstrings and comment blocks holding more prose lines than their limit.
    """

    docstring_lines: int = 4
    """
    The most prose lines a docstring may hold.
    """

    comment_lines: int = 3
    """
    The most lines a comment block may hold.
    """

    @override
    def violations(self, *, path: Path) -> Iterator[CodeRuleViolation]:
        for block in SourceProse(source=path.read_text()).blocks:
            limit = (
                self.docstring_lines
                if block.kind is ProseKind.DOCSTRING
                else self.comment_lines
            )
            if block.prose_lines <= limit:
                continue
            yield CodeRuleViolation(
                path=path,
                first_line=block.first_line,
                last_line=block.last_line,
                description=f"{block.kind} has {block.prose_lines} lines of prose, more than "
                f"{limit}; shorten it",
            )
