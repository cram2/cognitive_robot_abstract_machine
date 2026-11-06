"""
Finds functions and methods whose name begins with a preposition.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from typing_extensions import Iterator, override

from krrood.development_tools.agents.checker import CodeRuleChecker, CodeRuleViolation

# %% prepositions


class LeadingPreposition(StrEnum):
    """
    The prepositions that start a phrase only the class name in front of a method
    finishes.
    """

    AS = "as"
    FOR = "for"
    WITH = "with"
    OF = "of"


# %% the check


@dataclass(frozen=True)
class LeadingPrepositionChecker(CodeRuleChecker):
    """
    Finds every function and method whose first word, leading underscores aside, is a
    :class:`LeadingPreposition`.
    """

    @override
    def violations(self, path: Path) -> Iterator[CodeRuleViolation]:
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            first_word = node.name.lstrip("_").split("_")[0]
            if first_word not in LeadingPreposition:
                continue
            yield CodeRuleViolation(
                path,
                node.lineno,
                node.end_lineno,
                f"`{node.name}` begins with the preposition `{first_word}`; name what "
                f"it does as a verb phrase",
            )
