"""
Finds functions and methods whose name begins with a preposition.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Iterator, override

from krrood.development_tools.agents.checker import CodeRuleChecker, CodeRuleViolation
from krrood.entity_query_language.verbalization.vocabulary.english import Prepositions
from krrood.ormatic.utils import classproperty

# %% the check


@dataclass(kw_only=True, frozen=True)
class LeadingPrepositionChecker(CodeRuleChecker):
    """
    Finds every function and method whose first word, leading underscores aside, is one
    of the :attr:`leading_prepositions`.
    """

    @classproperty
    def leading_prepositions(cls) -> tuple[Prepositions, ...]:
        """
        The prepositions that start a phrase only the class name in front of a method
        finishes.
        """
        return Prepositions.AS, Prepositions.FOR, Prepositions.WITH, Prepositions.OF

    @override
    def violations(self, *, path: Path) -> Iterator[CodeRuleViolation]:
        leading_words = {preposition.text for preposition in self.leading_prepositions}
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            first_word = node.name.lstrip("_").split("_")[0]
            if first_word not in leading_words:
                continue
            yield CodeRuleViolation(
                path=path,
                first_line=node.lineno,
                last_line=node.end_lineno,
                description=f"`{node.name}` begins with the preposition `{first_word}`; name what "
                f"it does as a verb phrase",
            )
