"""
Tests of the check for function and method names that begin with a preposition.
"""

from __future__ import annotations

from krrood.development_tools.agents.leading_preposition import (
    LeadingPrepositionChecker,
)

from .dataset_files import MethodNameSnippet
from .dataset_lines import PrepositionalNamesLine

# %% names beginning with a preposition


def reported_first_lines(snippet: MethodNameSnippet) -> list[int]:
    """
    :return: The first line of each name in *snippet* the check reports, top to bottom.
    """
    return sorted(
        violation.first_line
        for violation in LeadingPrepositionChecker().violations(path=snippet)
    )


def test_every_function_and_method_beginning_with_a_preposition_is_reported() -> None:
    """
    Classmethods, methods, private methods and functions are all read without their
    class in a traceback.
    """
    assert reported_first_lines(MethodNameSnippet.PREPOSITIONAL_NAMES) == [
        PrepositionalNamesLine.CLASSMETHOD,
        PrepositionalNamesLine.METHOD,
        PrepositionalNamesLine.PRIVATE_METHOD,
        PrepositionalNamesLine.FUNCTION,
    ]


def test_verb_phrases_alternative_constructors_and_conversions_are_not_reported() -> (
    None
):
    """
    ``from_`` and ``to_`` are allowed, and a verb whose letters merely begin like a
    preposition is not one.
    """
    assert reported_first_lines(MethodNameSnippet.VERB_PHRASE_NAMES) == []
