"""
The rules on documentation.
"""

from __future__ import annotations


from krrood.development_tools.agents.rule import Section


class DocumentationRules(Section):
    """
    The rules on docstrings and comments.
    """

    DOCUMENT_EVERYTHING = (
        "Every class and method has meaningful, non-trivial documentation in "
        "reStructuredText. Every field has its own docstring directly below it, "
        "not a description in the class docstring"
    )

    OVERRIDES_INHERIT_DOCSTRINGS = (
        "An override that keeps its base contract has no docstring and inherits "
        "the base one. One that changes the contract documents only the difference "
        "and references the overridden method with `:meth:`"
    )

    DOCSTRINGS_STATE_THE_CONTRACT = (
        "A docstring states what the code does and its contract, short and to the "
        "point. It never contains:",
        (
            "how the code does it",
            (
                "the ways a caller can supply a value (for example, that it may also "
                "be given as a query instead of a concrete object)"
            ),
            (
                'the current callers or consumers ("used by X and Y") - that goes '
                "stale and reads as exhaustive"
            ),
            (
                "justification of design decisions, comparison with rejected or "
                "hypothetical designs, or review and implementation history"
            ),
            "type information - the type hints carry it",
        ),
    )

    SHORT_COMMENTS = (
        "Keep docstrings and comments short and never write walls of text. "
        "Comments are meaningful and DRY; remove comments that restate the code"
    )

    SPHINX_ROLES = (
        "Reference code with Sphinx roles (`:func:`, `:class:`, `:attr:`, "
        "`:meth:`), never with plain backticks, and use directives such as `.. "
        "note::` and `.. warning::` for notes and warnings"
    )

    NO_CAPITALS_FOR_EMPHASIS = (
        "Never use all-caps words for emphasis in docstrings or comments; use RST "
        "emphasis (`*word*`). Genuine identifiers, acronyms and enum/constant "
        "names (`UUID`, `WHERE`, `Definiteness.DEFINITE`) are exempt"
    )

    FORMAT_WITH_THE_HOOK = (
        "Always format modified Python files with ruff followed by docformatter, "
        "through the repository's pre-commit hook"
    )
