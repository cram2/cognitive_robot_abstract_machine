"""
The rules on documentation.
"""

from __future__ import annotations


from krrood.development_tools.agents.docstring_presence import (
    MissingDocstringChecker,
)
from krrood.development_tools.agents.prose_length import ProseLengthChecker
from krrood.development_tools.agents.rule import Section


class DocumentationRules(Section):
    """
    The rules on docstrings and comments.
    """

    DOCUMENT_EVERY_CONSTRUCT = (
        "Every module, class, function and field has a docstring. A field's "
        "docstring goes directly below the field, not in the class docstring. A "
        "method implementing an abstract method is exempt: the abstract method's "
        "docstring is enough",
        (),
        MissingDocstringChecker(),
    )

    MEANINGFUL_DOCUMENTATION = (
        "Documentation is written in reStructuredText and is meaningful and "
        "non-trivial: it says something the name and the type hints do not"
    )

    OVERRIDES_INHERIT_DOCSTRINGS = (
        "An implementation of an abstract method has no docstring and inherits "
        "the base one. An override of an implemented method documents only what "
        "differs from the base and references the overridden method with `:meth:`"
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

    SHORT_PROSE = (
        f"A docstring holds at most {ProseLengthChecker.docstring_lines} lines of "
        "prose, field lists and literal blocks aside, and a comment block at most "
        f"{ProseLengthChecker.comment_lines} lines",
        (),
        ProseLengthChecker(),
    )

    MEANINGFUL_COMMENTS = (
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
