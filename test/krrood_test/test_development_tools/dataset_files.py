"""
The input files of the development tools tests.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from krrood.patterns.path_enumeration import PathEnumeration


class DatasetFile(PathEnumeration):
    """
    The input files of this suite, under its ``dataset`` directory.
    """

    UNFORMATTED_MODULE = Path(__file__).parent / "dataset" / "unformatted_module.py"
    """
    A module ruff and docformatter both change, and agree on.
    """

    FORMATTED_MODULE = Path(__file__).parent / "dataset" / "formatted_module.py"
    """
    :attr:`UNFORMATTED_MODULE` after formatting.
    """

    DOCSTRING_CONFLICTING_MODULE = (
        Path(__file__).parent / "dataset" / "docstring_conflicting_module.py"
    )
    """
    A module whose docformatter output ruff changes again.
    """

    INVALID_SYNTAX_MODULE = (
        Path(__file__).parent / "dataset" / "invalid_syntax_module.py"
    )
    """
    A module ruff rejects.
    """

    HAND_WRITTEN_RULES = Path(__file__).parent / "dataset" / "hand_written_rules.md"
    """
    An ``AGENTS.md`` written by hand.
    """

    PROJECT_RULES_MODULE = Path(__file__).parent / "dataset" / "project_rules_module.py"
    """
    A module declaring a repository's own rules.
    """

    PROJECT_RULES_PACKAGE = Path(__file__).parent / "dataset" / "project_rules_package"
    """
    A package declaring a repository's own rules in several modules.
    """

    RULE_SECTIONS_TEXT = Path(__file__).parent / "dataset" / "rule_sections_text.md"
    """
    The ``AGENTS.md`` listing the design principles and verification sections of
    ``dataset/rule_sections.py``, with ``{source_path}`` standing for the path of that
    module.
    """

    OPTIONAL_IMPORTS_GUARDED = (
        Path(__file__).parent / "dataset" / "optional_imports_guarded.py"
    )
    """
    A module importing matplotlib only in decorated functions and under
    ``TYPE_CHECKING``.
    """

    OPTIONAL_IMPORTS_UNGUARDED = (
        Path(__file__).parent / "dataset" / "optional_imports_unguarded.py"
    )
    """
    A module importing matplotlib where nothing guards it, each such line marked with
    :attr:`DatasetMarker.UNGUARDED_IMPORT`.
    """

    OPTIONAL_IMPORTS_UNRELATED = (
        Path(__file__).parent / "dataset" / "optional_imports_unrelated.py"
    )
    """
    A module importing modules whose names only resemble matplotlib's.
    """


class DatasetMarker(StrEnum):
    """
    Comments marking lines of the dataset files that a test expects a finding on.
    """

    UNGUARDED_IMPORT = "# unguarded"
    """
    An import the optional-import check reports.
    """
