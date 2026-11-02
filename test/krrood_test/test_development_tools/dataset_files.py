"""
The input files of the development tools tests.
"""

from __future__ import annotations

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


class ProseSnippet(PathEnumeration):
    """
    The Python sources the prose length check measures.
    """

    OVERLONG_MODULE_DOCSTRING = (
        Path(__file__).parent
        / "dataset"
        / "prose_length"
        / "overlong_module_docstring.py"
    )
    """
    A module docstring of five prose lines.
    """

    FIELDS_AND_LITERAL_BLOCKS = (
        Path(__file__).parent
        / "dataset"
        / "prose_length"
        / "fields_and_literal_blocks.py"
    )
    """
    A function docstring of two prose lines, followed by a literal block and a field list.
    """

    OVERLONG_ATTRIBUTE_DOCSTRING = (
        Path(__file__).parent
        / "dataset"
        / "prose_length"
        / "overlong_attribute_docstring.py"
    )
    """
    A field whose docstring holds five prose lines.
    """

    COMMENT_BLOCKS = (
        Path(__file__).parent / "dataset" / "prose_length" / "comment_blocks.py"
    )
    """
    A three-line and a four-line comment block, and a section header above the first.
    """


class DocstringSnippet(PathEnumeration):
    """
    The Python sources the missing docstring check reads.
    """

    DOCUMENTED_MODULE = (
        Path(__file__).parent
        / "dataset"
        / "missing_docstrings"
        / "documented_module.py"
    )
    """
    A module that documents everything.
    """

    NO_MODULE_DOCSTRING = (
        Path(__file__).parent
        / "dataset"
        / "missing_docstrings"
        / "no_module_docstring.py"
    )
    """
    A module without a docstring whose function is documented.
    """

    UNDOCUMENTED_DEFINITIONS = (
        Path(__file__).parent
        / "dataset"
        / "missing_docstrings"
        / "undocumented_definitions.py"
    )
    """
    A documented module whose class leaves one field and one method undocumented, and
    whose override needs no docstring.
    """
