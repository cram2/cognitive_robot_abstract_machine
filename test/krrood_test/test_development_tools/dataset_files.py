"""
The input files of the development tools tests.
"""

from __future__ import annotations

from pathlib import Path

from krrood.patterns.path_enumeration import PathEnumeration

from .dataset.field_accessors import hand_written_accessors
from .dataset.leading_prepositions import prepositional_names, verb_phrase_names
from .dataset.missing_docstrings import (
    documented_module,
    no_module_docstring,
    undocumented_definitions,
)
from .dataset.positional_parameters import positional_definitions
from .dataset.prose_length import (
    comment_blocks,
    fields_and_literal_blocks,
    overlong_attribute_docstring,
    overlong_module_docstring,
)


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

    PROJECT_RULES_IMPORTING_REPOSITORY = (
        Path(__file__).parent / "dataset" / "project_rules_importing_repository.py"
    )
    """
    A module declaring a repository's own rules from the wording of another module of
    that repository.
    """

    REPOSITORY_WORDING = Path(__file__).parent / "dataset" / "repository_wording.py"
    """
    The module of the repository that :attr:`PROJECT_RULES_IMPORTING_REPOSITORY` imports.
    """

    PROJECT_RULES_PACKAGE = Path(__file__).parent / "dataset" / "project_rules_package"
    """
    A package declaring a repository's own rules in several modules.
    """

    PROJECT_SKILLS_MODULE = (
        Path(__file__).parent / "dataset" / "project_skills_module.py"
    )
    """
    A module declaring a repository's skills.
    """

    SKILL_TEXT = Path(__file__).parent / "dataset" / "skill_text.md"
    """
    The ``SKILL.md`` of the deploying skill of :attr:`PROJECT_SKILLS_MODULE`, with
    ``{generate_command}`` and ``{source_path}`` standing for the command generating it
    and the path of the module declaring it.
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

    OVERLONG_MODULE_DOCSTRING = Path(overlong_module_docstring.__file__)
    """
    A module docstring of five prose lines.
    """

    FIELDS_AND_LITERAL_BLOCKS = Path(fields_and_literal_blocks.__file__)
    """
    A function docstring of two prose lines, followed by a literal block and a field list.
    """

    OVERLONG_ATTRIBUTE_DOCSTRING = Path(overlong_attribute_docstring.__file__)
    """
    A field whose docstring holds five prose lines.
    """

    COMMENT_BLOCKS = Path(comment_blocks.__file__)
    """
    A three-line and a four-line comment block, and a section header above the first.
    """


class DocstringSnippet(PathEnumeration):
    """
    The Python sources the missing docstring check reads.
    """

    DOCUMENTED_MODULE = Path(documented_module.__file__)
    """
    A module that documents everything.
    """

    NO_MODULE_DOCSTRING = Path(no_module_docstring.__file__)
    """
    A module without a docstring whose function is documented.
    """

    UNDOCUMENTED_DEFINITIONS = Path(undocumented_definitions.__file__)
    """
    A documented module whose class leaves one field and two methods undocumented, and
    implements an abstract method, which needs no docstring.
    """

    UNIMPORTABLE_DEFINITIONS = Path(undocumented_definitions.__file__).with_name(
        "unimportable_definitions.py"
    )
    """
    A module that raises when imported, whose method and function are undocumented.
    """


class PositionalParametersSnippet(PathEnumeration):
    """
    The Python sources the keyword-only checks read.
    """

    POSITIONAL_DEFINITIONS = Path(positional_definitions.__file__)
    """
    A module with positional and keyword-only dataclasses and methods, and the ones the
    rules exempt.
    """

    UNIMPORTABLE_DEFINITIONS = Path(positional_definitions.__file__).with_name(
        "unimportable_definitions.py"
    )
    """
    A module that raises when imported, with a positional dataclass, a positional method
    of a class without bases and one of a class with a base.
    """


class FieldAccessorSnippet(PathEnumeration):
    """
    The Python sources the hand written field accessor check reads.
    """

    HAND_WRITTEN_ACCESSORS = Path(hand_written_accessors.__file__)
    """
    A module with a hand written data descriptor and a dataclass property setter, next
    to a plain class property setter and a descriptor built on the base, which pass.
    """


class MethodNameSnippet(PathEnumeration):
    """
    The Python sources the leading preposition check reads.
    """

    PREPOSITIONAL_NAMES = Path(prepositional_names.__file__)
    """
    A module whose classmethod, method, private method and function each begin their
    name with a preposition.
    """

    VERB_PHRASE_NAMES = Path(verb_phrase_names.__file__)
    """
    A module whose method names are verb phrases, an alternative constructor and a
    conversion.
    """
