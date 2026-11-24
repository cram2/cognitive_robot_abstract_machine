"""
The input files of the development tools tests.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from krrood.patterns.path_enumeration import PathEnumeration

from .dataset.field_accessors import hand_written_accessors
from .dataset.leading_prepositions import prepositional_names, verb_phrase_names
from .dataset.missing_docstrings import (
    undocumented_classproperties,
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

    DEPENDENT_WORKSPACE = Path(__file__).parent / "dataset" / "dependent_workspace"
    """
    A workspace whose members ``base``, ``middle`` and ``top`` depend on each other in a
    chain, next to ``side``, which depends on none, and ``family``, whose members
    ``family-core`` and ``family-extra`` nest in its directory.
    """

    LIBRARY_TEST_RUNS = Path(__file__).parent / "dataset" / "library_test_runs"
    """
    Tests a library's test run is checked with, run in a pytest process of their own.
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

    EXCLUDING_PYPROJECT = (
        Path(__file__).parent / "dataset" / "check_exclusion" / "pyproject.toml"
    )
    """
    A ``pyproject.toml`` keeping the files under ``excluded/`` out of ``krrood check``.
    """

    PROJECT_ISSUE_FORMS_MODULE = (
        Path(__file__).parent / "dataset" / "project_issue_forms_module.py"
    )
    """
    A module declaring a repository's issue forms.
    """

    ISSUE_FORM_TEXT = Path(__file__).parent / "dataset" / "issue_form_text.yml"
    """
    The file of the crash report form of :attr:`PROJECT_ISSUE_FORMS_MODULE`, with
    ``{generate_command}`` and ``{source_path}`` standing for the command generating it
    and the path of the module declaring it.
    """

    ISSUE_FORM_WITHOUT_TYPE_TEXT = (
        Path(__file__).parent / "dataset" / "issue_form_without_type_text.yml"
    )
    """
    The file of the announcement form of :attr:`PROJECT_ISSUE_FORMS_MODULE`, which sets
    no issue type, with the placeholders of :attr:`ISSUE_FORM_TEXT`.
    """

    CRASH_REPORT_ISSUE_BODY = (
        Path(__file__).parent / "dataset" / "crash_report_issue_body.md"
    )
    """
    The body GitHub gives an issue opened with the crash report form of
    :attr:`PROJECT_ISSUE_FORMS_MODULE`, answering both of its fields.
    """

    CRASH_REPORT_UNANSWERED_ISSUE_BODY = (
        Path(__file__).parent / "dataset" / "crash_report_unanswered_issue_body.md"
    )
    """
    :attr:`CRASH_REPORT_ISSUE_BODY` with its optional script left empty.
    """

    CRASH_REPORT_SCRIPT = Path(__file__).parent / "dataset" / "crash_report_script.py"
    """
    The script :attr:`CRASH_REPORT_ISSUE_BODY` answers with, without its code fence.
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

    UNDOCUMENTED_CLASSPROPERTIES = Path(undocumented_classproperties.__file__)
    """
    A documented module whose class implements an abstract classproperty, which needs
    no docstring, and overrides an implemented one, both undocumented.
    """


class CoreBoundarySnippet(PathEnumeration):
    """
    The Python sources the core boundary checker reads, inside mimic distributions
    whose ``pyproject.toml`` the checker finds above them.
    """

    DISTRIBUTION = Path(__file__).parent / "dataset" / "core_boundary_distribution"
    """
    The mimic distribution whose core may load only typing-extensions.
    """

    INSIDE_THE_BOUNDARY = (
        DISTRIBUTION / "src" / "mimic_package" / "core" / "inside_the_boundary.py"
    )
    """
    A core module loading only core modules, core dependencies and the standard library.
    """

    CROSSING_THE_BOUNDARY = INSIDE_THE_BOUNDARY.with_name("crossing_the_boundary.py")
    """
    A core module loading modules outside core, each such import marked with
    :attr:`DatasetMarker.CROSSING_IMPORT`.
    """

    MODULE_OUTSIDE_CORE = INSIDE_THE_BOUNDARY.parents[1] / "drawing.py"
    """
    A module outside core loading matplotlib.
    """

    MODULE_OF_UNDECLARED_CORE = (
        Path(__file__).parent
        / "dataset"
        / "undeclared_core_distribution"
        / "src"
        / "undeclared_package"
        / "drawing.py"
    )
    """
    A module loading matplotlib in a distribution declaring no core.
    """

    CORE_WITHOUT_DEPENDENCIES = (
        Path(__file__).parent
        / "dataset"
        / "dependency_free_core_distribution"
        / "src"
        / "dependency_free_package"
        / "__init__.py"
    )
    """
    A core package in a distribution declaring no core dependencies, each import of a
    distribution marked with :attr:`DatasetMarker.CROSSING_IMPORT`.
    """


class DatasetMarker(StrEnum):
    """
    Comments marking lines of the dataset files that a test expects a finding on.
    """

    CROSSING_IMPORT = "# crosses"
    """
    An import the core boundary checker reports.
    """

    RUFF_FINDING = "# reported"
    """
    A line the rule whose lint rules a snippet of :class:`RuffSnippet` breaks reports.
    """


class MimicModule(StrEnum):
    """
    The dotted names of modules of the mimic distribution
    :attr:`CoreBoundarySnippet.DISTRIBUTION`.
    """

    DRAWING = "mimic_package.drawing"
    """
    The module outside core that loads matplotlib.
    """


class MimicDistribution(StrEnum):
    """
    The distributions the mimic distribution's modules load.
    """

    MATPLOTLIB = "matplotlib"
    """
    The distribution only the mimic distribution's drawing extra requires.
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


class RuffSnippet(PathEnumeration):
    """
    The Python sources the checkers built on ruff read, each breaking the rule it names
    on the lines marked with :attr:`DatasetMarker.RUFF_FINDING`.
    """

    DIRECTORY = Path(__file__).parent / "dataset" / "ruff_findings"
    """
    The directory holding the snippets.
    """

    MUTABLE_DEFAULT = DIRECTORY / "mutable_default.py"
    """
    A function defaulting a parameter to a list, which also asserts on an unmarked line.
    """

    ASSERTION = DIRECTORY / "assertion.py"
    """
    A function checking a precondition with ``assert``.
    """

    MISSING_TYPE_HINTS = DIRECTORY / "missing_type_hints.py"
    """
    Every kind of parameter and return value left unannotated once, and a function
    annotated with ``Any``.
    """

    TYPING_CONSTRUCTS = DIRECTORY / "typing_constructs.py"
    """
    Imports from ``typing`` and annotations with its generic aliases.
    """

    MISSING_FUTURE_IMPORT = DIRECTORY / "missing_future_import.py"
    """
    A module without the import of postponed annotations.
    """

    QUOTED_ANNOTATIONS = DIRECTORY / "quoted_annotations.py"
    """
    A module postponing its annotations that still quotes a parameter and a return type.
    """

    RELATIVE_IMPORTS = DIRECTORY / "relative_imports.py"
    """
    Imports of modules of the same package, written relatively.
    """

    UNITTEST_CASE = DIRECTORY / "unittest_case.py"
    """
    A test on :class:`unittest.TestCase` using its assertion methods.
    """

    IMPORT_AFTER_CODE = DIRECTORY / "import_after_code.py"
    """
    An import after code has run, next to the blocks of imports that may follow
    other imports.
    """

    PROTOCOL_INTERFACE = DIRECTORY / "protocol_interface.py"
    """
    Interfaces declared as protocols, imported by name and through their module.
    """

    MUTABLE_DATACLASS_DEFAULT = DIRECTORY / "mutable_dataclass_default.py"
    """
    A dataclass field defaulting to a list, next to one using a default factory.
    """

    VANILLA_EXCEPTIONS = DIRECTORY / "vanilla_exceptions.py"
    """
    :class:`Exception` raised itself and a message worded where it is raised, next to
    a custom exception.
    """

    GLOBAL_STATEMENT = DIRECTORY / "global_statement.py"
    """
    A function rebinding a module variable.
    """

    BUILTIN_NAMES = DIRECTORY / "builtin_names.py"
    """
    A variable, a parameter and a method named after builtins.
    """

    BUILTIN_IMPORT_NAMES = DIRECTORY / "builtin_import_names.py"
    """
    An import and a lambda parameter named after builtins.
    """

    REBOUND_NAMES = DIRECTORY / "rebound_names.py"
    """
    An import bound twice, a loop variable assigned in its loop, a loop over a
    parameter's name and a loop over an import's name.
    """

    STANDARD_LIBRARY_NAME = DIRECTORY / "standard_library_name.py"
    """
    A module that breaks no rule but its name, once copied under the name of a standard
    library module.
    """

    SUPERFLUOUS_NESTING = DIRECTORY / "superfluous_nesting.py"
    """
    An ``else`` after a return and after a raise, an ``if`` that could join its
    parent's, and blocks nested six deep.
    """

    COMMENTED_OUT_CODE = DIRECTORY / "commented_out_code.py"
    """
    A line of code commented out, next to an explaining comment.
    """

    TYPE_CHECKING_RUNTIME_IMPORT = DIRECTORY / "type_checking_runtime_import.py"
    """
    Imports under ``TYPE_CHECKING``, one of them used when the module runs.
    """

    SUPPRESSED_FINDING = DIRECTORY / "suppressed_finding.py"
    """
    A mutable default marked as a deliberate exception with ruff's ``noqa`` comment.
    """
