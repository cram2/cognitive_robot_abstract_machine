"""
Finds the code that breaks a rule that lint rules of ruff enforce.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from fnmatch import fnmatchcase
from pathlib import Path

from ruff import find_ruff_bin
from typing_extensions import Any, Iterator, override

from krrood.development_tools.agents.checker import CodeRuleChecker, CodeRuleViolation
from krrood.development_tools.exceptions import RuffCheckFailed
from krrood.development_tools.formatting import RuffArgument

# %% ruff's vocabulary


class RuffRule(StrEnum):
    """
    The lint rules of ruff that rules build on, named as ruff names them and valued with
    their codes.
    """

    MUTABLE_ARGUMENT_DEFAULT = "B006"
    """
    A parameter defaults to a list, dictionary or set.
    """

    ASSERT = "S101"
    """
    An ``assert`` statement.
    """

    MISSING_TYPE_FUNCTION_ARGUMENT = "ANN001"
    """
    A parameter without a type hint.
    """

    MISSING_TYPE_ARGS = "ANN002"
    """
    Variadic positional parameters without a type hint.
    """

    MISSING_TYPE_KWARGS = "ANN003"
    """
    Variadic keyword parameters without a type hint.
    """

    MISSING_RETURN_TYPE_UNDOCUMENTED_PUBLIC_FUNCTION = "ANN201"
    """
    A public function or method without a return type.
    """

    MISSING_RETURN_TYPE_PRIVATE_FUNCTION = "ANN202"
    """
    A private function or method without a return type.
    """

    MISSING_RETURN_TYPE_SPECIAL_METHOD = "ANN204"
    """
    A special method, ``__init__`` included, without a return type.
    """

    MISSING_RETURN_TYPE_STATIC_METHOD = "ANN205"
    """
    A static method without a return type.
    """

    MISSING_RETURN_TYPE_CLASS_METHOD = "ANN206"
    """
    A class method without a return type.
    """

    NON_PEP585_ANNOTATION = "UP006"
    """
    A generic alias of ``typing``, such as ``List``, instead of the builtin.
    """

    NON_PEP604_ANNOTATION_UNION = "UP007"
    """
    A ``Union`` instead of ``X | Y``.
    """

    NON_PEP604_ANNOTATION_OPTIONAL = "UP045"
    """
    An ``Optional`` instead of ``X | None``.
    """

    QUOTED_ANNOTATION = "UP037"
    """
    A quoted type hint in a module that postpones its annotations.
    """

    MISSING_REQUIRED_IMPORT = "I002"
    """
    A module lacks an import the settings require.
    """

    RELATIVE_IMPORTS = "TID252"
    """
    A relative import of the kind the settings ban.
    """

    BANNED_API = "TID251"
    """
    A use of a module or member the settings ban.
    """

    PYTEST_UNITTEST_ASSERTION = "PT009"
    """
    An assertion method of :class:`unittest.TestCase`.
    """

    PYTEST_UNITTEST_RAISES_ASSERTION = "PT027"
    """
    The ``assertRaises`` family of :class:`unittest.TestCase`.
    """

    MODULE_IMPORT_NOT_AT_TOP_OF_FILE = "E402"
    """
    An import after other code of the module.
    """

    MUTABLE_DATACLASS_DEFAULT = "RUF008"
    """
    A dataclass field defaults to a list, dictionary or set.
    """

    RAISE_VANILLA_CLASS = "TRY002"
    """
    :class:`Exception` itself raised.
    """

    RAISE_VANILLA_ARGS = "TRY003"
    """
    A long message passed to an exception where it is raised.
    """

    GLOBAL_STATEMENT = "PLW0603"
    """
    A ``global`` statement rebinding a module variable.
    """

    BUILTIN_VARIABLE_SHADOWING = "A001"
    """
    A variable named after a builtin.
    """

    BUILTIN_ARGUMENT_SHADOWING = "A002"
    """
    A parameter named after a builtin.
    """

    BUILTIN_ATTRIBUTE_SHADOWING = "A003"
    """
    A class attribute or method named after a builtin the class body then uses.
    """

    BUILTIN_IMPORT_SHADOWING = "A004"
    """
    An import binding the name of a builtin.
    """

    STDLIB_MODULE_SHADOWING = "A005"
    """
    A top-level module named after a module of the standard library.
    """

    BUILTIN_LAMBDA_ARGUMENT_SHADOWING = "A006"
    """
    A parameter of a lambda named after a builtin.
    """

    IMPORT_SHADOWED_BY_LOOP_VARIABLE = "F402"
    """
    A loop variable rebinding an imported name.
    """

    REDEFINED_WHILE_UNUSED = "F811"
    """
    A name bound again before its first binding is used.
    """

    REDEFINED_ARGUMENT_FROM_LOCAL = "PLR1704"
    """
    A loop, ``with`` or ``except`` target rebinding a parameter.
    """

    REDEFINED_LOOP_NAME = "PLW2901"
    """
    A loop variable assigned to inside its loop.
    """

    SUPERFLUOUS_ELSE_RETURN = "RET505"
    """
    An ``else`` after a branch that returns.
    """

    SUPERFLUOUS_ELSE_RAISE = "RET506"
    """
    An ``else`` after a branch that raises.
    """

    COLLAPSIBLE_IF = "SIM102"
    """
    An ``if`` whose condition could join that of the ``if`` around it.
    """

    TOO_MANY_NESTED_BLOCKS = "PLR1702"
    """
    Blocks nested more than five deep; a preview rule.
    """

    COMMENTED_OUT_CODE = "ERA001"
    """
    A comment holding code.
    """

    RUNTIME_IMPORT_IN_TYPE_CHECKING_BLOCK = "TC004"
    """
    An import under ``TYPE_CHECKING`` that the module uses when it runs.
    """


class RuffSetting(StrEnum):
    """
    The settings lint rules are run with, each a TOML key-value pair of ruff's
    configuration.
    """

    FUTURE_ANNOTATIONS_REQUIRED = (
        'lint.isort.required-imports = ["from __future__ import annotations"]'
    )
    """
    Every module that holds code postpones its annotations.
    """

    RELATIVE_IMPORTS_BANNED = 'lint.flake8-tidy-imports.ban-relative-imports = "all"'
    """
    Relative imports are banned from the same package too, not only from parents.
    """

    TYPING_BANNED = (
        'lint.flake8-tidy-imports.banned-api."typing".msg = '
        '"import it from `typing_extensions`"'
    )
    """
    ``typing`` is banned in favour of ``typing_extensions``.
    """

    UNITTEST_TEST_CASE_BANNED = (
        'lint.flake8-tidy-imports.banned-api."unittest.TestCase".msg = '
        '"write a plain pytest test"'
    )
    """
    :class:`unittest.TestCase` is banned in favour of plain pytest tests.
    """

    PROTOCOL_BANNED = (
        "lint.flake8-tidy-imports.banned-api = { "
        '"typing.Protocol" = { msg = "declare an explicit superclass or mixin" }, '
        '"typing_extensions.Protocol" = { msg = "declare an explicit superclass or '
        'mixin" } }'
    )
    """
    ``Protocol`` is banned, from ``typing`` and ``typing_extensions`` alike.
    """

    PREVIEW_ENABLED = "lint.preview = true"
    """
    Lint rules still in preview run.
    """


class RuffCheckArgument(StrEnum):
    """
    The command line arguments ruff checks a file with, besides
    :attr:`~krrood.development_tools.formatting.RuffArgument.ISOLATED`.
    """

    CHECK = "check"
    """
    Run ruff's linter.
    """

    NO_CACHE = "--no-cache"
    """
    Leave no cache directory in the repository.
    """

    EXIT_ZERO = "--exit-zero"
    """
    Exit successfully when ruff finds code breaking a lint rule, so that only a failure
    to run exits with an error.
    """

    OUTPUT_FORMAT = "--output-format"
    """
    Choose how ruff prints what it finds; followed by :attr:`JSON`.
    """

    JSON = "json"
    """
    Print what ruff finds as a JSON list.
    """

    SELECT = "--select"
    """
    Run only the lint rules whose comma-separated codes follow.
    """

    CONFIG = "--config"
    """
    Apply the setting that follows.
    """

    TARGET_VERSION = "--target-version"
    """
    Choose the Python version the code must run on; followed by
    :attr:`OLDEST_SUPPORTED_PYTHON`.
    """

    OLDEST_SUPPORTED_PYTHON = "py310"
    """
    The oldest Python krrood supports, so that which lint rules apply does not depend on
    ruff's default.
    """

    PATHS_FOLLOW = "--"
    """
    Read every following argument as a path, never as an option.
    """


class RuffDiagnosticKey(StrEnum):
    """
    The keys read from a finding in ruff's JSON output.
    """

    CODE = "code"
    """
    The code of the lint rule; null for a syntax error.
    """

    MESSAGE = "message"
    """
    What is wrong.
    """

    LOCATION = "location"
    """
    Where the offending code starts.
    """

    END_LOCATION = "end_location"
    """
    Where the offending code ends.
    """

    ROW = "row"
    """
    The line of a location, counting from one.
    """


@dataclass(kw_only=True, frozen=True)
class RuffDiagnostic:
    """
    One finding ruff reports in a file.
    """

    rule_code: str | None
    """
    The code of the lint rule the code breaks; none for a syntax error.
    """

    message: str
    """
    What is wrong, in ruff's words.
    """

    first_line: int
    """
    The first line of the offending code, counting from one.
    """

    last_line: int
    """
    The last line of the offending code, counting from one.
    """

    @classmethod
    def from_json(cls, *, data: dict[str, Any]) -> RuffDiagnostic:
        """
        :param data: One finding of ruff's JSON output.
        :return: The finding it describes.
        """
        return cls(
            rule_code=data[RuffDiagnosticKey.CODE],
            message=data[RuffDiagnosticKey.MESSAGE],
            first_line=data[RuffDiagnosticKey.LOCATION][RuffDiagnosticKey.ROW],
            last_line=data[RuffDiagnosticKey.END_LOCATION][RuffDiagnosticKey.ROW],
        )


# %% the modules a rule covers


class PytestPathPattern(StrEnum):
    """
    The patterns of directory and file names that mark a module as part of a test suite:
    the usual test directories and the files pytest collects.
    """

    TEST_DIRECTORY = "test"
    """
    A test directory, named in the singular.
    """

    TESTS_DIRECTORY = "tests"
    """
    A test directory, named in the plural.
    """

    PREFIXED_TEST_FILE = "test_*.py"
    """
    A test file named with pytest's prefix.
    """

    SUFFIXED_TEST_FILE = "*_test.py"
    """
    A test file named with pytest's suffix.
    """

    FIXTURE_FILE = "conftest.py"
    """
    A file of pytest fixtures.
    """

    @classmethod
    def is_test_module(cls, *, path: Path) -> bool:
        """
        :param path: A Python source file.
        :return: Whether a directory or file name on its path below the working
            directory matches one of the patterns.
        """
        absolute_path = path.resolve()
        working_directory = Path.cwd()
        parts = (
            absolute_path.relative_to(working_directory).parts
            if absolute_path.is_relative_to(working_directory)
            else absolute_path.parts
        )
        return any(fnmatchcase(part, pattern) for part in parts for pattern in cls)


class CheckedModules(Enum):
    """
    The modules a rule covers.
    """

    EVERY_MODULE = auto()
    """
    Tests and the code they test alike.
    """

    TESTS = auto()
    """
    Only the modules of a test suite.
    """

    OUTSIDE_TESTS = auto()
    """
    Every module but those of a test suite.
    """

    def includes(self, *, path: Path) -> bool:
        """
        :param path: A Python source file.
        :return: Whether the rule covers it.
        """
        if self is CheckedModules.EVERY_MODULE:
            return True
        return PytestPathPattern.is_test_module(path=path) == (
            self is CheckedModules.TESTS
        )


# %% the check


@dataclass(kw_only=True, frozen=True)
class RuffRuleChecker(CodeRuleChecker):
    """
    Finds the code breaking any of :attr:`lint_rules`, in the modules :attr:`modules`
    covers.
    """

    lint_rules: tuple[RuffRule, ...]
    """
    The lint rules that together enforce the rule.
    """

    settings: tuple[RuffSetting, ...] = ()
    """
    The settings the lint rules are run with.
    """

    modules: CheckedModules = CheckedModules.EVERY_MODULE
    """
    The modules the rule covers.
    """

    @override
    def violations(self, *, path: Path) -> Iterator[CodeRuleViolation]:
        if not self.modules.includes(path=path):
            return
        for diagnostic in self.find_diagnostics(path=path):
            if diagnostic.rule_code not in self.lint_rules:
                continue
            yield CodeRuleViolation(
                path=path,
                first_line=diagnostic.first_line,
                last_line=diagnostic.last_line,
                description=f"{diagnostic.message} (ruff {diagnostic.rule_code})",
            )

    def find_diagnostics(self, *, path: Path) -> list[RuffDiagnostic]:
        """
        :param path: A Python source file.
        :return: What ruff finds in it with :attr:`lint_rules` and :attr:`settings`,
            ignoring every configuration file.
        :raises RuffCheckFailed: If ruff exits with an error.
        """
        settings = [
            argument
            for setting in self.settings
            for argument in (RuffCheckArgument.CONFIG, setting)
        ]
        result = subprocess.run(
            [
                find_ruff_bin(),
                RuffCheckArgument.CHECK,
                RuffArgument.ISOLATED,
                RuffCheckArgument.NO_CACHE,
                RuffCheckArgument.EXIT_ZERO,
                RuffCheckArgument.OUTPUT_FORMAT,
                RuffCheckArgument.JSON,
                RuffCheckArgument.TARGET_VERSION,
                RuffCheckArgument.OLDEST_SUPPORTED_PYTHON,
                RuffCheckArgument.SELECT,
                ",".join(self.lint_rules),
                *settings,
                RuffCheckArgument.PATHS_FOLLOW,
                str(path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuffCheckFailed(path=path, standard_error=result.stderr)
        return [
            RuffDiagnostic.from_json(data=data) for data in json.loads(result.stdout)
        ]
