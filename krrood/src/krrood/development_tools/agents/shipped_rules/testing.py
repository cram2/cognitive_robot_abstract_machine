"""
The rules on writing and running tests.
"""

from __future__ import annotations

import unittest

import pytest

from krrood.development_tools.agents.rule import CodeReference, Section
from krrood.ormatic.utils import classproperty


class TestingRules(Section):
    """
    The rules on tests.
    """


class WritingTestsRules(Section):
    """
    The rules on writing tests.
    """

    @classproperty
    def parent(cls) -> type[Section]:
        return TestingRules

    WRITE_TESTS_WITH_PYTEST = (
        "Write tests with pytest: plain test functions or classes, pytest "
        f"fixtures, {CodeReference(pytest.raises, pytest).path} and plain `assert`; "
        f"never {CodeReference(unittest.TestCase, unittest).path} "
        "or its assertion methods"
    )

    REUSE_FIXTURES = "Reuse the existing fixtures in `conftest.py`"

    WORK_TEST_DRIVEN = (
        "Work test-driven: prove a bug with a meaningful, failing test before "
        "fixing it. Every new feature and fix is covered by tests"
    )

    NEVER_MODIFY_A_FAILING_TEST = (
        "When fixing a failing test, never modify the test itself"
    )

    NAME_TEST_CLASSES_AFTER_BEHAVIOUR = (
        "Name test classes, and the mimic classes tests use, after the pattern or "
        "behaviour they exercise, not after the external class they stand in for"
    )

    KEEP_SNIPPETS_IN_FILES = (
        "Keep code snippets in separate files of the correct type and import or "
        "read them into the test; never embed them as strings"
    )


class AssertionsRules(Section):
    """
    The rules on what a test asserts.
    """

    @classproperty
    def parent(cls) -> type[Section]:
        return TestingRules

    ASSERT_EXACT_EXPECTED_VALUES = (
        "Assert equality to the expected value whenever it can be determined, "
        "never only a weaker check such as not-None or not-empty"
    )

    ASSERT_AGAINST_THE_DEFINITION = (
        "Assert against the definition, not a retyped copy of it: the enum member, "
        f"the {CodeReference(classproperty).name}, or the value from the fixture the code under test "
        "consumed. Where a type distinguishes the case - a distinct exception "
        "class, an enum member - assert the type instead of matching message text. "
        "A retyped literal keeps passing when the original changes"
    )

    TEST_ONE_BEHAVIOUR = (
        "Each test checks the one behaviour it names and fails only for its own "
        "reason: assert exactly the values that behaviour determines, not "
        "incidental output such as the wording of an error message the test is not "
        "about. Where another test already asserts a value exactly, derive the "
        "expected value from the production code that computes it instead of "
        "hardcoding a second copy. Tests are independent of each other"
    )


class ExternalServicesAndCIRules(Section):
    """
    The rules on tests that reach external services, and on tests in CI.
    """

    @classproperty
    def parent(cls) -> type[Section]:
        return TestingRules

    EVERY_TEST_RUNS_IN_CI = (
        "Every added test is part of the CI suite. A test that needs live external "
        "calls or credentials CI does not have is skipped there (or removed, if "
        "new) so it cannot break the pipeline"
    )

    APPROVE_CI_CREDENTIALS = (
        "A test that needs credentials in CI must be approved by the user and have "
        "those credentials available in CI; otherwise it is skipped there"
    )

    MOCK_EXTERNAL_APIS = (
        "Mock external APIs; call one live only to download a dataset other tests need"
    )

    PAIR_LIVE_TESTS_WITH_MOCKED_ONES = (
        "A test that calls an external API live must have a skip condition so it "
        "does not run in CI, and must be paired with an equivalent mocked test "
        "that does"
    )


class RunningTestsRules(Section):
    """
    The rules on running tests.
    """

    @classproperty
    def parent(cls) -> type[Section]:
        return TestingRules

    RUN_TESTS_WITH_PYTEST = "Run tests with pytest"

    USE_THE_PROJECT_INTERPRETER = (
        "Always use the project's configured Python interpreter for tests and commands"
    )

    RUN_THE_SUITE_IN_PARTS = (
        "Never run the whole test suite in one invocation: run one package or a "
        "few test directories at a time, and wait for each run to finish before "
        "starting the next"
    )

    CAP_TEST_MEMORY = (
        "Make sure a test run can never exhaust the machine's working memory: "
        "check the free memory before starting, cap the run's memory so the run is "
        "killed rather than the machine, and account for other test runs already "
        "going on the same machine"
    )
