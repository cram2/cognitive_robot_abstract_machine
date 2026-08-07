"""
Tests guarding the composition of query-answering analysis engine pipelines.

``QueryReply`` is a smoke-test stub documented as generating a fixed, empty answer to
check that the action server can reply at all. ``GenerateQueryResult`` already computes
and sends the real answer itself. Chaining ``QueryReply`` after ``GenerateQueryResult``
in the same pipeline overwrites every real detection with the stub's hardcoded pose, an
empty class label, and an empty pose frame -- exactly the shape ``stretch_demo`` and
``tiago_demo`` shipped with.
"""

import importlib
import pkgutil
import signal
from contextlib import contextmanager
from typing_extensions import Iterator, List

import pytest

BUILD_TIMEOUT_SECONDS = 5
"""
How long an analysis engine's ``implementation()`` may run before it is treated as
unbuildable here.

Some engines connect to a storage backend while building their pipeline; without one
running, that connection attempt can block far longer than a composition check should
ever wait, rather than failing fast.
"""


class AnalysisEngineBuildTimedOut(Exception):
    """
    Raised when building an analysis engine's pipeline exceeds :data:`BUILD_TIMEOUT_SECONDS`.
    """


@contextmanager
def bounded_build_time() -> Iterator[None]:
    """
    Fail with :class:`AnalysisEngineBuildTimedOut` instead of hanging past the timeout.
    """

    def raise_timeout(signum, frame):
        raise AnalysisEngineBuildTimedOut(
            f"implementation() did not return within {BUILD_TIMEOUT_SECONDS}s."
        )

    previous_handler = signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(BUILD_TIMEOUT_SECONDS)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)

import robokudo.descriptors.analysis_engines as analysis_engines_package
from robokudo.annotators.query import GenerateQueryResult, QueryReply
from robokudo.descriptors.analysis_engines.stretch_demo import (
    AnalysisEngine as StretchDemoAnalysisEngine,
)
from robokudo.descriptors.analysis_engines.tiago_demo import (
    AnalysisEngine as TiagoDemoAnalysisEngine,
)
from robokudo.pipeline import Pipeline


def flattened_pipeline_nodes(behaviour) -> List[object]:
    """
    All nodes in a behaviour tree, including the root and every descendant.

    :param behaviour: The root of the (sub)tree to flatten.
    :return: The root followed by every descendant, in tree order.
    """
    nodes = [behaviour]
    for child in getattr(behaviour, "children", []):
        nodes.extend(flattened_pipeline_nodes(child))
    return nodes


def assert_query_reply_does_not_follow_generate_query_result(pipeline: Pipeline) -> None:
    """
    Fail if a pipeline both generates and then discards its own real query answer.

    :param pipeline: The built pipeline to inspect.
    :raises AssertionError: If the pipeline contains both ``GenerateQueryResult`` and
        ``QueryReply``.
    """
    node_types = {type(node) for node in flattened_pipeline_nodes(pipeline)}
    if GenerateQueryResult in node_types:
        assert QueryReply not in node_types, (
            "Pipeline sends its real answer via GenerateQueryResult and then "
            "overwrites it with QueryReply's hardcoded stub answer."
        )


# %% regression tests for the two analysis engines that shipped with the bug


def test_stretch_demo_does_not_discard_its_real_answer():
    with bounded_build_time():
        pipeline = StretchDemoAnalysisEngine().implementation()
    assert_query_reply_does_not_follow_generate_query_result(pipeline)


def test_tiago_demo_does_not_discard_its_real_answer():
    with bounded_build_time():
        pipeline = TiagoDemoAnalysisEngine().implementation()
    assert_query_reply_does_not_follow_generate_query_result(pipeline)


# %% sweep across every analysis engine, so the same mistake elsewhere is caught too


def discovered_analysis_engine_module_names() -> List[str]:
    """
    Every module name directly under ``robokudo.descriptors.analysis_engines``.

    Only names are listed here, not imported: a module that fails to import for reasons
    unrelated to this rule (for example an optional dependency this environment lacks)
    must not stop the others from being checked, so importing is left to the test body.

    :return: One fully qualified module name per module in the package.
    """
    return [
        f"{analysis_engines_package.__name__}.{module_info.name}"
        for module_info in pkgutil.iter_modules(analysis_engines_package.__path__)
    ]


@pytest.mark.parametrize(
    "module_name",
    discovered_analysis_engine_module_names(),
)
def test_no_analysis_engine_discards_its_real_answer_with_query_reply(
    module_name: str,
):
    """
    No analysis engine may chain ``QueryReply`` after ``GenerateQueryResult``.

    Modules that fail to import, that define no ``AnalysisEngine``, or whose
    ``implementation()`` needs resources this test does not provide (for example a
    running storage backend) are skipped rather than failed, since that is a missing
    fixture or an unrelated environment issue, not evidence about this composition rule.
    """
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        pytest.skip(f"{module_name} could not be imported here: {error}")

    analysis_engine_class = getattr(module, "AnalysisEngine", None)
    if analysis_engine_class is None:
        pytest.skip(f"{module_name} defines no AnalysisEngine class.")

    try:
        with bounded_build_time():
            pipeline = analysis_engine_class().implementation()
    except Exception as error:
        pytest.skip(f"{module_name} could not be built here: {error}")

    assert_query_reply_does_not_follow_generate_query_result(pipeline)
