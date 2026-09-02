import time

import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import TaskStatus
from coraplex.failure_handling.failure_handler import FailureHandler
from coraplex.language import CodeNode, ParallelNode, TryAllNode, TryInOrderNode
from coraplex.plans.factories import sequential
from coraplex.plans.failures import AllChildrenFailed, PlanFailure

from .conftest import (
    ConsultationCountingHandler,
    FailingLeaf,
    SequenceRetryingStrategy,
    context_with,
    sequence_over_a_leaf,
)
from .test_failure_handler import HandledFailure


# %% escalation along the plan tree


def test_escalating_hands_the_failure_to_the_parent_node():
    handler = ConsultationCountingHandler()
    root, leaf = sequence_over_a_leaf(context_with(handler))
    failure = HandledFailure(node=leaf)

    with pytest.raises(HandledFailure):
        leaf.escalate(failure)

    assert handler.consultations == 1


def test_escalating_at_the_root_raises_the_failure():
    root, leaf = sequence_over_a_leaf(Context(world=None, robot=None))
    failure = PlanFailure(node=root)

    with pytest.raises(PlanFailure) as raised:
        root.escalate(failure)

    assert raised.value is failure


def test_a_propagated_failure_is_recorded_along_the_whole_chain():
    root, leaf = sequence_over_a_leaf(Context(world=None, robot=None))
    failure = PlanFailure(node=leaf)

    with pytest.raises(PlanFailure):
        leaf.handle_failure(failure)

    assert leaf.status == TaskStatus.FAILED
    assert root.status == TaskStatus.FAILED
    assert root.reason is failure


# %% targets that never get a perform frame


def test_a_resolution_reaches_a_target_that_has_no_perform_frame():
    """
    A sequence runs its children inside one merged execution list, so it never gets a
    perform frame.

    Escalation walks the plan tree instead of the call stack, so the resolution still
    reaches it.
    """
    handler = FailureHandler(strategies=[SequenceRetryingStrategy()])
    root, leaf = sequence_over_a_leaf(context_with(handler))
    failure = HandledFailure(node=leaf)

    leaf.handle_failure(failure)

    assert failure.resolution is None
    assert root.status != TaskStatus.FAILED


def test_the_handler_is_consulted_once_while_a_failure_escalates():
    handler = ConsultationCountingHandler()
    root, leaf = sequence_over_a_leaf(context_with(handler))
    failure = HandledFailure(node=leaf)

    with pytest.raises(HandledFailure):
        leaf.handle_failure(failure)

    assert handler.consultations == 1


# %% perform routes to the node that raised


def test_perform_hands_a_failure_to_the_node_that_raised_it():
    handler = FailureHandler(strategies=[SequenceRetryingStrategy()])
    leaf = FailingLeaf(failure_type=HandledFailure, remaining_failures=1)
    root = sequential([leaf], context_with(handler))

    root.perform()

    assert leaf.executions == 2
    assert root.status == TaskStatus.SUCCEEDED


def test_a_sequence_target_reruns_all_of_the_sequences_work():
    """
    Re-running a targeted sequence runs its whole merged work again, including children
    that already succeeded.
    """
    handler = FailureHandler(strategies=[SequenceRetryingStrategy()])
    executions = []
    first = CodeNode(code=lambda: executions.append("first"))
    failing = FailingLeaf(failure_type=HandledFailure, remaining_failures=1)
    root = sequential([first, failing], context_with(handler))

    root.perform()

    assert executions == ["first", "first"]
    assert failing.executions == 2
    assert root.status == TaskStatus.SUCCEEDED


# %% failure-tolerant nodes stop the escalation walk


def test_a_tolerated_failure_leaves_the_try_in_order_and_its_ancestors_unfailed():
    handler = ConsultationCountingHandler()
    executions = []
    failing = FailingLeaf()
    succeeding = CodeNode(code=lambda: executions.append(True))
    try_node = TryInOrderNode()
    root = sequential([try_node], context_with(handler))
    try_node.add_child(failing)
    try_node.add_child(succeeding)

    root.perform()

    assert executions == [True]
    assert failing.status == TaskStatus.FAILED
    assert try_node.status != TaskStatus.FAILED
    assert root.status == TaskStatus.SUCCEEDED
    assert root.reason is None
    assert handler.consultations == 1


def test_all_children_failing_escalates_from_the_try_in_order_node():
    try_node = TryInOrderNode()
    root = sequential([try_node], Context(world=None, robot=None))
    try_node.add_child(FailingLeaf())
    try_node.add_child(FailingLeaf())

    with pytest.raises(AllChildrenFailed):
        root.perform()

    assert try_node.status == TaskStatus.FAILED
    assert root.status == TaskStatus.FAILED


def test_a_tolerated_failure_leaves_the_try_all_and_its_ancestors_unfailed():
    handler = ConsultationCountingHandler()
    executions = []
    failing = FailingLeaf()
    succeeding = CodeNode(code=lambda: executions.append(True))
    try_node = TryAllNode()
    root = sequential([try_node], context_with(handler))
    try_node.add_child(failing)
    try_node.add_child(succeeding)

    root.perform()

    assert executions == [True]
    assert failing.status == TaskStatus.FAILED
    assert try_node.status != TaskStatus.FAILED
    assert root.status == TaskStatus.SUCCEEDED
    assert handler.consultations == 1


def test_all_children_failing_escalates_from_the_try_all_node():
    try_node = TryAllNode()
    root = sequential([try_node], Context(world=None, robot=None))
    try_node.add_child(FailingLeaf())
    try_node.add_child(FailingLeaf())

    with pytest.raises(AllChildrenFailed):
        root.perform()

    assert try_node.status == TaskStatus.FAILED
    assert root.status == TaskStatus.FAILED


def test_a_parallel_child_failure_stays_in_its_thread_until_all_children_finished():
    """
    While the children perform in worker threads, a child's failure must not touch the
    shared ancestors; the parallel node re-raises it on the main thread afterwards, from
    where it escalates normally.
    """
    handler = ConsultationCountingHandler()
    observed_root_statuses = []
    failing = FailingLeaf()
    parallel_node = ParallelNode()

    def observe_root_after_the_failure():
        deadline = time.time() + 2
        while failing.status != TaskStatus.FAILED and time.time() < deadline:
            time.sleep(0.001)
        time.sleep(0.05)
        observed_root_statuses.append(root.status)

    observer = CodeNode(code=observe_root_after_the_failure)
    root = sequential([parallel_node], context_with(handler))
    parallel_node.add_child(failing)
    parallel_node.add_child(observer)

    with pytest.raises(PlanFailure):
        root.perform()

    assert observed_root_statuses == [TaskStatus.RUNNING]
    assert parallel_node.status == TaskStatus.FAILED
    assert root.status == TaskStatus.FAILED
    assert handler.consultations == 1
