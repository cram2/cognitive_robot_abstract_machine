from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llmr.execution_loop import ExecutionLoop, ExecutionResult
from llmr.pipeline.clarification import (
    ArmCapacityError,
    ArmCapacityRequest,
    ClarificationNeededError,
    ClarificationRequest,
)
from llmr.planning.motion_precondition_planner import ExecutionState, PreconditionResult


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_loop(
    pipeline=None,
    world=None,
    context=None,
    decomposer=None,
    recovery_handler=None,
) -> ExecutionLoop:
    """Construct an ExecutionLoop with MagicMock dependencies."""
    loop = ExecutionLoop(
        world=world or MagicMock(),
        pipeline=pipeline or MagicMock(),
        context=context or MagicMock(),
        decomposer=decomposer,
        recovery_handler=recovery_handler,
    )
    # Replace the real planner with a mock so tests don't need a real world
    loop._planner = MagicMock()
    return loop


def _make_precondition_result(action=None) -> PreconditionResult:
    action = action or MagicMock()
    return PreconditionResult(preconditions=[MagicMock()], action=action)


# ── ExecutionResult dataclass ─────────────────────────────────────────────────


class TestExecutionResult:
    def test_default_optional_fields(self):
        r = ExecutionResult(
            instruction="pick up milk",
            action=None,
            preconditions=[],
            success=True,
        )
        assert r.error is None
        assert r.clarification is None
        assert r.arm_capacity_error is None
        assert r.skipped is False

    def test_success_false(self):
        r = ExecutionResult(
            instruction="x",
            action=None,
            preconditions=[],
            success=False,
            error=RuntimeError("oops"),
        )
        assert r.success is False
        assert r.error is not None


# ── ExecutionLoop.reset_state ─────────────────────────────────────────────────


class TestResetState:
    def test_reinitialises_exec_state(self):
        loop = _make_loop()
        from pycram.datastructures.enums import Arms

        loop._exec_state.last_pickup_arm = Arms.LEFT
        loop.reset_state()
        assert loop._exec_state.last_pickup_arm is None
        assert loop._exec_state.held_objects == {}


# ── ExecutionLoop.run_single ──────────────────────────────────────────────────


class TestRunSingle:
    def test_success_path(self):
        pipeline = MagicMock()
        action = MagicMock()
        pipeline.run.return_value = action

        loop = _make_loop(pipeline=pipeline)
        plan_result = _make_precondition_result(action)
        loop._planner.compute.return_value = plan_result

        with patch.object(loop, "_execute"):
            result = loop.run_single("pick up milk")

        assert result.success is True
        assert result.action is plan_result.action
        assert result.error is None

    def test_pipeline_raises_clarification_error(self):
        pipeline = MagicMock()
        req = ClarificationRequest(entity_name="milk", entity_role="object")
        pipeline.run.side_effect = ClarificationNeededError(req)

        loop = _make_loop(pipeline=pipeline)
        result = loop.run_single("pick up milk")

        assert result.success is False
        assert result.clarification is req
        assert isinstance(result.error, ClarificationNeededError)

    def test_pipeline_raises_generic_exception(self):
        pipeline = MagicMock()
        pipeline.run.side_effect = RuntimeError("LLM timeout")

        loop = _make_loop(pipeline=pipeline)
        result = loop.run_single("pick up milk")

        assert result.success is False
        assert result.clarification is None
        assert "LLM timeout" in str(result.error)

    def test_precondition_planning_failure(self):
        pipeline = MagicMock()
        action = MagicMock()
        pipeline.run.return_value = action

        loop = _make_loop(pipeline=pipeline)
        loop._planner.compute.side_effect = RuntimeError("planner error")

        result = loop.run_single("pick up milk")

        assert result.success is False
        assert result.action is action
        assert "planner error" in str(result.error)

    def test_execute_failure_no_recovery(self):
        pipeline = MagicMock()
        action = MagicMock()
        pipeline.run.return_value = action

        loop = _make_loop(pipeline=pipeline)
        loop._planner.compute.return_value = _make_precondition_result(action)

        with patch.object(loop, "_execute", side_effect=RuntimeError("execution failed")):
            result = loop.run_single("pick up milk")

        assert result.success is False
        assert "execution failed" in str(result.error)

    def test_state_updated_on_success(self):
        pipeline = MagicMock()
        action = MagicMock()
        pipeline.run.return_value = action

        loop = _make_loop(pipeline=pipeline)
        plan_result = _make_precondition_result(action)
        loop._planner.compute.return_value = plan_result

        with patch.object(loop, "_execute"):
            loop.run_single("pick up milk")

        loop._planner.update_state.assert_called_once_with(plan_result.action, loop._exec_state)


# ── ExecutionLoop.run ─────────────────────────────────────────────────────────


class TestRun:
    def test_single_instruction_success(self):
        pipeline = MagicMock()
        action = MagicMock()
        pipeline.run.return_value = action

        loop = _make_loop(pipeline=pipeline)
        plan_result = _make_precondition_result(action)
        loop._planner.compute.return_value = plan_result

        with patch.object(loop, "_execute"):
            results = loop.run(["pick up milk"])

        assert len(results) == 1
        assert results[0].success is True

    def test_clarification_stops_planning(self):
        pipeline = MagicMock()
        req = ClarificationRequest(entity_name="milk", entity_role="object")
        pipeline.run.side_effect = ClarificationNeededError(req)

        loop = _make_loop(pipeline=pipeline)
        results = loop.run(["pick up milk"])

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].clarification is req

    def test_generic_pipeline_failure(self):
        pipeline = MagicMock()
        pipeline.run.side_effect = RuntimeError("network error")

        loop = _make_loop(pipeline=pipeline)
        results = loop.run(["pick up milk"])

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].clarification is None

    def test_empty_instructions_returns_empty_list(self):
        loop = _make_loop()
        with patch.object(loop, "_execute"):
            results = loop.run([])
        assert results == []

    def test_decomposer_expands_compound_instruction(self):
        """When a decomposer is present it splits one instruction into sub-steps."""
        from llmr.task_decomposer import DecomposedPlan

        decomposer = MagicMock()
        decomposer.decompose.return_value = DecomposedPlan(
            steps=["pick up milk", "place milk on counter"],
            dependencies={1: [0]},
        )

        pipeline = MagicMock()
        action = MagicMock()
        pipeline.run.return_value = action

        loop = _make_loop(pipeline=pipeline, decomposer=decomposer)
        loop._planner.compute.return_value = _make_precondition_result(action)

        with patch.object(loop, "_execute"):
            results = loop.run(["pick up milk and place it on counter"])

        assert len(results) == 2
        decomposer.decompose.assert_called_once()

    def test_failed_step_returns_early_without_dependent_steps(self):
        """When step 0 fails with an exception the loop returns immediately.

        The plan loop exits early on any exception, so only the failed step
        appears in the results — later dependent steps are never processed.
        """
        from llmr.task_decomposer import DecomposedPlan

        decomposer = MagicMock()
        decomposer.decompose.return_value = DecomposedPlan(
            steps=["pick up milk", "place milk on counter"],
            dependencies={1: [0]},
        )

        pipeline = MagicMock()
        pipeline.run.side_effect = RuntimeError("grounding failed")

        loop = _make_loop(pipeline=pipeline, decomposer=decomposer)

        results = loop.run(["pick up milk and place it on counter"])

        # Only the failed step is reported; the dependent step is never reached
        assert len(results) == 1
        assert results[0].success is False
        assert "grounding failed" in str(results[0].error)


# ── Recovery path ─────────────────────────────────────────────────────────────


class TestRunSingleRecovery:
    def test_recovery_success_on_second_attempt(self):
        from llmr.recovery_handler import RecoveryAttemptResult

        pipeline = MagicMock()
        action = MagicMock()
        pipeline.run.return_value = action

        recovery_handler = MagicMock()
        recovery_handler.max_retries = 1

        new_action = MagicMock()
        new_preconditions = [MagicMock()]
        recovery_handler.attempt_recovery.return_value = RecoveryAttemptResult(
            success=True,
            action=new_action,
            preconditions=new_preconditions,
            recovery_schema=MagicMock(),
            attempt_number=1,
        )

        loop = _make_loop(pipeline=pipeline, recovery_handler=recovery_handler)
        loop._planner.compute.return_value = _make_precondition_result(action)

        call_count = {"n": 0}

        def fake_execute(actions):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("ik failed")
            # Second attempt succeeds

        with patch.object(loop, "_execute", side_effect=fake_execute):
            result = loop.run_single("pick up milk")

        assert result.success is True
        assert result.action is new_action

    def test_recovery_abort_returns_failure(self):
        from llmr.recovery_handler import RecoveryAttemptResult

        pipeline = MagicMock()
        action = MagicMock()
        pipeline.run.return_value = action

        recovery_handler = MagicMock()
        recovery_handler.max_retries = 1
        recovery_handler.attempt_recovery.return_value = RecoveryAttemptResult(
            success=False,
            action=None,
            preconditions=[],
            recovery_schema=MagicMock(),
            attempt_number=1,
            error=RuntimeError("aborted"),
        )

        loop = _make_loop(pipeline=pipeline, recovery_handler=recovery_handler)
        loop._planner.compute.return_value = _make_precondition_result(action)

        with patch.object(loop, "_execute", side_effect=RuntimeError("exec failed")):
            result = loop.run_single("pick up milk")

        assert result.success is False
