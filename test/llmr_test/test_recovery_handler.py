from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llmr.recovery_handler import RecoveryAttemptResult, RecoveryHandler, _serialise_failed_action
from llmr.workflows.schemas.recovery import RecoverySchema


# ── _serialise_failed_action ─────────────────────────────────────────────────


class TestSerialiseFailedAction:
    def test_none_returns_no_action_message(self):
        result = _serialise_failed_action(None)
        assert "No action was produced" in result

    def test_pydantic_model_dump(self):
        action = MagicMock()
        action.__class__.__name__ = "PickUpAction"
        action.model_dump.return_value = {"arm": "LEFT", "object_designator": "milk_body"}
        result = _serialise_failed_action(action)
        assert "PickUpAction" in result
        assert "arm" in result
        assert "LEFT" in result

    def test_dict_fallback_when_no_model_dump(self):
        class _FakeAction:
            def __init__(self):
                self.arm = "RIGHT"
                self._private = "hidden"

        result = _serialise_failed_action(_FakeAction())
        assert "_FakeAction" in result
        assert "arm" in result
        assert "_private" not in result

    def test_type_name_always_present(self):
        action = MagicMock()
        action.__class__.__name__ = "PlaceAction"
        action.model_dump.side_effect = Exception("dump failed")
        # Also make vars() fail to exercise last-resort path
        with patch("llmr.recovery_handler.vars", side_effect=TypeError("no vars")):
            result = _serialise_failed_action(action)
        assert "PlaceAction" in result

    def test_model_dump_excludes_none_fields(self):
        action = MagicMock()
        action.__class__.__name__ = "PickUpAction"
        action.model_dump.return_value = {"arm": "LEFT"}
        result = _serialise_failed_action(action)
        assert "LEFT" in result


# ── RecoveryHandler.attempt_recovery ─────────────────────────────────────────


def _make_handler(max_retries: int = 2) -> RecoveryHandler:
    return RecoveryHandler(world=MagicMock(), max_retries=max_retries)


def _make_replan_schema(revised_instruction: str | None = "pick up cup with left arm"):
    return RecoverySchema(
        recovery_strategy="REPLAN_FULL",
        revised_instruction=revised_instruction,
        failure_diagnosis="Arm collision detected.",
        reasoning="Switch to left arm.",
    )


def _make_abort_schema():
    return RecoverySchema(
        recovery_strategy="ABORT",
        failure_diagnosis="Object not reachable.",
        reasoning="No recovery possible.",
    )


class TestAttemptRecovery:
    def test_llm_returns_none_gives_failure(self):
        handler = _make_handler()
        with patch("llmr.recovery_handler.run_recovery_resolver", return_value=None):
            result = handler.attempt_recovery(
                instruction="pick up milk",
                failed_action=None,
                error=RuntimeError("ik"),
                exec_state=MagicMock(),
                pipeline=MagicMock(),
                planner=MagicMock(),
            )
        assert result.success is False
        assert result.error is not None
        assert result.recovery_schema is None

    def test_abort_strategy_returns_failure(self):
        handler = _make_handler()
        with patch(
            "llmr.recovery_handler.run_recovery_resolver",
            return_value=_make_abort_schema(),
        ):
            result = handler.attempt_recovery(
                instruction="pick up milk",
                failed_action=None,
                error=RuntimeError("ik"),
                exec_state=MagicMock(),
                pipeline=MagicMock(),
                planner=MagicMock(),
            )
        assert result.success is False
        assert "aborted" in str(result.error).lower() or "recovery" in str(result.error).lower()

    def test_replan_full_empty_revised_instruction_is_failure(self):
        handler = _make_handler()
        schema = _make_replan_schema(revised_instruction="")
        with patch("llmr.recovery_handler.run_recovery_resolver", return_value=schema):
            result = handler.attempt_recovery(
                instruction="pick up milk",
                failed_action=None,
                error=RuntimeError("ik"),
                exec_state=MagicMock(),
                pipeline=MagicMock(),
                planner=MagicMock(),
            )
        assert result.success is False

    def test_replan_full_pipeline_success_returns_success(self):
        handler = _make_handler()
        schema = _make_replan_schema()

        pipeline = MagicMock()
        new_action = MagicMock()
        pipeline.run.return_value = new_action

        planner = MagicMock()
        new_preconditions = [MagicMock()]
        planner.compute.return_value = MagicMock(
            action=new_action,
            preconditions=new_preconditions,
        )

        with patch("llmr.recovery_handler.run_recovery_resolver", return_value=schema):
            result = handler.attempt_recovery(
                instruction="pick up milk",
                failed_action=None,
                error=RuntimeError("ik"),
                exec_state=MagicMock(),
                pipeline=pipeline,
                planner=planner,
            )

        assert result.success is True
        assert result.action is new_action
        assert result.preconditions is new_preconditions
        assert result.recovery_schema is schema

    def test_replan_full_pipeline_failure_returns_failure(self):
        handler = _make_handler()
        schema = _make_replan_schema()

        pipeline = MagicMock()
        pipeline.run.side_effect = RuntimeError("pipeline down")

        with patch("llmr.recovery_handler.run_recovery_resolver", return_value=schema):
            result = handler.attempt_recovery(
                instruction="pick up milk",
                failed_action=None,
                error=RuntimeError("ik"),
                exec_state=MagicMock(),
                pipeline=pipeline,
                planner=MagicMock(),
            )

        assert result.success is False
        assert "pipeline down" in str(result.error)

    def test_replan_full_planner_failure_returns_new_action(self):
        handler = _make_handler()
        schema = _make_replan_schema()

        pipeline = MagicMock()
        new_action = MagicMock()
        pipeline.run.return_value = new_action

        planner = MagicMock()
        planner.compute.side_effect = RuntimeError("planner error")

        with patch("llmr.recovery_handler.run_recovery_resolver", return_value=schema):
            result = handler.attempt_recovery(
                instruction="pick up milk",
                failed_action=None,
                error=RuntimeError("ik"),
                exec_state=MagicMock(),
                pipeline=pipeline,
                planner=planner,
            )

        assert result.success is False
        assert result.action is new_action  # action set even on planner failure

    def test_attempt_number_preserved(self):
        handler = _make_handler()
        with patch("llmr.recovery_handler.run_recovery_resolver", return_value=None):
            result = handler.attempt_recovery(
                instruction="x",
                failed_action=None,
                error=RuntimeError("e"),
                exec_state=MagicMock(),
                pipeline=MagicMock(),
                planner=MagicMock(),
                attempt_number=3,
            )
        assert result.attempt_number == 3

    def test_serialise_world_called_with_exec_state(self):
        handler = _make_handler()
        exec_state = MagicMock()
        with patch("llmr.recovery_handler.run_recovery_resolver", return_value=None):
            with patch(
                "llmr.recovery_handler._serialise_world_for_llm",
                return_value="ctx",
            ) as mock_ser:
                handler.attempt_recovery(
                    instruction="x",
                    failed_action=None,
                    error=RuntimeError("e"),
                    exec_state=exec_state,
                    pipeline=MagicMock(),
                    planner=MagicMock(),
                )
        mock_ser.assert_called_once_with(handler.world, exec_state)
