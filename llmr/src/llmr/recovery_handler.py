
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing_extensions import TYPE_CHECKING, List, Optional, Union

from semantic_digital_twin.world import World

from pycram.datastructures.partial_designator import PartialDesignator
from pycram.robot_plans.actions.base import ActionDescription

from llmr.pipeline.action_pipeline import ActionPipeline, _serialise_world_for_llm  # type: ignore[attr-defined]
from llmr.planning.motion_precondition_planner import ExecutionState  # noqa: F401 — re-exported

if TYPE_CHECKING:
    from llmr.planning.motion_precondition_planner import MotionPreconditionPlanner
from llmr.workflows.nodes.recovery_resolver import run_recovery_resolver
from llmr.workflows.schemas.recovery import RecoverySchema

logger = logging.getLogger(__name__)


# ── Result dataclass ─────────────────────────────────────────────────────────


@dataclass
class RecoveryAttemptResult:
    """Outcome of a single recovery attempt."""

    success: bool
    action: Optional[ActionDescription]
    preconditions: List[ActionDescription]
    recovery_schema: Optional[RecoverySchema]
    attempt_number: int
    error: Optional[Exception] = None


# ── Action serialiser ────────────────────────────────────────────────────────


def _serialise_failed_action(
    action: Optional[Union[ActionDescription, PartialDesignator]],
) -> str:
    """Produce a human-readable description of a failed action for the LLM.

    Tries to extract the action's type name and public field values.  Falls
    back gracefully at each level so the recovery resolver always receives
    *something* meaningful.

    :param action: The action that failed, or ``None`` if planning never
        produced one (e.g. the pipeline itself failed).
    :return: Multi-line human-readable string.
    """
    if action is None:
        return "No action was produced (pipeline failed before action construction)."

    action_type = type(action).__name__
    lines = [f"Action type: {action_type}"]

    # Attempt Pydantic-style dump first (richest representation)
    if hasattr(action, "model_dump"):
        try:
            params = action.model_dump(exclude_none=True)
            for key, val in params.items():
                lines.append(f"  {key}: {val}")
            return "\n".join(lines)
        except Exception:
            pass

    # Fall back to __dict__ / vars()
    try:
        for key, val in vars(action).items():
            if not key.startswith("_"):
                lines.append(f"  {key}: {val}")
        return "\n".join(lines)
    except Exception:
        pass

    # Last resort
    lines.append(f"  (details unavailable: {action!s})")
    return "\n".join(lines)


# ── RecoveryHandler ──────────────────────────────────────────────────────────


@dataclass
class RecoveryHandler:
    """Performs LLM-driven replanning after a failed action execution."""

    world: World
    max_retries: int = field(default=2)

    # ── Public API ───────────────────────────────────────────────────────────

    def attempt_recovery(
        self,
        instruction: str,
        failed_action: Optional[Union[ActionDescription, PartialDesignator]],
        error: Exception,
        exec_state: ExecutionState,
        pipeline: ActionPipeline,
        planner: MotionPreconditionPlanner,
        attempt_number: int = 1,
    ) -> RecoveryAttemptResult:
        """Perform one recovery attempt for a failed action execution.

        :param instruction: Original NL instruction that triggered the failure.
        :param failed_action: The action that was attempted, or ``None`` if the pipeline failed.
        :param error: The exception raised by ``_execute()``.
        :param exec_state: Current execution state (arm occupancy, etc.).
        :param pipeline: The ``ActionPipeline`` from the calling ``ExecutionLoop``.
        :param planner: The ``MotionPreconditionPlanner`` from the calling loop.
        :param attempt_number: Current attempt count (1-based), for logging.
        :return: ``RecoveryAttemptResult`` describing the outcome.
        """
        logger.info(
            "RecoveryHandler: attempt %d for '%s' (error: %s)",
            attempt_number,
            instruction,
            error,
        )

        # ── Step 1: build context strings ────────────────────────────────────
        world_ctx = _serialise_world_for_llm(self.world, exec_state)
        failed_action_desc = _serialise_failed_action(failed_action)
        error_message = f"{type(error).__name__}: {error}"

        # ── Step 2: recovery resolver LLM call ───────────────────────────────
        schema = run_recovery_resolver(
            world_context=world_ctx,
            original_instruction=instruction,
            failed_action_description=failed_action_desc,
            error_message=error_message,
        )

        if schema is None:
            logger.warning(
                "RecoveryHandler: LLM call failed for attempt %d of '%s'.",
                attempt_number,
                instruction,
            )
            return RecoveryAttemptResult(
                success=False,
                action=None,
                preconditions=[],
                recovery_schema=None,
                attempt_number=attempt_number,
                error=RuntimeError("Recovery resolver LLM call returned None."),
            )

        logger.info(
            "RecoveryHandler: strategy=%s  diagnosis=%s",
            schema.recovery_strategy,
            schema.failure_diagnosis,
        )

        # ── Step 3a: ABORT ────────────────────────────────────────────────────
        if schema.recovery_strategy == "ABORT":
            logger.warning(
                "RecoveryHandler: ABORT for '%s'.  Reasoning: %s",
                instruction,
                schema.reasoning,
            )
            return RecoveryAttemptResult(
                success=False,
                action=None,
                preconditions=[],
                recovery_schema=schema,
                attempt_number=attempt_number,
                error=RuntimeError(f"Recovery aborted: {schema.failure_diagnosis}"),
            )

        # ── Step 3b: REPLAN_FULL ──────────────────────────────────────────────
        revised_instruction = schema.revised_instruction
        if not revised_instruction:
            logger.error(
                "RecoveryHandler: REPLAN_FULL but revised_instruction is empty "
                "for '%s'.  Treating as ABORT.",
                instruction,
            )
            return RecoveryAttemptResult(
                success=False,
                action=None,
                preconditions=[],
                recovery_schema=schema,
                attempt_number=attempt_number,
                error=RuntimeError(
                    "Recovery strategy is REPLAN_FULL but revised_instruction is empty."
                ),
            )

        logger.info(
            "RecoveryHandler: replanning '%s' → '%s'",
            instruction,
            revised_instruction,
        )

        try:
            new_action = pipeline.run(revised_instruction, exec_state=exec_state)
        except Exception as pipeline_exc:
            logger.error(
                "RecoveryHandler: pipeline failed for revised instruction '%s': %s",
                revised_instruction,
                pipeline_exc,
            )
            return RecoveryAttemptResult(
                success=False,
                action=None,
                preconditions=[],
                recovery_schema=schema,
                attempt_number=attempt_number,
                error=pipeline_exc,
            )

        try:
            plan_result = planner.compute(new_action, exec_state)
        except Exception as plan_exc:
            logger.error(
                "RecoveryHandler: precondition planning failed for recovered action: %s",
                plan_exc,
            )
            return RecoveryAttemptResult(
                success=False,
                action=new_action,
                preconditions=[],
                recovery_schema=schema,
                attempt_number=attempt_number,
                error=plan_exc,
            )

        logger.info(
            "RecoveryHandler: recovery plan ready — [%s]",
            ", ".join(type(a).__name__ for a in plan_result.preconditions + [plan_result.action]),
        )

        return RecoveryAttemptResult(
            success=True,
            action=plan_result.action,
            preconditions=plan_result.preconditions,
            recovery_schema=schema,
            attempt_number=attempt_number,
        )
