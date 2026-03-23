"""Pydantic schema for recovery resolution — LLM output when an action execution fails.

This schema is the output contract for the recovery resolver node.  The LLM
receives the original instruction, a description of the failed action, and the
error message, then decides whether to attempt a full replan (with a revised
instruction) or abort.

Recovery strategy values:
    REPLAN_FULL  — LLM provides a ``revised_instruction`` that avoids the
                   failure.  The revised instruction is run through the full
                   ``ActionPipeline`` so no special-casing is needed downstream.
    ABORT        — LLM determines the task is unrecoverable (object unreachable,
                   pose impossible, etc.).  The failure is propagated.
"""

from __future__ import annotations

from typing_extensions import Literal, Optional

from pydantic import BaseModel, Field

__all__ = ["RecoverySchema"]


class RecoverySchema(BaseModel):
    """LLM output for the recovery resolution node.

    Produced by the recovery resolver when an action execution has failed.
    The caller uses ``recovery_strategy`` to decide the next step and
    ``revised_instruction`` (when ``REPLAN_FULL``) to re-enter the standard
    pipeline.
    """

    recovery_strategy: Literal["REPLAN_FULL", "ABORT"] = Field(
        description=(
            "REPLAN_FULL: provide a revised instruction to replan from scratch. "
            "ABORT: the task cannot be recovered; propagate the failure."
        )
    )
    revised_instruction: Optional[str] = Field(
        default=None,
        description=(
            "A rewritten natural-language instruction that avoids the failure. "
            "Required when recovery_strategy is REPLAN_FULL, null otherwise. "
            "Must be specific enough to resolve the original failure cause "
            "(e.g. specify a different arm, approach direction, or target)."
        ),
    )
    failure_diagnosis: str = Field(
        description=(
            "One or two sentences diagnosing why the action failed. "
            "Reference the specific error message and action parameters."
        )
    )
    reasoning: str = Field(
        description=(
            "One or two sentences explaining the chosen recovery strategy and "
            "how the revised instruction (if any) addresses the root cause."
        )
    )
