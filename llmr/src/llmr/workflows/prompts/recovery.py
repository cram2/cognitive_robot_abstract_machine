"""Prompt template for the recovery resolver node.

The LLM receives:
  - A world context snapshot (object poses, semantic annotations, robot arm state)
  - The original natural-language instruction
  - A description of the action that was attempted (type + parameters)
  - The error message from the failed execution

It must diagnose the failure and either produce a revised instruction for a
full replan or decide the task is unrecoverable (ABORT).

Design rules:
1. Prefer REPLAN_FULL when the failure is clearly caused by a wrong parameter
   choice (wrong arm, wrong approach direction, unreachable pose, etc.).
2. Choose ABORT only when the task is fundamentally impossible given the current
   world state (object does not exist, surface is inaccessible, etc.).
3. The revised instruction must be a complete, self-contained NL instruction —
   not a diff or a correction.  It will be fed directly into the ActionPipeline.
"""

from langchain_core.prompts import ChatPromptTemplate

_RECOVERY_RESOLVER_SYSTEM = """\
You are a robot recovery planner.  A robot attempted to execute an action but it
failed.  Your job is to diagnose the failure and decide the best recovery strategy.

## RECOVERY STRATEGIES

REPLAN_FULL
  - Use this when the failure is caused by a recoverable parameter choice.
  - Common causes: wrong arm selected, wrong approach direction, pose unreachable
    from the chosen angle, gripper orientation mismatch, navigation target blocked.
  - Provide a complete revised natural-language instruction that explicitly states
    the corrected parameters (e.g. "Pick up the milk with the LEFT arm from the
    BACK side").  The instruction will be re-processed by the full planning pipeline.

ABORT
  - Use this only when recovery is not possible given the current world state.
  - Common causes: the target object no longer exists, the target surface is
    completely inaccessible, a required precondition cannot be satisfied.
  - Do NOT use ABORT simply because one parameter failed — try REPLAN_FULL first.

## REASONING STYLE
Think step-by-step:
1. What action was attempted and what parameters were chosen?
2. What does the error message indicate about why it failed?
3. Is the failure caused by a recoverable parameter choice or a fundamental obstacle?
4. If recoverable, what specific parameter changes would avoid the failure?
5. Formulate a revised instruction that encodes the corrected parameters explicitly.

Then produce the structured output.
  - ``failure_diagnosis``: 1-2 sentences diagnosing the root cause.
  - ``reasoning``: 1-2 sentences justifying the chosen strategy.
  - ``revised_instruction``: only when strategy is REPLAN_FULL; null otherwise.
"""

_RECOVERY_RESOLVER_HUMAN = """\
## WORLD CONTEXT
{world_context}

## ORIGINAL INSTRUCTION
{original_instruction}

## FAILED ACTION
{failed_action_description}

## ERROR MESSAGE
{error_message}

Diagnose the failure and return RecoverySchema.
"""

recovery_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _RECOVERY_RESOLVER_SYSTEM),
        ("human", _RECOVERY_RESOLVER_HUMAN),
    ]
)
