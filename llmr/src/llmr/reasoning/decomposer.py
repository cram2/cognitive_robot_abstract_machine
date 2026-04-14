"""TaskDecomposer — splits compound NL instructions into atomic robot steps with object-flow dependency tracking."""
from __future__ import annotations

import typing
from dataclasses import dataclass, field
from typing_extensions import Dict, List

from pydantic import BaseModel

if typing.TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


@dataclass
class DecomposedPlan:
    """Result of decomposing a compound NL instruction into atomic steps."""

    steps: List[str]
    """
    Ordered list of atomic instruction strings.
    Each maps to exactly one PyCRAM action.
    """

    dependencies: Dict[int, List[int]] = field(default_factory=dict)
    """
    Object-flow dependency graph.
    Key: step index (0-based).
    Value: list of step indices that MUST complete before this step.
    Only includes steps that actually have dependencies.
    """


# --------------------------------------------------------------------------- #
# Internal LLM schemas                                                         #
# --------------------------------------------------------------------------- #

class _AtomicStep(BaseModel):
    """LLM structured-output schema for one atomic step and its object-flow dependencies."""
    instruction: str
    dependencies: List[int]


class _DecomposedInstructions(BaseModel):
    """LLM structured-output schema for the full decomposed plan."""
    steps: List[_AtomicStep]


# --------------------------------------------------------------------------- #
# System prompt                                                                 #
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = """\
You are a robot task decomposer. Split the user's compound instruction into
minimal atomic robot steps. Each step must correspond to exactly ONE action verb.

Rules:
STEP COUNT:
  - Number of output steps must equal the number of action verbs in the instruction.
  - Never invent steps that are not stated or strongly implied.

DECOMPOSITION:
  - Each step must be a complete, self-contained sentence.
  - Preserve exact object names from the original instruction.
  - Replace pronouns ("it", "them", "that") with the specific object name.
  - Do NOT merge two actions into one step.

DEPENDENCIES (object-flow only):
  - Add a dependency from step A to step B only if the OUTPUT object of A is
    the INPUT object of B (e.g. pick up X → place X).
  - Do NOT add dependencies based purely on instruction order.
  - Steps that are spatially or logically independent have no dependencies.

SOURCE vs TARGET:
  - "from / off / out of" → source (where object currently is)
  - "on / onto / into / to" → target (where object should go)

Return structured JSON matching the schema.
"""


# --------------------------------------------------------------------------- #
# TaskDecomposer                                                               #
# --------------------------------------------------------------------------- #

class TaskDecomposer:
    """
    Splits a (potentially compound) NL instruction into atomic steps with
    object-flow dependency tracking.

    The LLM is injected — no global singletons, no hardcoded provider.

    Example::

        decomposer = TaskDecomposer(llm=my_llm)
        plan = decomposer.decompose("go to the table and pick up the milk")
        # plan.steps == ["navigate to the table", "pick up the milk"]
        # plan.dependencies == {}  (no object-flow dependency between them)
    """

    def __init__(self, llm: "BaseChatModel") -> None:
        """
        :param llm: Injected LangChain BaseChatModel — no global singletons.
        """
        self._llm = llm

    def decompose(self, instruction: str) -> DecomposedPlan:
        """
        Decompose a compound instruction into atomic steps.

        Falls back gracefully: if the LLM fails (API error, schema mismatch),
        the original instruction is returned as a single-step plan so execution
        is never blocked.

        :param instruction: The compound (or single) NL instruction.
        :returns: DecomposedPlan with steps and dependency graph.
        """
        structured_llm = self._llm.with_structured_output(_DecomposedInstructions)
        try:
            result: _DecomposedInstructions = structured_llm.invoke([
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
            ])
            steps = self._dedup(result.steps)
            deps = self._build_deps(steps)
            return DecomposedPlan(
                steps=[s.instruction for s in steps],
                dependencies=deps,
            )
        except Exception:
            # Graceful fallback: treat the whole instruction as one step
            return DecomposedPlan(steps=[instruction], dependencies={})

    # ---------------------------------------------------------------------- #
    # Private helpers                                                          #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _dedup(steps: List[_AtomicStep]) -> List[_AtomicStep]:
        """Remove exact-duplicate instructions; preserves order."""
        seen: List[str] = []
        deduped: List[_AtomicStep] = []
        for step in steps:
            if step.instruction not in seen:
                seen.append(step.instruction)
                deduped.append(step)
        return deduped

    @staticmethod
    def _build_deps(steps: List[_AtomicStep]) -> Dict[int, List[int]]:
        """
        Build a validated dependency graph.
        Clamps out-of-range indices and removes self-references.
        """
        n = len(steps)
        return {
            i: [d for d in step.dependencies if 0 <= d < n and d != i]
            for i, step in enumerate(steps)
            if any(0 <= d < n and d != i for d in step.dependencies)
        }
