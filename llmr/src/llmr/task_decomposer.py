
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing_extensions import Dict, List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llmr.workflows.llm_configuration import default_llm

logger = logging.getLogger(__name__)


# ── Public result type ──────────────────────────────────────────────────────────


@dataclass
class DecomposedPlan:
    """Result of decomposing a (possibly compound) NL instruction into atomic steps."""

    steps: List[str]
    dependencies: Dict[int, List[int]] = field(default_factory=dict)


# ── LLM output schema ───────────────────────────────────────────────────────────


class _AtomicStep(BaseModel):
    """Single atomic step with its dependency indices."""

    instruction: str = Field(description="The atomic sub-instruction text.")
    dependencies: List[int] = Field(
        default_factory=list,
        description=(
            "0-based indices of OTHER steps in this list that must succeed before "
            "this step can execute.  Base dependencies on OBJECT FLOW only: "
            "if this step uses an object that an earlier step acquires (picks up), "
            "list that earlier step's index.  Do NOT add dependencies based solely "
            "on instruction order.  Steps that operate on different objects with no "
            "shared preconditions must have an empty list."
        ),
    )


class _DecomposedInstructions(BaseModel):
    """LLM output schema for the decomposer."""

    steps: List[_AtomicStep] = Field(
        description=(
            "Ordered list of atomic steps. "
            "Each step maps to exactly one supported action type "
            "(PickUpAction or PlaceAction). "
            "If the original instruction is already atomic, return it as a single-element list."
        )
    )


# ── Prompt ─────────────────────────────────────────────────────────────────────

_DECOMPOSER_SYSTEM = """\
You are a robot task decomposer.

Your job is to split a natural language instruction into a list of ATOMIC steps,
where each step corresponds to EXACTLY ONE of the following supported action types:

  - PickUpAction  (triggered by: pick up, grab, grasp, lift, take, fetch, get, retrieve)
  - PlaceAction   (triggered by: place, put, set down, deposit, lay, put down, drop off)

For each step you must also declare its dependencies: the 0-based indices of OTHER
steps whose successful execution is a precondition for this step.

Dependency rules (IMPORTANT):
- Dependencies are based on OBJECT FLOW, not instruction order.
- A PlaceAction depends on the PickUpAction that acquires the same object.
- A PickUpAction that acquires a DIFFERENT object from other steps is INDEPENDENT
  (empty dependencies) even if it appears between two related steps.
- Do NOT add a dependency just because two steps are adjacent.

Decomposition rules:
- If the instruction is already atomic (single action), return it as-is in a
  single-element list with empty dependencies.
- If the instruction is compound (multiple actions joined by "and", "then",
  "after that", etc.), split it into one entry per action.
- Preserve all object names, target names, and qualifiers VERBATIM from the original.
- Steps must be in execution order (e.g. pick up before placing).
- Replace pronouns ("it", "the object") in later steps with the actual object name
  from the earlier step, so each step is self-contained.
- Do NOT invent new objects, targets, or actions not mentioned in the instruction.
- Do NOT decompose beyond the supported action types — if the instruction contains
  an unsupported action, include it verbatim as a single step with empty dependencies.
- Count the action verbs in the instruction. The number of steps must equal the
  number of distinct action verbs. One verb → one step, always.
- Never repeat the same step. Every step in the list must be unique.
- Never invent implicit prerequisite steps. If the instruction says "place the
  bottle on the shelf", return ONLY that one step — do NOT add a "pick up the
  bottle" step before it. The caller guarantees the robot already holds the object.
  Dependencies express ordering constraints on steps WITHIN the instruction only.

SOURCE vs TARGET location (CRITICAL):
- Prepositions "from", "off", "out of", "off of" describe the SOURCE location
  where an object is picked up.  They are spatial qualifiers for PickUpAction —
  they do NOT imply a separate PlaceAction.
- Prepositions "on", "onto", "on top of", "into", "at", "to" describe the TARGET
  location for PlaceAction.
- "Pick up the milk from the counter" → ONE step: PickUpAction (source=counter is
  just context, NOT a place target).  Do NOT generate a PlaceAction for it.

Examples:

  "grab the bottle and put it on the shelf"
    → steps:
        0: instruction="grab the bottle",               dependencies=[]
        1: instruction="put the bottle on the shelf",   dependencies=[0]

  "fetch the cereal box from the cabinet"
    → steps:
        0: instruction="fetch the cereal box from the cabinet", dependencies=[]
        ← "from the cabinet" is the source, NOT a place target. Single step only.

  "pick up the book from the table and place it on the bookshelf"
    → steps:
        0: instruction="pick up the book from the table",    dependencies=[]
        1: instruction="place the book on the bookshelf",    dependencies=[0]
        ← source ("from the table") stays with step 0, target ("bookshelf") goes to step 1.

  "grab the cup. grab the plate. put the cup on the tray"
    → steps:
        0: instruction="grab the cup",              dependencies=[]
        1: instruction="grab the plate",            dependencies=[]   ← independent
        2: instruction="put the cup on the tray",   dependencies=[0]  ← object flow, not adjacency

  "retrieve the bottle and place it on the dining table and pick up the box"
    → steps:
        0: instruction="retrieve the bottle",                   dependencies=[]
        1: instruction="place the bottle on the dining table",  dependencies=[0]
        2: instruction="pick up the box",                       dependencies=[]   ← independent

  "take the mug off the shelf"
    → steps:
        0: instruction="take the mug off the shelf", dependencies=[]
        ← "off the shelf" is the source. Single step only.
"""

_DECOMPOSER_HUMAN = """\
Instruction: {instruction}

Decompose into atomic steps with dependencies.
"""

_decomposer_prompt = ChatPromptTemplate.from_messages(
    [("system", _DECOMPOSER_SYSTEM), ("human", _DECOMPOSER_HUMAN)]
)

_PICKUP_VERBS: frozenset[str] = frozenset(
    {"pick", "grab", "grasp", "lift", "take", "fetch", "get", "retrieve"}
)
_PLACE_VERBS: frozenset[str] = frozenset({"place", "put", "set", "deposit", "lay", "drop"})


# ── TaskDecomposer ─────────────────────────────────────────────────────────────


class TaskDecomposer:
    """Splits compound NL instructions into atomic sub-instructions with dependencies.

    Each returned step maps to exactly one supported action type (PickUpAction or
    PlaceAction).  The ``dependencies`` dict in the returned :class:`DecomposedPlan`
    encodes object-flow dependencies so that :class:`ExecutionLoop` can skip steps
    whose prerequisites failed without blocking independent steps.
    """

    def __init__(self) -> None:
        self._llm = default_llm.with_structured_output(
            _DecomposedInstructions, method="function_calling"
        )
        self._chain = _decomposer_prompt | self._llm

    def decompose(self, instruction: str) -> DecomposedPlan:
        """Decompose *instruction* into an atomic plan with dependency graph.

        :param instruction: Raw natural language instruction (may be compound).
        :return: :class:`DecomposedPlan` with at least one step.
        """
        try:
            result: _DecomposedInstructions = self._chain.invoke({"instruction": instruction})
            steps = [s.instruction.strip() for s in result.steps if s.instruction.strip()]
            if not steps:
                logger.warning(
                    "Decomposer returned empty list for '%s' – using original.", instruction
                )
                return DecomposedPlan(steps=[instruction])

            # Guard: if the original instruction contains only one action verb and
            # the LLM returned multiple steps, keep only the step whose text best
            # matches the original instruction (longest common prefix heuristic).
            instr_words = set(instruction.lower().split())
            n_pickup = len(instr_words & _PICKUP_VERBS)
            n_place = len(instr_words & _PLACE_VERBS)
            n_verbs = n_pickup + n_place
            if n_verbs == 1 and len(steps) > 1:
                # Find the step that is most similar to the original instruction.
                best = max(
                    range(len(steps)),
                    key=lambda idx: len(
                        set(steps[idx].lower().split()) & set(instruction.lower().split())
                    ),
                )
                logger.warning(
                    "Decomposer returned %d steps for single-verb instruction '%s' "
                    "— keeping best match: '%s'.",
                    len(steps),
                    instruction,
                    steps[best],
                )
                result.steps = [result.steps[best]]
                steps = [steps[best]]

            # Guard: deduplicate identical steps the LLM may hallucinate.
            # Keep only the first occurrence; remap result.steps indices accordingly.
            seen: Dict[str, int] = {}  # step text → first index in deduplicated list
            dedup_indices: List[int] = []  # original index → deduplicated index
            dedup_steps_objs = []
            for orig_i, step_obj in enumerate(result.steps):
                text = step_obj.instruction.strip()
                if not text:
                    dedup_indices.append(-1)
                    continue
                if text not in seen:
                    seen[text] = len(dedup_steps_objs)
                    dedup_steps_objs.append(step_obj)
                else:
                    logger.warning(
                        "Decomposer produced duplicate step '%s' for '%s' — removing.",
                        text,
                        instruction,
                    )
                dedup_indices.append(seen[text])

            steps = [s.instruction.strip() for s in dedup_steps_objs]

            # Build dependency map — only include steps that have at least one dep,
            # and clamp indices to valid range to guard against LLM hallucination.
            n = len(steps)
            dependencies: Dict[int, List[int]] = {}
            for i, step in enumerate(dedup_steps_objs):
                valid_deps = [d for d in step.dependencies if 0 <= d < n and d != i]
                if valid_deps:
                    dependencies[i] = valid_deps

            logger.info(
                "Decomposed '%s' → steps=%s deps=%s",
                instruction,
                steps,
                dependencies,
            )
            return DecomposedPlan(steps=steps, dependencies=dependencies)

        except Exception as exc:
            logger.error(
                "Decomposer failed for '%s': %s – falling back to original.", instruction, exc
            )
            return DecomposedPlan(steps=[instruction])
