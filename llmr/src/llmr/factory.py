"""User-facing factory functions for NL-driven plan construction.

All pycram access goes through llmr.pycram_bridge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import Any, Callable, Dict, List, Optional

from krrood.symbol_graph.symbol_graph import Symbol
from llmr.bridge.introspect import PycramIntrospector
from llmr.bridge.match_reader import required_match
from llmr.exceptions import LLMActionClassificationFailed
from llmr.pycram_bridge import PycramContext, PycramPlanNode, execute_single

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


def nl_plan(
    instruction: str,
    context: PycramContext,
    llm: "BaseChatModel",
    groundable_type: type = Symbol,
    action_registry: Optional[Dict[str, type]] = None,
) -> PycramPlanNode:
    """
    Build a single executable PlanNode from a natural-language instruction.

    Internally:
      1. LLM classifies the instruction → action class
      2. Builds an underspecified Match for required schema fields
      3. Sets a strict LLMBackend as context.query_backend
      4. Returns execute_single(match, context)

    :param instruction:    The natural-language instruction.
    :param context:        PyCRAM Context (world + robot + query_backend).
    :param llm:            LangChain-compatible chat model.
    :param groundable_type: Symbol subclass for scene objects.
                           Passed to LLMBackend for entity grounding; never imported here.
    :param action_registry: Optional {class_name: class} dict.
                           Auto-discovered by the PyCRAM bridge if None.
    :returns: A PlanNode ready to be performed.
    :raises ValueError: If the LLM cannot classify the action type.
    """
    from llmr.reasoning.slot_filler import classify_action

    # Step 1: Classify action type from NL instruction
    action_cls = classify_action(
        instruction=instruction,
        llm=llm,
        action_registry=action_registry,
    )
    if action_cls is None:
        raise LLMActionClassificationFailed(instruction=instruction)

    # Step 2: Build an underspecified Match for required schema fields.
    match = required_match(action_cls, PycramIntrospector())

    # Step 3: Set strict LLMBackend on context.
    context.query_backend = _backend(
        llm=llm,
        groundable_type=groundable_type,
        instruction=instruction,
        strict_required=True,
    )

    # Step 4: Return the plan node
    return execute_single(match, context)


def nl_sequential(
    instruction: str,
    context: PycramContext,
    llm: "BaseChatModel",
    groundable_type: type = Symbol,
    action_registry: Optional[Dict[str, type]] = None,
) -> List[PycramPlanNode]:
    """
    Decompose a multi-step NL instruction and return one PlanNode per atomic step.

    Each step gets its own LLMBackend instance with a step-specific instruction,
    so LLM reasoning context is preserved per step.

    :param instruction:    The (possibly multi-step) natural-language instruction.
    :param context:        PyCRAM Context shared across all steps.
    :param llm:            LangChain-compatible chat model.
    :param groundable_type: Symbol subclass for scene objects.
    :param action_registry: Optional action class registry (auto-discovered if None).
    :returns: List of PlanNodes, one per atomic step, in execution order.
    """
    from llmr.reasoning.decomposer import TaskDecomposer

    decomposed = TaskDecomposer(llm=llm).decompose(instruction)
    return [
        nl_plan(
            step,
            context,
            llm,
            groundable_type=groundable_type,
            action_registry=action_registry,
        )
        for step in decomposed.steps
    ]


def resolve_match(
    match: Any,
    context: PycramContext,
    llm: "BaseChatModel",
    groundable_type: type = Symbol,
    instruction: Optional[str] = None,
    world_context_provider: Optional[Callable[[], str]] = None,
    strict_required: bool = False,
) -> PycramPlanNode:
    """Return a PlanNode for an already-built underspecified Match.

    Plan helper for Role 2.  For a non-executing resolved action instance, use
    ``resolve_params``.

    :param match:           A fully or partially underspecified krrood Match expression.
    :param context:         PyCRAM Context (world + robot).
    :param llm:             LangChain-compatible chat model.
    :param groundable_type: Symbol subclass scoping entity grounding. Defaults to Symbol.
    :param instruction:     Optional NL hint included in the slot-filler prompt.
                            Omit when the action type and fixed slots carry the intent.
    :param world_context_provider: Optional callable returning runtime world context text.
    :param strict_required: Raise if required fields remain unresolved before construction.
    :returns: A PlanNode ready to be performed.
    """
    context.query_backend = _backend(
        llm=llm,
        groundable_type=groundable_type,
        instruction=instruction,
        world_context_provider=world_context_provider,
        strict_required=strict_required,
    )
    return execute_single(match, context)


def resolve_params(
    match: Any,
    llm: "BaseChatModel",
    groundable_type: type = Symbol,
    instruction: Optional[str] = None,
    world_context_provider: Optional[Callable[[], str]] = None,
    strict_required: bool = False,
) -> Any:
    """Resolve an underspecified Match and return the concrete action instance.

    Role 2 non-executing API: no action classification, no Match construction, and no
    PyCRAM PlanNode creation.  The supplied Match is still updated by the backend as
    part of normal KRROOD evaluation.
    """
    backend = _backend(
        llm=llm,
        groundable_type=groundable_type,
        instruction=instruction,
        world_context_provider=world_context_provider,
        strict_required=strict_required,
    )
    return next(iter(backend.evaluate(match)))


def _backend(
    llm: "BaseChatModel",
    groundable_type: type,
    instruction: Optional[str],
    strict_required: bool,
    world_context_provider: Optional[Callable[[], str]] = None,
) -> Any:
    """Create the LLM backend used by factory entry points."""
    from llmr.backend import LLMBackend

    return LLMBackend(
        llm=llm,
        groundable_type=groundable_type,
        instruction=instruction,
        world_context_provider=world_context_provider,
        strict_required=strict_required,
    )
