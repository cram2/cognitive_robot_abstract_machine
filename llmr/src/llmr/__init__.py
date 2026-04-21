"""
llmr — LLM-powered GenerativeBackend for KRROOD.

KRROOD-integrated package that implements LLMBackend(GenerativeBackend), allowing
PyCRAM underspecified action Match expressions to be resolved via LLM reasoning.

Package-independence guarantees
---------------------------------
- No direct world-package imports anywhere in this package.
- PyCRAM imports are isolated to llmr.pycram_bridge.adapter.
- World context is derived from SymbolGraph (krrood), not from a world object.
- Symbol subclasses, including robot components, are grounded from SymbolGraph.

Dependency direction: llmr → krrood (one-way, no circular imports).

Package layout
--------------
  backend.py              LLMBackend — the GenerativeBackend implementation
  factory.py              nl_plan() / nl_sequential() / resolve_params() — user-facing entry points
  exceptions.py           typed llmr exceptions
  schemas.py              EntityDescriptionSchema, SlotValue, ActionReasoningOutput, ActionClassification
  bridge/                 single gateway to krrood — all krrood calls funnel here
    introspect.py         PycramIntrospector, FieldKind, ActionSchema, FieldSpec
    world_reader.py       SymbolGraph read + serialize_world_from_symbol_graph
    match_reader.py       Match snapshot: MatchData, MatchSlot, read_match, required_match
  pycram_bridge/
    adapter.py            PyCRAM execution and action-discovery boundary
  reasoning/
    slot_filler.py        run_slot_filler(), classify_action() — LLM prompt pipeline
    decomposer.py         TaskDecomposer — compound NL → atomic steps
    llm_config.py         make_llm(), LLMProvider — LLM factory
  resolution/
    grounder.py           EntityGrounder — description → Symbol instance
    slot_resolution.py    LLM slot output coercion and grounding dispatch

Quickstart — simple (fully NL-driven)
---------------------------------------
::

    from llmr import nl_plan, nl_sequential
    from llmr.reasoning.llm_config import make_llm, LLMProvider
    from your_world_package import Body  # caller's groundable type

    llm = make_llm(LLMProvider.OPENAI, model="gpt-4o")

    plan = nl_plan(
        "pick up the milk from the table",
        context=context,
        llm=llm,
        groundable_type=Body,
    )
    plan.perform()

    for plan in nl_sequential(
        "go to the table, pick up the milk and put it in the fridge",
        context=context,
        llm=llm,
        groundable_type=Body,
    ):
        plan.perform()

Quickstart — power user (action type known, LLM fills free slots)
-------------------------------------------------------------------
::

    from krrood.entity_query_language.query.match import Match
    from llmr import resolve_match, resolve_params
    from your_action_package import PickUpAction
    from your_action_package import GraspDescription

    match = Match(PickUpAction)(
        object_designator=...,
        arm=...,
        grasp_description=Match(GraspDescription)(
            approach_direction=...,
            vertical_alignment=...,
            manipulator=...,
        ),
    )
    action = resolve_params(
        match,
        llm=llm,
        instruction="pick up the milk from the table",
        groundable_type=Body,
    )
    plan = resolve_match(match, context=context, llm=llm, groundable_type=Body)
    plan.perform()
"""

from llmr.backend import LLMBackend
from llmr.exceptions import (
    LLMActionClassificationFailed,
    LLMActionRegistryEmpty,
    LLMProviderNotSupported,
    LLMSlotFillingFailed,
    LLMUnresolvedRequiredFields,
)
from llmr.factory import nl_plan, nl_sequential, resolve_match, resolve_params

__all__ = [
    "LLMBackend",
    "nl_plan",
    "nl_sequential",
    "resolve_match",
    "resolve_params",
    "LLMActionClassificationFailed",
    "LLMActionRegistryEmpty",
    "LLMProviderNotSupported",
    "LLMSlotFillingFailed",
    "LLMUnresolvedRequiredFields",
]
