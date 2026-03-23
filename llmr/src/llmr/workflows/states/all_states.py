"""LangGraph state type definitions for the generative backend.

Mirrors the structure of ``llmr.workflows.states.all_states`` with state classes
for Phase 1 (slot filling) and Phase 2 (discrete resolution).

Each state is a TypedDict-style MessagesState so that LangGraph can correctly
track message history alongside the domain-specific fields.
"""

from __future__ import annotations

from typing_extensions import Any, Dict, List, Optional

from langgraph.graph import MessagesState


class SlotFillingState(MessagesState):
    """State for the Phase 1 slot-filling pipeline.

    Travels from the entry point through the slot-filler node and into the
    EntityGrounder.  After grounding, ``grounded_bodies`` holds the resolved
    ``Body`` indices (serialised as ints) so the state remains serialisable.
    """

    # ── Inputs ────────────────────────────────────────────────────────────────
    instruction: str
    """Raw natural-language instruction from the user."""

    world_context: str
    """Serialised world snapshot (kinematic nodes + semantic annotations) used
    as optional reference context for the LLM."""

    # ── Intermediate / outputs ────────────────────────────────────────────────
    slot_schema: Optional[Dict[str, Any]]
    """Serialised PickUpSlotSchema produced by the slot-filler LLM node.
    Stored as a plain dict so it remains JSON-serialisable inside LangGraph."""

    grounded_body_indices: Optional[List[int]]
    """World-body indices resolved by EntityGrounder from object_description.
    A list because the description may match multiple candidate bodies."""

    error: Optional[str]
    """Non-fatal error or warning message from any node (e.g. no body found)."""


class DiscreteResolutionState(MessagesState):
    """State for the Phase 2 discrete-resolution pipeline.

    The slot_schema (with possibly-null discrete fields) flows in; the
    resolved_schema (all discrete fields filled) flows out.
    """

    # ── Inputs ────────────────────────────────────────────────────────────────
    world_context: str
    """Serialised world snapshot for the resolution LLM."""

    known_parameters: str
    """Human-readable summary of parameters already known from Phase 1."""

    parameters_to_resolve: str
    """Human-readable list of parameters that are still null."""

    # ── Output ────────────────────────────────────────────────────────────────
    resolved_schema: Optional[Dict[str, Any]]
    """Serialised discrete resolution schema produced by the resolver LLM
    (e.g. ``PickUpDiscreteResolutionSchema`` or ``PlaceDiscreteResolutionSchema``
    depending on the action type being resolved)."""

    error: Optional[str]


class RecoveryState(MessagesState):
    """State for the recovery resolution pipeline.

    Flows through the recovery resolver node when an action execution has
    failed.  The node receives the failure context and produces a
    ``RecoverySchema`` describing whether to replan or abort.
    """

    # ── Inputs ────────────────────────────────────────────────────────────────
    world_context: str
    """Serialised world snapshot at the time of failure."""

    original_instruction: str
    """The NL instruction that was being executed when the failure occurred."""

    failed_action_description: str
    """Human-readable description of the action that failed (type + parameters)."""

    error_message: str
    """The exception message from the failed execution."""

    # ── Output ────────────────────────────────────────────────────────────────
    resolved_schema: Optional[Dict[str, Any]]
    """Serialised ``RecoverySchema`` produced by the recovery resolver LLM."""

    error: Optional[str]
    """Non-fatal error from the LLM node itself (distinct from the robot error)."""
