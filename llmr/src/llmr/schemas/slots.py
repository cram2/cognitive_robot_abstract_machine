"""
Pydantic schemas for the LLM slot-filling output.

Single generic ActionReasoningOutput — no per-action-type subclassing.
The LLM resolves whatever free slots the Match expression declares.

SlotValue carries either:
  - entity_description (for ENTITY/POSE slots) — used by the grounder
  - value string (for ENUM, PRIMITIVE, COMPLEX sub-field slots)
"""
from __future__ import annotations

from typing_extensions import List, Optional

from pydantic import BaseModel, Field

from llmr.schemas.entities import EntityDescriptionSchema


class SlotValue(BaseModel):
    """A single resolved slot produced by the LLM reasoning step."""

    field_name: str
    """
    Name of the Match field being resolved.
    For complex sub-fields use dotted notation: 'grasp_description.grasp_type'.
    Must match an attribute name (or sub-attribute) on the action class.
    """

    value: Optional[str] = None
    """
    Resolved concrete value as a string.
    - ENUM / parameter slots: the enum member name (e.g. 'LEFT', 'FRONT').
    - ENTITY slots: the world body display name — kept for fallback grounding
      if entity_description is absent.
    - COMPLEX sub-field slots (dotted names): the resolved sub-field value.
    Null when entity_description fully captures the resolution.
    """

    entity_description: Optional[EntityDescriptionSchema] = None
    """
    For ENTITY and POSE slots: the LLM's semantic description of the entity.
    The grounder uses name + semantic_type + spatial_context + attributes to
    find the matching Symbol instance in SymbolGraph.
    Required for ENTITY/POSE slots; null for parameter/enum/primitive slots.
    """

    reasoning: str = ""
    """Per-slot explanation of why this value was chosen."""


class ActionReasoningOutput(BaseModel):
    """
    Structured output from the LLM slot-filling step inside LLMBackend._evaluate().

    Generic across all action types — no per-action subclassing needed.
    One SlotValue per free slot (top-level and complex sub-fields combined).
    """

    action_type: str
    """The action class name being resolved (echoed back for traceability)."""

    slots: List[SlotValue]
    """
    One entry per free slot in the Match expression.
    Complex fields are represented as multiple dotted entries, e.g.:
      SlotValue(field_name='grasp_description.grasp_type', value='TOP')
      SlotValue(field_name='grasp_description.approach_direction', value='FRONT')
    Entity sub-fields inside complex fields may also appear as dotted entries
    with entity_description populated.
    """

    overall_reasoning: str = ""
    """High-level explanation of the resolution strategy."""


class ActionClassification(BaseModel):
    """Output of the action classification step used by nl_plan() factory."""

    action_type: str
    """
    Exact Python class name of the chosen action.
    E.g. 'PickUpAction', 'NavigateAction', 'PlaceAction'.
    Must match a key in the action registry.
    """

    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    """LLM self-reported confidence. Informational only."""

    reasoning: str = ""
    """Why this action type was chosen."""
