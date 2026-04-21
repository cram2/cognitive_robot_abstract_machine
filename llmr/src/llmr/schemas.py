"""Pydantic schemas for LLM structured output.

Two categories:

  - :class:`EntityDescriptionSchema` — the LLM's pre-grounding description of a
    world entity; the grounder turns it into a :class:`Symbol` instance.
  - :class:`SlotValue`, :class:`ActionReasoningOutput`,
    :class:`ActionClassification` — slot-filling and action-classification
    outputs consumed by :class:`LLMBackend` and the NL factory entry points.
"""

from __future__ import annotations

from typing_extensions import Dict, List, Optional

from pydantic import BaseModel, Field

# ── Entity description (pre-grounding) ────────────────────────────────────────


class EntityDescriptionSchema(BaseModel):
    """
    Semantic description of an entity BEFORE it is resolved to a world object.

    The LLM populates this from the NL instruction. LLMBackend uses it to
    reason about which world body the user is referring to — considering
    name, semantic type, spatial context, and discriminating attributes.

    This is deliberately a description (not a grounded ID) so the LLM can
    apply contextual reasoning rather than pure name matching.
    """

    name: str
    """
    The noun phrase as it appears in the instruction.
    E.g. "milk bottle", "red cup on the left", "the heavy box".
    """

    semantic_type: Optional[str] = None
    """
    Ontological type hint from the instruction or world annotations.
    E.g. "FoodItem", "Container", "SupportSurface".
    Used to narrow candidate bodies by annotation type.
    """

    spatial_context: Optional[str] = None
    """
    Spatial relationship string from the instruction.
    E.g. "on the kitchen counter", "next to the sink", "in the fridge".
    Used for proximity-based disambiguation when multiple candidates exist.
    """

    attributes: Optional[Dict[str, str]] = None
    """
    Discriminating key/value attributes from the instruction.
    E.g. {"color": "red", "size": "large", "material": "glass"}.
    """


# ── Slot-filling output ───────────────────────────────────────────────────────


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
    Nested complex Match leaves are represented as dotted entries, e.g.:
      SlotValue(field_name='grasp_description.grasp_type', value='TOP')
      SlotValue(field_name='grasp_description.approach_direction', value='FRONT')
    Entity sub-fields inside complex fields may also appear as dotted entries
    with entity_description populated.
    """

    overall_reasoning: str = ""
    """High-level explanation of the resolution strategy."""


# ── Action classification ─────────────────────────────────────────────────────


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
