"""
Shared Pydantic schemas for entity descriptions.

EntityDescriptionSchema is used by the LLM slot-filler to describe a world entity
before it is resolved to a Symbol instance by the grounder.
"""
from __future__ import annotations

from typing_extensions import Dict, Optional

from pydantic import BaseModel


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
