"""Shared Pydantic schemas used across all action types.

Types defined here are cross-action and must not live inside any single
action's schema file.  All action-specific schema modules import from here.
"""

from __future__ import annotations

from typing_extensions import Dict, Optional

from pydantic import BaseModel, Field


class EntityDescriptionSchema(BaseModel):
    """Semantic description of an object entity extracted from NL.

    This is pre-grounding – it captures what the LLM *understood* about the
    object.  The ``EntityGrounder`` converts this into a concrete ``Body``.
    """

    name: str = Field(
        description="The object name or noun phrase as mentioned in the instruction "
        "(e.g. 'cup', 'red mug', 'milk bottle')."
    )
    semantic_type: Optional[str] = Field(
        default=None,
        description="Ontological/semantic type hint if inferrable from context "
        "(e.g. 'Artifact', 'Container', 'FoodItem').  Null if unknown.",
    )
    spatial_context: Optional[str] = Field(
        default=None,
        description="Spatial relationship that can narrow down candidates "
        "(e.g. 'on the table', 'inside the fridge', 'to the left').  "
        "Null if not mentioned.",
    )
    attributes: Optional[Dict[str, str]] = Field(
        default=None,
        description="Additional discriminating attributes from the instruction "
        "(e.g. {'color': 'red', 'size': 'small'}).  Null if none.",
    )
