"""Pydantic models for intent and entity parsing."""

from __future__ import annotations

from enum import Enum
from typing_extensions import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class IntentType(str, Enum):
    POUR = "Pouring"
    CUT = "Cutting"
    PICK = "PickingUp"
    PLACE = "Placing"
    OPEN = "Opening"
    CLOSE = "Closing"
    PULL = "Pulling"
    STIR = "Stirring"
    MIX = "Mixing"
    COOL = "Cooling"


class StatusType(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PriorityType(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Roles(BaseModel):
    agent: Optional[str] = Field(None, description="The actor performing the action")
    patient: Optional[str] = Field(None, description="Primary object being acted upon")
    instrument: Optional[str] = Field(None, description="Tool used to perform the action")
    source_location: Optional[str] = Field(None, description="Origin location")
    destination_location: Optional[str] = Field(None, description="Target location")
    beneficiary: Optional[str] = Field(None, description="Who benefits from the action")


class SpatialInfo(BaseModel):
    position: Optional[str] = Field(None, description="Relative position")
    orientation: Optional[str] = Field(None, description="Object orientation")
    coordinates: Optional[Dict[str, float]] = Field(None, description="Absolute coordinates if available")


class Parameters(BaseModel):
    amount: Optional[str] = Field(None, description="Quantity with units")
    attribute: Optional[List[str]] = Field(None, description="Object qualities")
    manner: Optional[str] = Field(None, description="Execution style")
    time: Optional[str] = Field(None, description="Scheduled time")
    duration: Optional[str] = Field(None, description="Action duration")
    priority: Optional[PriorityType] = Field(PriorityType.MEDIUM, description="Task importance")
    force: Optional[str] = Field(None, description="Applied force level")
    speed: Optional[str] = Field(None, description="Execution speed")
    spatial_info: Optional[SpatialInfo] = Field(None, description="Spatial relationship details")


class Conditions(BaseModel):
    preconditions: Optional[List[str]] = Field(None, description="Must be true before execution")
    postconditions: Optional[List[str]] = Field(None, description="Expected state after execution")
    constraints: Optional[List[str]] = Field(None, description="Execution constraints")


class Metadata(BaseModel):
    status: StatusType = Field(StatusType.PENDING, description="Current execution status")
    confidence: Optional[float] = Field(None, description="Parser confidence score (0.0-1.0)")
    comments: Optional[str] = Field(None, description="Human-readable notes")
    safety_constraints: Optional[List[str]] = Field(None, description="Safety requirements")
    estimated_duration: Optional[str] = Field(None, description="Predicted execution time")
    dependencies: Optional[List[str]] = Field(None, description="Other action IDs this depends on")
    alternatives: Optional[List[str]] = Field(None, description="Alternative actions if this fails")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            if v > 1.0:
                v = v / 100.0
            v = max(0.0, min(1.0, v))
        return v


class Instruction(BaseModel):
    action_id: str = Field(..., description="Unique identifier")
    atomic_instruction: Optional[str] = Field(None, description="Co-reference resolved atomic action")
    intent: IntentType = Field(..., description="Primary action type")
    roles: Roles = Field(..., description="Semantic roles and participants")
    parameters: Optional[Parameters] = Field(None, description="Execution parameters")
    conditions: Optional[Conditions] = Field(None, description="Logical conditions")
    metadata: Optional[Metadata] = Field(None, description="Execution metadata")

    @model_validator(mode="after")
    def validate_roles_by_intent(self) -> "Instruction":
        """Ensure required roles are present for each intent type."""
        intent_requirements: Dict[IntentType, List[str]] = {
            IntentType.POUR: ["patient", "destination_location"],
            IntentType.CUT: ["patient"],
            IntentType.PICK: ["patient"],
            IntentType.PLACE: ["patient", "destination_location"],
            IntentType.OPEN: ["patient"],
            IntentType.PULL: ["patient"],
        }
        requirements = intent_requirements.get(self.intent)
        if requirements:
            missing = [f for f in requirements if getattr(self.roles, f) is None]
            if missing:
                raise ValueError(f"Intent '{self.intent}' requires role(s): {missing}")
        return self


class InstructionList(BaseModel):
    instructions: List[Instruction]
