"""Pydantic models for CRAM action generation."""

from typing_extensions import List, Optional

from pydantic import BaseModel, Field


class Closing(BaseModel):
    """A Pydantic model for the 'Closing' action."""

    obj_to_close: str
    action_verb: str
    utensil: Optional[str] = None


class Cooling(BaseModel):
    """A Pydantic model for the 'Cooling' action."""

    action_verb: str
    amount: Optional[float] = None
    location: Optional[str] = None
    obj_to_be_cooled: Optional[str] = None
    unit: Optional[str] = None


class Cutting(BaseModel):
    """A Pydantic model for the 'Cutting' action."""

    obj_to_be_cut: str
    utensil: str
    action_verb: str
    amount: Optional[float] = None
    unit: Optional[str] = None
    cram_plan: Optional[str] = None


class Mixing(BaseModel):
    """A Pydantic model for the 'Mixing' action."""

    content: List[str]
    action_verb: str


class Opening(BaseModel):
    """A Pydantic model for the 'Opening' action."""

    obj_to_be_opened: str
    action_verb: str


class PickingUp(BaseModel):
    """A Pydantic model for the 'PickingUp' action."""

    obj_to_be_grabbed: str
    action_verb: str
    location: Optional[str] = None
    cram_plan: Optional[str] = None


class Placing(BaseModel):
    """A Pydantic model for the 'Placing' action."""

    obj_to_be_put: str
    action_verb: str
    location: Optional[str] = None
    cram_plan: Optional[str] = None


class Pouring(BaseModel):
    """A Pydantic model for the 'Pouring' action."""

    stuff: str = Field(description="entity being poured")
    source: str = Field(description="container from which the substance is poured")
    goal: str = Field(description="container/location to which the substance is poured")
    action_verb: str = Field(description="verb representing the action, e.g., 'pour'")
    unit: Optional[str] = Field(description="Units (liters, drops, ounces etc.,)", default=None)
    amount: Optional[float] = Field(description="Amount of quantity to pour", default=None)
    cram_plan: Optional[str] = None


class Pulling(BaseModel):
    """A Pydantic model for the 'Pulling' action."""

    obj_to_be_pulled: str
    action_verb: str
    cram_plan: Optional[str] = None


class Stirring(BaseModel):
    """A Pydantic model for the 'Stirring' action."""

    action_verb: str
    content: List[str]
    cram_plan: Optional[str] = None
