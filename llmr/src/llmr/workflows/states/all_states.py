"""State type definitions for the llmr workflow graphs."""

from typing_extensions import List

from langgraph.graph import MessagesState

from ..models.intent_entity_models import InstructionList


class ActionDecompState(MessagesState):
    """State for the Action Decomposition (AD) pipeline."""

    instruction: str
    action_type: str
    action_core: List[str]
    action_core_attributes: List[str]
    enriched_action_core_attributes: List[str]
    cram_plan_response: List[str]
    context: str
    intents: InstructionList
    user_id: str
    thread_id: str


class MainPipelineState(MessagesState):
    """Combined state for the top-level pipeline (AD)."""

    instruction: str
    action_type: str
    action_core: List[str]
    action_core_attributes: List[str]
    enriched_action_core_attributes: List[str]
    cram_plan_response: List[str]
    intents: InstructionList
    context: str


class PyCramGroundingState(MessagesState):
    """State for the PyCRAM grounding pipeline."""

    atomics: str
    cram_plans: str
    belief_state_context: str
    context: str
    grounded_cram_plans: List[str]
    action_names: List[str]
    designator_models: str


__all__ = [
    "ActionDecompState",
    "MainPipelineState",
    "PyCramGroundingState",
]
