
from __future__ import annotations

import logging
from typing_extensions import Any, Dict, Literal, Optional, Union

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from llmr.workflows.llm_configuration import default_llm
from llmr.workflows.prompts.slot_filler import slot_filler_prompt
from llmr.workflows.schemas.common import EntityDescriptionSchema
from llmr.workflows.schemas.pick_up import GraspParamsSchema, PickUpSlotSchema
from llmr.workflows.schemas.place import PlaceSlotSchema
from llmr.workflows.states.all_states import SlotFillingState

logger = logging.getLogger(__name__)


# ── Public type alias ──────────────────────────────────────────────────────────

ActionSlotSchema = Union[PickUpSlotSchema, PlaceSlotSchema]
"""The return type of ``run_slot_filler``.

Callers can use ``isinstance(schema, PickUpSlotSchema)`` or check
``schema.action_type`` to branch on the action type.
"""


# ── Private intermediate schema ────────────────────────────────────────────────


class _SlotFillerOutput(BaseModel):
    """Private schema used only inside this node for the LLM structured output call.

    The LLM fills this flat schema in one call; we then project it onto the
    correct typed per-action schema.  This class is never exposed outside the
    node — all public functions return ``PickUpSlotSchema`` or ``PlaceSlotSchema``.
    """

    action_type: Literal["PickUpAction", "PlaceAction"] = Field(
        description="The action type this instruction maps to."
    )
    object_description: EntityDescriptionSchema = Field(
        description="The primary object: to pick up (PickUpAction) or to place (PlaceAction)."
    )
    arm: Optional[Literal["LEFT", "RIGHT", "BOTH"]] = Field(
        default=None,
        description="Which arm to use.  Null unless the instruction explicitly names one.",
    )
    # PickUpAction only
    grasp_params: Optional[GraspParamsSchema] = Field(
        default=None,
        description="Grasp configuration.  Only for PickUpAction; null for PlaceAction.",
    )
    # PlaceAction only
    target_description: Optional[EntityDescriptionSchema] = Field(
        default=None,
        description="Target placement location.  Only for PlaceAction; null for PickUpAction.",
    )


def _to_typed_schema(raw: _SlotFillerOutput) -> ActionSlotSchema:
    """Project the intermediate LLM output onto the correct per-action schema."""
    if raw.action_type == "PickUpAction":
        return PickUpSlotSchema(
            object_description=raw.object_description,
            arm=raw.arm,
            grasp_params=raw.grasp_params,
        )
    elif raw.action_type == "PlaceAction":
        return PlaceSlotSchema(
            object_description=raw.object_description,
            target_description=raw.target_description,
            arm=raw.arm,
        )
    else:  # future-proof guard
        raise ValueError(f"Unrecognised action_type from LLM: {raw.action_type!r}")


# ── LLM binding (lazy) ────────────────────────────────────────────────────────

_slot_filler_llm: Optional[Any] = None


def _get_slot_filler_llm() -> Any:
    """Return the slot-filler structured-output LLM, creating it on first call."""
    global _slot_filler_llm
    if _slot_filler_llm is None:
        _slot_filler_llm = default_llm.with_structured_output(
            _SlotFillerOutput, method="function_calling"
        )
    return _slot_filler_llm


# ── LangGraph node ────────────────────────────────────────────────────────────


def slot_filler_node(state: SlotFillingState) -> Dict[str, Any]:
    """LangGraph node: fills ``_SlotFillerOutput`` then projects to typed schema."""
    instruction: str = state["instruction"]
    world_context: str = state.get("world_context", "")

    chain = slot_filler_prompt | _get_slot_filler_llm()
    try:
        raw: _SlotFillerOutput = chain.invoke(
            {"instruction": instruction, "world_context": world_context}
        )
        typed = _to_typed_schema(raw)
        logger.debug(
            "slot_filler_node – action_type=%s, object=%s, semantic_type=%s",
            typed.action_type,
            typed.object_description.name,
            typed.object_description.semantic_type,
        )
        return {"slot_schema": typed.model_dump(), "error": None}
    except Exception as exc:  # noqa: BLE001
        logger.error("slot_filler_node error: %s", exc, exc_info=True)
        return {"slot_schema": None, "error": str(exc)}


# ── LangGraph graph ───────────────────────────────────────────────────────────

_builder: StateGraph = StateGraph(SlotFillingState)
_builder.add_node("slot_filler", slot_filler_node)
_builder.add_edge(START, "slot_filler")
_builder.add_edge("slot_filler", END)

slot_filler_graph = _builder.compile()


# ── Public entry point ────────────────────────────────────────────────────────


def run_slot_filler(
    instruction: str,
    world_context: str = "",
) -> Optional[ActionSlotSchema]:
    """Run the slot-filler node and return a typed per-action schema.

    :param instruction: Natural language robot instruction.
    :param world_context: Optional serialised world state string for context.
    :return: ``PickUpSlotSchema`` or ``PlaceSlotSchema`` on success; ``None`` on failure.
    """
    final_state = slot_filler_graph.invoke(
        {"instruction": instruction, "world_context": world_context}
    )

    if final_state.get("error") or final_state.get("slot_schema") is None:
        logger.warning("run_slot_filler: %s", final_state.get("error"))
        return None

    raw_dict: dict = final_state["slot_schema"]
    action_type = raw_dict.get("action_type")

    if action_type == "PickUpAction":
        return PickUpSlotSchema.model_validate(raw_dict)
    elif action_type == "PlaceAction":
        return PlaceSlotSchema.model_validate(raw_dict)
    else:
        logger.error("run_slot_filler: unexpected action_type %r in state.", action_type)
        return None
