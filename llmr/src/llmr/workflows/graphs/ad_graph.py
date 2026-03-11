"""Action Decomposition (AD) LangGraph workflow."""

from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph.state import END, START, StateGraph
from pymongo import MongoClient
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

# Suppress Pydantic's serialization warning that fires from LangChain's
# structured-output internals when `parsed` receives an InstructionList
# instead of None (LangChain internal typing limitation, not a user bug).
warnings.filterwarnings(
    "ignore",
    message=r".*PydanticSerializationUnexpectedValue.*field_name='parsed'.*",
    category=UserWarning,
)

from ..parsers.intent_entity import ReflectiveParser
from ..lg_memory.memory.long_term_memory_extractor import MemoryExtractor
from ..lg_memory.memory.long_term_memory_store import LongTermMemoryStore
from ..lg_memory.memory.long_term_memory_retriever import MemoryRetriever
from ..lg_memory.memory.semantic_cache import MessageSemanticCache
from ..lg_memory.utils.embeddings import EmbeddingHelper, NoOpEmbeddingHelper
from ..llm_configuration import default_llm
from ..prompts.cram_gen_prompts import (
    cram_plan_prompt,
    field_prompt,
    field_props_prompt,
)
from ..models.cram_gen_models import (
    Closing,
    Cooling,
    Cutting,
    Mixing,
    Opening,
    PickingUp,
    Placing,
    Pouring,
    Pulling,
    Stirring,
)
from ..states.all_states import ActionDecompState, MainPipelineState
from ..utils import read_json_from_file, think_remover

load_dotenv(Path(__file__).parent.parent / ".env")

# ── MongoDB setup ──────────────────────────────────────────────────────────────

MONGO_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
print("MONGO_URI: ", MONGO_URI, "\n")
_mongo_client = MongoClient(MONGO_URI)

memory_db = MongoDBSaver(
    _mongo_client,
    db_name="langgraph_memory",
    checkpoint_collection_name="cram_checkpoints",
    allowed_msgpack_modules=[
        ("llmr.workflows.models.intent_entity_models", "IntentType"),
        ("llmr.workflows.models.intent_entity_models", "StatusType"),
        ("llmr.workflows.models.intent_entity_models", "PriorityType"),
    ],
)

# ── Long-term memory ───────────────────────────────────────────────────────────

long_term_store = LongTermMemoryStore(_mongo_client)
semantic_cache = MessageSemanticCache(_mongo_client, score_threshold=0.92)
embedder = EmbeddingHelper() if os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") else NoOpEmbeddingHelper()
memory_retriever = MemoryRetriever(long_term_store, embedder)
memory_extractor = MemoryExtractor(long_term_store, embedder)

# ── Action classes lookup ──────────────────────────────────────────────────────

_ACTION_CLASSES: list[type] = [
    Closing,
    Cooling,
    Cutting,
    Mixing,
    Opening,
    PickingUp,
    Placing,
    Pouring,
    Pulling,
    Stirring,
]

_default_llm = default_llm

# ── CRAM resource data ─────────────────────────────────────────────────────────

_json_path: Path = Path(__file__).parent.parent / "resources" / "cram_action_cores.json"
_json_data: dict = read_json_from_file(_json_path)


# ── Private helpers ────────────────────────────────────────────────────────────

def _extract_action_core_attributes(
    instruction: str, action_core: str, required_fields: list
) -> str:
    """Call the field extraction LLM chain and strip think-tags."""
    chain = field_prompt | _default_llm
    response = chain.invoke(
        {
            "instruction": instruction,
            "action_core": action_core,
            "target_attributes": required_fields,
        }
    )
    return think_remover(response.content)


def _enrich_attributes(action_core_attributes: str, context: str) -> str:
    """Semantically enrich extracted attributes with property dictionaries."""
    chain = field_props_prompt | _default_llm
    response = chain.invoke(
        {"action_core_attributes": action_core_attributes, "context": context}
    )
    return think_remover(response.content)


def _generate_cram_plan(
    instruction: str, cram_plan_syntax: str, enriched_attributes: str
) -> str:
    """Fill the CRAM plan template with enriched attribute data."""
    chain = cram_plan_prompt | _default_llm
    response = chain.invoke(
        {
            "user_instruction": instruction,
            "cram_plan_syntax": cram_plan_syntax,
            "enriched_json_data": enriched_attributes,
        }
    )
    return think_remover(response.content)


def _resolve_action_class(action_core: str) -> type:
    """Return the Pydantic class matching *action_core*, or raise ValueError."""
    action_class = next(
        (
            c
            for c in _ACTION_CLASSES
            if c.__name__ == action_core or getattr(c, "action_core", None) == action_core
        ),
        None,
    )
    if action_class is None:
        raise ValueError(f"Unknown action_core: {action_core}")
    return action_class


# ── Memory nodes ───────────────────────────────────────────────────────────────

def load_user_context_node(state: ActionDecompState) -> dict:
    """Load relevant memories before processing the instruction."""
    logger.info("Loading user context")
    user_id = state.get("user_id", "default_user")
    instruction = state["instruction"]

    memory_context = memory_retriever.format_for_prompt(
        user_id=user_id,
        current_message=instruction,
        tags=["instruction", "preference", "context"],
    )

    logger.debug("Memory context: %s", memory_context)
    return {"context": memory_context or state.get("context", "")}


def save_learnings_node(state: ActionDecompState) -> dict:
    """Extract and save useful information from the current session."""
    logger.info("Saving learnings")
    user_id = state.get("user_id", "default_user")
    instruction = state["instruction"]

    conversation_messages = [
        HumanMessage(content=instruction),
        AIMessage(content=f"Action cores: {state.get('action_core', [])}"),
    ]

    memory_extractor.extract_and_save(
        messages=conversation_messages,
        user_id=user_id,
        source_thread_id=state.get("thread_id", ""),
    )
    return {}


# ── Main nodes ─────────────────────────────────────────────────────────────────

def intent_node(state: ActionDecompState) -> dict:
    """Parse natural language instruction into symbolic intents."""
    parser = ReflectiveParser(model_name="gpt")
    result = parser.parse(state["instruction"])
    logger.debug("Parsed intents: %s", result)
    return {"intents": result}


def updated_cram_node(state: ActionDecompState) -> dict:
    """Generate CRAM plans for each parsed intent."""
    instruction = state["instruction"]
    intents = state["intents"]
    context = state.get("context", "")

    action_cores: list[str] = []
    action_attributes: list[str] = []
    enriched_attributes: list[str] = []
    cram_plans: list[str] = []

    for intent in intents["instructions"]:
        action_core: str = intent["intent"].value
        required_fields = _json_data[action_core]["action_roles"]
        cram_plan_syntax = _json_data[action_core]["cram_plan"]

        _resolve_action_class(action_core)  # validates action_core exists

        raw_attributes = _extract_action_core_attributes(
            instruction, action_core, required_fields
        )
        enriched = _enrich_attributes(raw_attributes, context)
        cram_plan = _generate_cram_plan(instruction, cram_plan_syntax, enriched)

        action_cores.append(action_core)
        action_attributes.append(raw_attributes)
        enriched_attributes.append(enriched)
        cram_plans.append(cram_plan)

    return {
        "action_core": action_cores,
        "action_core_attributes": action_attributes,
        "enriched_action_core_attributes": enriched_attributes,
        "cram_plan_response": cram_plans,
    }


# ── Caching helper ─────────────────────────────────────────────────────────────

def run_with_cache(
    instruction: str,
    user_id: str = "default_user",
    thread_id: str = "default_thread",
    use_cache: bool = True,
) -> dict:
    """Run the AD graph with optional semantic cache (hit → return cached, miss → run + store)."""
    if use_cache:
        cached_result = semantic_cache.lookup(instruction)
        if cached_result:
            logger.info("Cache hit — returning cached CRAM plan")
            return json.loads(cached_result)

    logger.info("Cache miss — running graph" if use_cache else "Cache disabled — running graph")
    invoke_config = {"configurable": {"thread_id": thread_id}}
    result = action_decomp_graph.invoke(
        {
            "instruction": instruction,
            "user_id": user_id,
            "thread_id": thread_id,
            "context": "",
        },
        config=invoke_config,
    )

    if use_cache:
        cache_payload = json.dumps(
            {
                "instruction": result.get("instruction", ""),
                "action_core": result.get("action_core", []),
                "action_core_attributes": result.get("action_core_attributes", []),
                "enriched_action_core_attributes": result.get("enriched_action_core_attributes", []),
                "cram_plan_response": result.get("cram_plan_response", []),
                "intents": result.get("intents", {}),
            }
        )
        semantic_cache.save(instruction, cache_payload)
    return result


# ── Graph assembly ─────────────────────────────────────────────────────────────

action_decomp_graph_builder = StateGraph(ActionDecompState)
action_decomp_graph_builder.add_node("load_user_context", load_user_context_node)
action_decomp_graph_builder.add_node("intent_node", intent_node)
action_decomp_graph_builder.add_node("updated_cram_node", updated_cram_node)
action_decomp_graph_builder.add_node("save_learnings", save_learnings_node)

action_decomp_graph_builder.add_edge(START, "load_user_context")
action_decomp_graph_builder.add_edge("load_user_context", "intent_node")
action_decomp_graph_builder.add_edge("intent_node", "updated_cram_node")
action_decomp_graph_builder.add_edge("updated_cram_node", "save_learnings")
action_decomp_graph_builder.add_edge("save_learnings", END)

action_decomp_graph = action_decomp_graph_builder.compile(checkpointer=memory_db)


def action_decomp_agent_node(state: MainPipelineState) -> dict:
    """LangGraph node: run the full AD pipeline and surface results to parent graph."""
    instruction = state["instruction"]

    final_internal_state = action_decomp_graph.invoke({"instruction": instruction})

    return {
        "intents": final_internal_state["intents"],
        "action_core": final_internal_state["action_core"],
        "action_core_attributes": final_internal_state["action_core_attributes"],
        "enriched_action_core_attributes": final_internal_state["enriched_action_core_attributes"],
        "cram_plan_response": final_internal_state["cram_plan_response"],
    }


if __name__ == "__main__":
    invoke_config = {"configurable": {"thread_id": 1}}
    result = action_decomp_graph.invoke(
        {"instruction": "cut the apple with a knife and place the pieces in the bowl"},
        config=invoke_config,
    )
    graph_state = action_decomp_graph.get_state(config=invoke_config)
    print("*" * 10)
    for key, value in graph_state.values.items():
        print(key, ":", value)
