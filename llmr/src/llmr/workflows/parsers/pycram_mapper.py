"""PyCRAM action designator mapping agent for the llmr workflow."""

from __future__ import annotations

import logging

import requests
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import END, START, StateGraph

logger = logging.getLogger(__name__)

from ..llm_configuration import gpt_llm_small, default_llm, ollama_llm_large
from ..prompts.pycram_prompts import (
    designator_prompt,
    entity_mapping_prompt,
    updated_model_selector_prompt,
)
from ..models.pycram_models import (
    ActionNames,
    Actions,
    GroundedCramPlans,
    NavigateActionDescription,
    ParkArmsActionDescription,
    PickUpActionDescription,
    PlaceActionDescription,
    TransportActionDescription,
)
from ..states.all_states import PyCramGroundingState

pycram_graph_memory = MemorySaver()

# ── Structured LLM clients ─────────────────────────────────────────────────────

_default_llm = default_llm
_ollama_action_names_llm = ollama_llm_large.with_structured_output(
    ActionNames, method="json_schema"
)
_openai_action_names_llm = gpt_llm_small.with_structured_output(
    ActionNames, method="json_schema"
)
_ollama_actions_llm = ollama_llm_large.with_structured_output(
    Actions, method="json_schema"
)
_openai_actions_llm = _default_llm.with_structured_output(
    Actions, method="function_calling"
)

# ── Belief-state context overview (module-level constant) ──────────────────────

_BELIEF_STATE_OVERVIEW: str = """
You are a World State Analyzer for a robotic simulation system that uses Semantic Digital Twin (SDT) as its belief state representation. Your role is to interpret, analyze, and explain the state of the simulated world based on the data structures and semantic information provided to you.

## OVERVIEW OF SEMANTIC DIGITAL TWIN (SDT)

The Semantic Digital Twin represents the world as a semantic scene graph combined with a kinematic tree, all wrapped by a central `World` object. This representation goes beyond simple meshes and geometry—it's a rich, structured model of entities, their relationships, and semantic meaning.

## CORE CONCEPTS YOU MUST UNDERSTAND

### 1. KINEMATIC TREE STRUCTURE
The world is organized as a tree of interconnected entities with the following building blocks:

**Bodies (`Body`)**
- Atomic physical entities in the world
- Can carry visual geometry (what you see) and collision geometry (for physics)
- Have properties including:
  - `name`: A `PrefixedName` with both a `name` and `prefix` field
  - `index`: Numerical identifier in the world
  - `collision_config`: Configuration for collision checking behavior
  - `temp_collision_config`: Temporary collision configuration (can be None)
- Examples: robot links, furniture parts, objects, containers

**Connections (`Connection`)**
- Link bodies together via 4x4 transformation matrices
- Define the kinematic relationships between bodies
- Three main types:
  - **Fixed**: Rigid connections with no degrees of freedom (e.g., a table leg to tabletop)
  - **Active**: Controllable degrees of freedom (e.g., robot joints, motorized mechanisms)
  - **Passive**: Uncontrolled degrees of freedom (e.g., floating base for localization, free-moving objects)

**Degrees of Freedom (DoFs) (`DegreeOfFreedom`)**
- Explicit variables representing motion capabilities
- Have limits and derivatives: position, velocity, acceleration, jerk
- Can be shared across connections for coupled kinematics (e.g., parallel gripper fingers)
- Enable differentiable forward kinematics computation

**Transforms**
- 4x4 transformation matrices defining spatial relationships
- Forward kinematics computed by traversing the kinematic tree and multiplying transforms
- Designed to be differentiable using CasADi symbolic expressions

### 2. REGIONS AS SEMANTIC SPATIAL AREAS
**Regions (`Region`)**
- First-class entities representing semantic areas rather than physical bodies
- Live in the same kinematic tree as bodies
- Can be connected to and move with bodies
- Examples: table support surfaces, container openings, reachable zones, task areas
- Purpose: Define meaningful spatial locations for reasoning and planning

### 3. SEMANTIC ANNOTATIONS
- Python classes that add meaning to geometry and kinematics
- Layer actionable concepts on top of raw data
- Examples: "this body is a handle", "these bodies form a drawer", "this is a graspable object"
- Enable transition from geometry -> semantic understanding
- Queryable via Entity Query Language (EQL)

### 4. WORLD REASONING
**WorldReasoner**
- Uses Ripple Down Rules for classification and inference
- Infers world concepts and attributes (e.g., semantic annotation types)
- Rules are versioned and migrate with the codebase
- Enables high-level queries like "the handle attached to the drawer that is currently accessible"

### 5. DATA STRUCTURES YOU'LL ENCOUNTER

**PrefixedName**
- Structure: `PrefixedName(name='object_name', prefix='namespace')`
- Provides hierarchical naming: `prefix.name`
- Enables organization and disambiguation (e.g., multiple robots, environments)

**CollisionCheckingConfig**
- Defines collision behavior for bodies
- Properties:
  - `buffer_zone_distance`: Safety margin around body (can be None)
  - `violated_distance`: Threshold for collision violation (typically 0.0)
  - `disabled`: Whether collision checking is disabled (can be None)
  - `max_avoided_bodies`: Maximum number of bodies to avoid simultaneously
"""


# ── Private helpers ────────────────────────────────────────────────────────────

def _fetch_belief_state_data() -> tuple[dict, dict]:
    """Fetch kinematic nodes and semantic annotations from the belief state server."""
    kinematic_nodes = requests.get("http://127.0.0.1:5001/ks_nodes_all").json()
    semantic_annotations = requests.get("http://127.0.0.1:5001/semantic_annotations").json()
    return kinematic_nodes, semantic_annotations


def _build_belief_state_context(kinematic_nodes: dict, semantic_annotations: dict) -> str:
    """Combine the overview with live belief state data into a single context string."""
    return (
        _BELIEF_STATE_OVERVIEW
        + "\nBelow is the information about the current belief state.\n"
        + f"Kinematic Structure Entities present in the world:\n{kinematic_nodes}\n"
        + f"Semantic Annotations of Entities present in the world:\n{semantic_annotations}\n"
    )


# ── Graph nodes ────────────────────────────────────────────────────────────────

def belief_state_context_node(state: PyCramGroundingState) -> dict:
    """Fetch live belief state, ground entities, and build enriched CRAM plans."""
    kinematic_nodes, semantic_annotations = _fetch_belief_state_data()
    belief_state_context_str = _build_belief_state_context(kinematic_nodes, semantic_annotations)

    atomic_instructions = state["atomics"]
    cram_plans = state["cram_plans"]

    entity_mapping_llm = _default_llm.with_structured_output(
        GroundedCramPlans, method="json_schema"
    )
    chain = entity_mapping_prompt | entity_mapping_llm
    response = chain.invoke(
        {
            "belief_state_context": belief_state_context_str,
            "atomic_instructions": atomic_instructions,
            "cram_plans": cram_plans,
        }
    )
    return {
        "belief_state_context": belief_state_context_str,
        "grounded_cram_plans": response.grounded_plans,
    }


def action_name_selector_node(state: PyCramGroundingState) -> dict:
    """Select the appropriate PyCRAM action model names for the given instructions."""
    instructions = state["atomics"]
    cram_plans = state["grounded_cram_plans"]
    belief_state_context = state["belief_state_context"]

    cram_plans_text = (
        "\n".join(cram_plans) if isinstance(cram_plans, list) else str(cram_plans)
    )

    symbolic_context = (
        "\nBelow is the available knowledge that should be taken as factual knowledge.\n\n"
        "## PARSED SYMBOLIC INFORMATION ##\n\n"
        f"User instructions: {instructions}\n"
        f"cram_plan_response: {cram_plans_text}\n"
    )

    full_context = belief_state_context + symbolic_context

    chain = updated_model_selector_prompt | _openai_action_names_llm
    response = chain.invoke({"symbolic_context": full_context})

    return {"context": full_context, "action_names": response.model_names}


def pycram_designator_node(state: PyCramGroundingState) -> dict:
    """Instantiate PyCRAM action designators and send them to the robot runner."""
    instructions = state["atomics"]
    action_names = state["action_names"]
    cram_plans = state["grounded_cram_plans"]

    chain = designator_prompt | _openai_actions_llm
    response = chain.invoke(
        {
            "atomic_instructions": instructions,
            "grounded_cram_plans": cram_plans,
            "pycram_actions": action_names,
        }
    )

    serialized_response = response.model_dump()
    payload = {
        "model_names": [model.__class__.__name__ for model in response.models],
        "models": serialized_response["models"],
    }

    requests.post("http://127.0.0.1:5001/runner", json=payload)
    return {}


# ── Graph assembly ─────────────────────────────────────────────────────────────

pycram_mapper_builder = StateGraph(PyCramGroundingState)
pycram_mapper_builder.add_node("context_maker", belief_state_context_node)
pycram_mapper_builder.add_node("action_name_selector", action_name_selector_node)
pycram_mapper_builder.add_node("pycram_designator", pycram_designator_node)

pycram_mapper_builder.add_edge(START, "context_maker")
pycram_mapper_builder.add_edge("context_maker", "action_name_selector")
pycram_mapper_builder.add_edge("action_name_selector", "pycram_designator")
pycram_mapper_builder.add_edge("pycram_designator", END)

pycram_mapper_graph = pycram_mapper_builder.compile()


def pycram_mapper_node(state: dict) -> dict:
    """LangGraph node: run the full PyCRAM grounding pipeline.

    Invokes the inner graph and surfaces the selected action names.
    """
    instruction: str = state["instruction"]
    intents: dict = state["intents"]
    cram_plan_response = state["cram_plan_response"]

    final_grounding_state = pycram_mapper_graph.invoke(
        {
            "instruction": instruction,
            "intents": intents,
            "cram_plan_response": cram_plan_response,
        }
    )

    action_names = final_grounding_state["action_names"]
    logger.debug("Action names selected: %s", action_names)
    return {"pycram_action_names": action_names}


if __name__ == "__main__":
    res = _default_llm.invoke("capital of paris")
    print(res)
    print(type(res))
    print(res.content)
