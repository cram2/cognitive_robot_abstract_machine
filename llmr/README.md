# llmr — LLM-based Reasoner

A pipeline that converts natural language instructions into executable PyCRAM robot actions using LLM-driven reasoning and LangGraph workflows.

## Workflow

```
Natural Language Instruction
        ↓
   NL → structured intent (IntentType + roles)
        ↓
   Action Decomposition Graph   (graphs/ad_graph.py)
   1. Field Extraction    — extract semantic roles from instruction
   2. Semantic Enrichment — add object properties (size, material, etc.)
   3. CRAM Plan Generation — fill LISP-style plan template
        ↓
   CRAM Plan String
   e.g. (an action (type PickingUp) (object (:tag cup (an object (type Artifact)...))))
        ↓
   CRAMToPyCRAMSerializer    (adapters/cram_to_pycram.py)
   Parse S-expression → CRAMActionPlan (intermediate representation)
        ↓
   SimulationBridge          (adapters/simulation_bridge.py)
   Resolve symbolic names → live world Body objects → PartialDesignator
        ↓
   PyCRAM Execution
```


## Structure

```
src/llmr/
├── parsers/        LLM agent nodes (intent parser, pycram mapper)
├── graphs/         LangGraph workflows (ad_graph)
├── models/         Pydantic schemas (intents, CRAM actions, PyCRAM models)
├── prompts/        LangChain prompt templates
├── adapters/    CRAM → PyCRAM conversion and simulation bridge
├── states/         LangGraph state definitions
├── lg_memory/      Long-term memory and semantic cache (MongoDB)
└── workflows/      LLM configuration and utilities

tests/              Unit tests (no LLM calls — all mocked)
examples/           example usages
```

## Installation

```bash
workon cram-env
pip install -e .
```

Requires a `OPENAI_API_KEY` and running MongoDB instance (only for cache) `MONGODB_URI` set in `src/llmr/workflows/.env`.

## Quick Start

```python
from llmr.workflows.graphs.ad_graph import run_with_cache
from llmr.adapters.simulation_bridge import SimulationBridge

result = run_with_cache("pick up the cup from the table", use_cache=False)
cram_plans = result["cram_plan_response"]

bridge = SimulationBridge(world, robot)
bridge.execute_batch(cram_plans, arm=Arms.RIGHT)
```

## Running Tests

```bash
workon cram-env
pytest tests/
```
