# llmr — LLM-based Reasoner

A pipeline that converts natural language instructions into executable PyCRAM robot actions using LLM-driven slot filling, entity grounding, motion precondition planning, and recovery handling.

## Workflow

```
Natural Language Instruction
        ↓
   TaskDecomposer              (task_decomposer.py)
   Compound instruction → ordered list of atomic steps
        ↓
   ActionPipeline              (pipeline/action_pipeline.py)
   1. Slot Filling     — classify action type and extract parameters via LLM
   2. Entity Grounding — resolve object names to live world Body objects
   3. Action Dispatch  — build typed PyCRAM ActionDescription
        ↓
   MotionPreconditionPlanner   (planning/motion_precondition_planner.py)
   Compute preparatory actions (navigate, park arms, raise torso)
        ↓
   PyCRAM SequentialPlan
   Execute preconditions + action on the robot
        ↓
   RecoveryHandler             (recovery_handler.py)
   On failure: LLM proposes replan or abort
```

## Structure

```
src/llmr/
├── execution_loop.py       Top-level orchestrator (run instructions end-to-end)
├── task_decomposer.py      Compound NL → ordered atomic steps
├── recovery_handler.py     Failure recovery via LLM replanning
├── world_setup.py          Convenience world/robot initialisation helpers
├── pipeline/
│   ├── action_pipeline.py      Slot filling → grounding → dispatch
│   ├── action_dispatcher.py    Build typed PyCRAM actions (PickUp, Place, …)
│   ├── entity_grounder.py      Resolve object names to world Body objects
│   └── clarification.py        Clarification request / arm-capacity errors
├── planning/
│   └── motion_precondition_planner.py  Navigate + posture preconditions
└── workflows/
    ├── llm_configuration.py    LLM provider factory (OpenAI / Ollama)
    ├── nodes/                  LangGraph nodes (slot_filler, resolver, recovery_resolver)
    ├── prompts/                LangChain prompt templates
    ├── schemas/                Pydantic output schemas (PickUp, Place, Recovery)
    └── states/                 LangGraph state definitions

test/llmr_test/             Unit tests (all LLM calls mocked)
```

## Installation

```bash
workon cram-env
pip install -e llmr/
```

Requires `OPENAI_API_KEY` set in `llmr/.env`:

```
OPENAI_API_KEY=your-openai-api-key-here
```

## Quick Start

```python
from llmr import ExecutionLoop, ActionPipeline, TaskDecomposer
from llmr import load_pr2_apartment_world

world, robot = load_pr2_apartment_world()

loop = ExecutionLoop(
    pipeline=ActionPipeline(world=world),
    task_decomposer=TaskDecomposer(),
    world=world,
)

results = loop.run(["pick up the milk from the table and place it on the island"])
```

## Running Tests

```bash
workon cram-env
pytest test/llmr_test/
```
