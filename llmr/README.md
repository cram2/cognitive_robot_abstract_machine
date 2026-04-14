# llm_reasoner

**LLM-powered GenerativeBackend for KRROOD** — Translate natural language instructions into robot actions using LLMs.

llm_reasoner is a standalone package that implements `LLMBackend(GenerativeBackend)`, allowing PyCRAM underspecified action expressions to be resolved via LLM reasoning. It bridges natural language and structured robot planning.

---

## Quick Start

### Installation

```bash
# From repo root
pip install ./llm_reasoner
```

### Basic Usage: Natural Language → Robot Action

```python
from llm_reasoner import nl_plan, nl_sequential
from llm_reasoner.reasoning.llm_config import make_llm, LLMProvider

# Create an LLM
llm = make_llm(LLMProvider.OPENAI, model="gpt-4o")

# Execute a single instruction
plan = nl_plan(
    "pick up the milk from the table",
    context=context,
    llm=llm,
    groundable_type=Body,  # your world's object type
)
plan.perform()

# Execute multi-step instructions (auto-decomposed)
for plan in nl_sequential(
    "go to the kitchen, pick up the milk, and place it in the fridge",
    context=context,
    llm=llm,
):
    plan.perform()
```

### Power User: LLM Parameter Resolution

```python
from krrood.entity_query_language.query.match import Match
from llm_reasoner import resolve_params, resolve_match
from your_actions import PickUpAction

# Build underspecified action match
match = Match(PickUpAction)(object_designator=..., arm=...)

# Option 1: Get resolved action instance (no execution)
action = resolve_params(
    match,
    llm=llm,
    instruction="pick up the milk",
    groundable_type=Body,
)

# Option 2: Get executable plan (with context)
plan = resolve_match(
    match,
    context=context,
    llm=llm,
    groundable_type=Body,
)
plan.perform()
```

---

## Module Reference

### 📋 Core Modules

#### **`backend.py`** — LLMBackend (GenerativeBackend)
The main executor: resolves free Match slots using LLM reasoning.

**Key Classes:**
- `LLMBackend(GenerativeBackend)` — Main implementation
  - `__init__(llm, groundable_type=Symbol, instruction=None, strict_required=False)`
  - `evaluate(match)` — Yields resolved action instances

**Workflow:**
1. Receives an underspecified Match with free slots (`...`)
2. Introspects action class to get field types and docstrings
3. Calls LLM to fill free slots
4. Grounds entity slots using EntityGrounder
5. Coerces primitive/enum slots to correct types
6. Reconstructs complex nested fields
7. Returns concrete action instance


#### **`factory.py`** — User-Facing Entry Points
High-level functions for common workflows.

**Main Functions:**
- `nl_plan(instruction, context, llm, groundable_type, action_registry)` → `PycramPlanNode`  
  Single NL instruction → executable plan
  
- `nl_sequential(instruction, context, llm, groundable_type, action_registry)` → `List[PycramPlanNode]`  
  Multi-step NL instruction → ordered list of atomic plans with dependency tracking

- `resolve_match(match, context, llm, groundable_type, instruction, strict_required)` → `PycramPlanNode`  
  Already-built Match → executable plan (role 2)

- `resolve_params(match, llm, groundable_type, instruction, strict_required)` → Action Instance  
  Already-built Match → concrete action (no execution)

---

### 🌐 Reasoning Modules (`reasoning/`)

#### **`slot_filler.py`** — LLM-Driven Parameter Resolution
Dynamic prompts from action introspection; LLM fills free slots.

**Key Functions:**
- `classify_action(instruction, llm, action_registry)` → Action Class  
  Map NL instruction to action class via LLM

- `run_slot_filler(instruction, action_cls, free_slot_names, fixed_slots, world_context, llm)` → `ActionReasoningOutput`  
  LLM fills free action slots with reasoning

**Prompt Features:**
- Per-field docstrings extracted via AST parsing
- Enum members listed for ENUM slots
- Complex fields expanded to dotted sub-fields
- World state context provided
- Entity slots get dedicated section for semantic grounding


#### **`decomposer.py`** — Task Decomposition
Splits compound instructions into atomic steps with dependency tracking.

**Key Class:**
- `TaskDecomposer(llm)` → `DecomposedPlan`
  - `decompose(instruction)` — Multi-step NL → atomic steps
  - Returns: `DecomposedPlan(steps, dependencies)`

**Features:**
- Object-flow dependency detection (pick X → place X)
- Pronoun resolution (it → actual object name)
- Duplicate step removal
- Graceful fallback (returns original instruction if LLM fails)

#### **`llm_config.py`** — LLM Factory
Create LLM instances without hardcoding.

**Key Functions:**
- `make_llm(provider, model, temperature=0.0, **kwargs)` → `BaseChatModel`
  - `provider`: `LLMProvider.OPENAI | LLMProvider.OLLAMA`
  - Returns: LangChain-compatible chat model

---

### 🔌 PyCRAM Bridge (`pycram_bridge/`)

**Purpose:** Isolation layer for all PyCRAM imports and action discovery.

#### **`adapter.py`** — PyCRAM Boundary
Only module that imports PyCRAM directly.

**Key Functions:**
- `discover_action_classes(package_root="pycram.robot_plans.actions")` → `Dict[str, type]`  
  Scan PyCRAM action package tree; return concrete action classes

- `execute_single(match, context)` → `PycramPlanNode`  
  Wrap PyCRAM's execution factory

**Key Protocols (Duck-Typed):**
- `PycramContext` — has `query_backend` attribute
- `PycramPlanNode` — has `perform()` method

#### **`introspector.py`** — Action Schema Analysis
Classifies action dataclass fields by resolution strategy.

**Key Classes:**
- `FieldKind(Enum)` — Classification of fields:
  - `ENTITY` — Symbol subclass → grounded from SymbolGraph
  - `POSE` — Pose/HomogeneousTransformationMatrix
  - `ENUM` — Enum subclass → coerced from string
  - `COMPLEX` — Dataclass → recursively constructed
  - `PRIMITIVE` — bool/int/float/str → taken directly
  - `TYPE_REF` — Type[X] annotation → resolved via SymbolGraph

- `FieldSpec` — Field metadata (name, type, kind, docstring, enum members, sub-fields)

- `ActionSchema` — Full introspection result for one action

- `PycramIntrospector()` — Main introspector
  - `introspect(action_cls)` → `ActionSchema`

---

### 🌍 World Context (`world/`)

#### **`grounder.py`** — Entity Resolution
Maps LLM entity descriptions to Symbol instances in SymbolGraph.

**Key Class:**
- `EntityGrounder(groundable_type=Symbol)`
  - `ground(description: EntityDescriptionSchema)` → `GroundingResult`

**Grounding Tiers:**
1. **Tier 1 (Annotation-based):** Semantic type → Symbol subclass via SymbolGraph.class_diagram
2. **Tier 2 (Name-based):** Substring matching on body display names

**Key Functions:**
- `resolve_symbol_class(semantic_type: str)` → `Optional[Type[Symbol]]`  
  Resolve semantic type string to Symbol subclass

- `ground_entity(description, grounder)` → Symbol instance or None

#### **`serializer.py`** — World State Serialization
Convert SymbolGraph to LLM-readable string.

**Key Function:**
- `serialize_world_from_symbol_graph(groundable_type=Symbol, extra_context="")` → `str`  
  Serialize scene objects and semantic annotations

**Output Sections:**
- Scene objects (excluding structural links like `*_link`)
- Semantic annotations (type hints for objects)
- Extra context (caller-provided)

**Helper Functions (Duck-Typed):**
- `body_display_name(body)` — Extract clean name (handles PrefixedName chains)
- `body_xyz(body)` → `(x, y, z)` — Get position from global_pose
- `body_bounding_box(body)` → `(depth, width, height)` — Get bounding box

---

### 📦 Data Schemas (`schemas/`)

#### **`entities.py`** — Pre-Grounding Entity Description
Used by LLM slot-filler to describe world entities before resolution.

**Key Class:**
- `EntityDescriptionSchema(BaseModel)`
  - `name: str` — Noun phrase from instruction
  - `semantic_type: Optional[str]` — Ontological type hint
  - `spatial_context: Optional[str]` — Location qualifier
  - `attributes: Optional[Dict[str, str]]` — Discriminating properties

#### **`slots.py`** — LLM Output Schemas
Structured outputs from LLM reasoning steps.

**Key Classes:**
- `SlotValue(BaseModel)` — One resolved slot
  - `field_name: str` — Parameter name (dotted for sub-fields)
  - `value: Optional[str]` — Resolved value (for primitives/enums)
  - `entity_description: Optional[EntityDescriptionSchema]` — For entity slots
  - `reasoning: str` — Explanation

- `ActionReasoningOutput(BaseModel)` — Full slot-filling result
  - `action_type: str` — Action class name
  - `slots: List[SlotValue]` — All resolved slots
  - `overall_reasoning: str` — Strategy explanation

- `ActionClassification(BaseModel)` — Action type classification result
  - `action_type: str` — Chosen action class name
  - `confidence: float` — 0.0–1.0 confidence
  - `reasoning: str` — Why this action was chosen

---

## Data Flow Diagrams

### Flow 1: Natural Language → Plan Execution

```
nl_plan("pick up the milk")
    ↓
classify_action() [LLM]
    ↓ action_cls = PickUpAction
_fully_underspecified(action_cls)
    ↓ match = Match(PickUpAction)(object_designator=...)
LLMBackend.evaluate(match)
    ├─ introspect(PickUpAction) → ActionSchema
    ├─ run_slot_filler() [LLM fills slots]
    ├─ ground_entity() [SymbolGraph lookup]
    ├─ _coerce_value() [type conversion]
    └─ reconstruct_complex() [nested fields]
    ↓ action = PickUpAction(object_designator=milk_symbol, ...)
execute_single(action, context)
    ↓
PlanNode.perform() ← Execution in PyCRAM
```

### Flow 2: Multi-Step Decomposition

```
nl_sequential("go to table, pick up milk, put in fridge")
    ↓
TaskDecomposer.decompose() [LLM]
    ├─ steps: ["go to the table", "pick up the milk", "place it in the fridge"]
    └─ dependencies: {2: [1]} ← "place" depends on "pick up"
    ↓
[nl_plan(step) for each step]
    ↓
[PlanNode, PlanNode, PlanNode] ← In dependency order
    ↓
for plan in plans: plan.perform()
```

---

## Testing

### Running Tests

```bash
# All tests
pytest test/llm_reasoner_test/ -v

# Specific module
pytest test/llm_reasoner_test/reasoning/test_slot_filler.py -v
```

### Test Infrastructure

**ScriptedLLM** — Deterministic LLM for testing
- Cycles through pre-built Pydantic instances
- Zero API calls, fully reproducible
- Suits for integration testing

---

## Key Concepts

### Underspecified Match Expression (from KRROOD)

A Match with free variables (`...`) that the backend will resolve:

```python
from krrood.entity_query_language.query.match import Match

# Fully specified (no resolution needed)
action = Match(PickUpAction)(object_designator=milk, grasp_type=GraspType.FRONT)

# Underspecified (free slots for LLM)
match = Match(PickUpAction)(object_designator=..., grasp_type=...)
# LLMBackend fills in: object_designator → milk, grasp_type → GraspType.FRONT
```

### Entity Grounding

Process of resolving LLM entity descriptions to actual world objects:

```
LLM Output: {"name": "milk", "semantic_type": "MILK"}
    ↓
EntityGrounder.ground()
    ├─ Tier 1: Resolve "MILK" → MILK class via SymbolGraph
    ├─         Find all MILK instances in world
    ├─ Tier 2: Filter by name ("milk" substring match)
    └─ Result: milk_symbol from SymbolGraph
```

### Field Kind Classification

**Purpose:** Determine how to resolve each action parameter

| Kind | Type | Resolution | Example |
|------|------|-----------|---------|
| ENTITY | Symbol subclass | SymbolGraph grounding | Body, Manipulator |
| POSE | Pose/HTM | Extract .global_pose | Pose |
| ENUM | Enum subclass | String-to-enum coercion | GraspType.FRONT |
| COMPLEX | Dataclass | Recursive field construction | GraspDescription |
| PRIMITIVE | bool/int/float/str | Taken directly from LLM | 30.0, "success" |
| TYPE_REF | Type[X] | Resolve via SymbolGraph | Type[SemanticAnnotation] |

---

## Related

- **KRROOD** — Entity Query Language and SymbolGraph
- **PyCRAM** — Robot action framework
- **CRAM** — Cognitive Robot Abstract Machine
