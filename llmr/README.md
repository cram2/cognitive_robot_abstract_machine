# llmr

**LLM-powered `GenerativeBackend` for KRROOD.** Resolves underspecified PyCRAM action `Match` expressions into executable plans using an LLM.

## Install

```bash
pip install ./llmr
```

## Quick start

### Natural language → executable plan

```python
from llmr import nl_plan, nl_sequential
from llmr.reasoning.llm_config import LLMProvider, make_llm

llm = make_llm(LLMProvider.OPENAI, model="gpt-4o")

# Single instruction
nl_plan("pick up the milk from the table", context=ctx, llm=llm, groundable_type=Body).perform()

# Multi-step (auto-decomposed)
for plan in nl_sequential("go to the kitchen, pick up the milk, place it in the fridge",
                          context=ctx, llm=llm, groundable_type=Body):
    plan.perform()
```

### Pre-built Match → resolved action or plan

```python
from krrood.entity_query_language.query.match import Match
from llmr import resolve_params, resolve_match

match = Match(PickUpAction)(object_designator=..., arm=..., grasp_description=...)

action = resolve_params(match, llm=llm, instruction="pick up the milk", groundable_type=Body)
plan   = resolve_match (match, context=ctx, llm=llm, groundable_type=Body); plan.perform()
```

## How it works

```
NL instruction ─► classify_action ─► required_match(action_cls)
                                           │
                                           ▼
                                     LLMBackend._evaluate
                  ┌────────────────────────┼────────────────────────┐
                  ▼                        ▼                        ▼
           run_slot_filler        EntityGrounder               coerce enum /
             (LLM prompt)        (SymbolGraph)                  primitive
                  │                        │                        │
                  └────────────► write resolved values back ◄────────┘
                                           │
                                           ▼
                                  finalize_match → action
                                           │
                                           ▼
                                  execute_single → PlanNode
```

## Package layout

| Module | Purpose |
|---|---|
| `backend.py` | `LLMBackend(GenerativeBackend)` — main evaluation pipeline |
| `factory.py` | `nl_plan`, `nl_sequential`, `resolve_match`, `resolve_params` |
| `grounder.py` | `EntityGrounder` — two-tier entity resolution (annotation → name) |
| `slot_resolution.py` | Dispatch resolved LLM values by `FieldKind` |
| `bridge/introspect.py` | `PycramIntrospector`, `FieldKind`, `FieldSpec`, `ActionSchema` |
| `bridge/match_reader.py` | `MatchData` / `MatchSlot` snapshots — `read_match`, `write_slot_value`, `finalize_match`, `required_match` |
| `bridge/world_reader.py` | `serialize_world_from_symbol_graph`, `resolve_symbol_class`, body helpers |
| `pycram_bridge/adapter.py` | `discover_action_classes`, `execute_single` — single PyCRAM boundary |
| `reasoning/slot_filler.py` | `classify_action`, `run_slot_filler` — LLM prompts from introspection |
| `reasoning/decomposer.py` | `TaskDecomposer` — compound NL → atomic steps with deps |
| `reasoning/llm_config.py` | `make_llm`, `LLMProvider` |
| `schemas/entities.py` | `EntityDescriptionSchema` — pre-grounding entity description |
| `schemas/slots.py` | `SlotValue`, `ActionReasoningOutput`, `ActionClassification` |

## Field kinds

| Kind | Resolved by |
|---|---|
| `ENTITY` | `EntityGrounder` → `Symbol` instance in `SymbolGraph` |
| `POSE` | `EntityGrounder` + `.global_pose` |
| `ENUM` | String → `Enum` member coercion |
| `PRIMITIVE` | Direct coercion (`bool`, `int`, `float`, `str`) |
| `TYPE_REF` | `resolve_symbol_class` → `Symbol` subclass |
| `COMPLEX` | Recursed as a nested KRROOD `Match` leaf |

## Testing

```bash
pytest test/llmr_test --confcutdir=test/llmr_test
```

The suite uses `ScriptedLLM` (a deterministic `BaseChatModel`) — no API key, no network. Live-LLM smoke tests live in `test/llmr_test/live/` and activate only with `LLMR_LIVE_TESTS=1` and a valid `OPENAI_API_KEY`.

## Design invariants

- **Single krrood boundary.** All krrood access is funneled through `llmr.bridge.*`.
- **Single pycram boundary.** All pycram imports live in `llmr.pycram_bridge.adapter`.
- **No world-package imports.** World context is derived from `SymbolGraph`, not from a concrete world object.
- **One-way dependency.** `llmr → krrood`; no circular imports.
