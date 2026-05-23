---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Inference Explanation Internals

## Architecture Overview

The `InferenceExplanation` machinery sits at the end of the EQL evaluation pipeline:

```
EQL evaluation pipeline
        │
        ▼
EvaluationContext  ──observers──►  SatisfiedConditionTracker
                                   EvaluationTracker
                                   InferenceRecorder
                                           │
                                           ▼ on_result_yielded
                                   register_inference(instance, variable_node, result)
                                           │
                                           ▼
                              instance._inference_explanation_ = InferenceExplanation(...)
```

## EvaluationObservers

The evaluation pipeline in `SymbolicExpression._evaluate_` notifies an `EvaluationContext` at
four points:

| Hook | Purpose |
|---|---|
| `on_evaluate_enter` | Record which expressions have been entered (→ `EVALUATED_IDS_KEY`) |
| `on_result_yielded` | Stamp satisfied/evaluated IDs onto each `OperationResult` |
| `on_conclusions_processed` | At the conditions root: walk the truth map and build the final `satisfied_condition_ids` |
| `on_evaluate_exit` | No-op by default (available for custom observers) |

Three built-in observers wire into this:

- **`EvaluationTracker`** — accumulates `EVALUATED_IDS_KEY` as the tree is traversed; stamps
  the cumulative set on every yielded `OperationResult`.
- **`SatisfiedConditionTracker`** — propagates `satisfied_condition_ids` forward through the
  result chain; at the conditions root, builds the final set by consulting the chain's truth map.
- **`InferenceRecorder`** — on `on_result_yielded`, calls `register_inference` for every
  `InstantiatedVariable` that produced a binding.

## Why `InferenceExplanation` Inherits from `Symbol`

This was a deliberate design choice driven by four principles:

1. **No global registry** — the explanation is stored on the inferred instance
   (`instance._inference_explanation_`). Its lifecycle is identical to the instance's lifecycle;
   when the instance is GC'd, the explanation is too. This satisfies the *Object Ownership / RAII*
   principle.

2. **Life-Cylcle Management stays together** — inheriting from `Symbol` means explanations are registered in the
   `SymbolGraph` and managed with the rest of object instances. Clearing SymbolGraph clears all instance related knowledge.

3. **Tell Don't Ask** — callers ask the object for its own metadata
   (`explain_inference(instance)` → `instance._inference_explanation_`) rather than querying an
   external registry.

4. **Avoiding the `WeakKeyDictionary` anti-pattern** — the previous design stored explanations
   in `INFERENCE_RECORD: WeakKeyDictionary[instance → explanation]`. Because
   `explanation.instance` held a strong reference back to the key, the keys could never be
   collected. Removing the global registry and storing a `weakref` from the explanation back to
   its instance breaks the cycle.

## Why `_inference_explanation_` is a Field on `Symbol`

The `Symbol` base class declares:

```python
_inference_explanation_: Optional[InferenceExplanation] = field(
    default=None, init=False, repr=False, compare=False
)
```

Using `init=False` ensures the field is never a constructor parameter in any `Symbol` subclass —
it remains invisible to callers. It is set later by `register_inference` via normal attribute
assignment.

The field lives on `Symbol` (not on a separate mixin or on `InferenceExplanation` itself)
because:
- Every inferred object is a `Symbol`; no extra protocol or interface is needed.
- The field is excluded from `__repr__` and `__eq__` to avoid polluting equality semantics and
  string output.
- Runtime import of `InferenceExplanation` is guarded by `TYPE_CHECKING` to prevent a circular
  import between `symbol_graph.py` and `explanation.py`.

## The Weakref Back-Reference

`InferenceExplanation` stores the inferred instance as a `weakref.ref`:

```python
_instance_ref: Optional[weakref.ref] = field(
    default=None, init=False, repr=False, compare=False
)
```

This prevents a strong reference cycle:

```
instance  ──strong──►  InferenceExplanation  ──weak──►  (back to instance)
```

If `_instance_ref` were a strong reference, `instance` and `InferenceExplanation` would form a
cycle that prevents both from being collected even when no external code holds either.

## Memory Management: The `lru_cache` Pitfall

A `@lru_cache` applied to an instance method is a **class-level** cache. Its keys contain
`self`, so every expression object that has ever been looked up is kept strongly referenced by
the cache — indefinitely, because the class itself is never collected.

`SymbolicExpression._get_expression_by_id_` was originally decorated with `@lru_cache`. Because
EQL `Variable` instances store their domain data (e.g. `world.bodies`) in
`_re_enterable_domain_generator_.materialized_values`, keeping a `Variable` alive keeps the
entire domain (and through the `world` back-reference on each entity, the `World` object) alive.

The fix is to use a **per-instance dict** stored in `self.__dict__`:

```python
def _get_expression_by_id_(self, id_: uuid.UUID) -> SymbolicExpression:
    cache: dict = self.__dict__.setdefault('_expression_id_cache_', {})
    if id_ not in cache:
        try:
            cache[id_] = next(...)
        except StopIteration:
            raise NoExpressionFoundForGivenID(self, id_)
    return cache[id_]
```

When the expression object is GC'd, `__dict__` (and the embedded cache) is collected with it.
No external root is created.

## API Reference
- {py:class}`~krrood.entity_query_language.explanation.explanation.InferenceExplanation`
- {py:func}`~krrood.entity_query_language.explanation.explanation.register_inference`
- {py:func}`~krrood.entity_query_language.explanation.explanation.explain_inference`
- {py:class}`~krrood.symbol_graph.symbol_graph.Symbol`
