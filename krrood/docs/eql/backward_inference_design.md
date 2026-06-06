# Backward Inference Design Document

## Overview

Backward inference answers "what do we know about conclusion value X?" by walking the
RDR rule-tree DAG in reverse. Each leaf rule that can produce the target value yields a
**sufficient condition set** — the conjunction of all path conditions (selectors) that
must be true for that rule to fire. The full answer is a disjunction of all such sets
(DNF form).

The reasoning paradigm is **backward chaining**: goal-directed reasoning that works
backwards from a conclusion to the preconditions that imply it.

---

## Architecture

### Module layout

```
krrood/src/krrood/entity_query_language/rdr/
  backward_inference.py   — Core data types, traversal, index, public API
  single_class.py         — EQLSingleClassRDR.what_do_we_know_about()
  magics.py               — %knows IPython magic factory
  interactive.py          — IPythonInterface.rdr field and magic registration
  utils.py                — _extract_value(), _conclusions_of() (shared utilities)
  __init__.py             — Re-exports public API symbols
```

```
test/krrood_test/test_eql_rdr/
  test_backward_inference.py   — 40 tests for core backward inference
  test_interactive_expert.py   — 7 tests for %knows magic
```

### Data flow

```
                 what_do_we_know_about(root, value)
                              |
                    BackwardInferenceIndex.query()
                              |
                    _build_full_index(root)
                              |
                    _collect_rule_paths(node, guard)
                              |
                    Guards accumulated per selector:
                    Refinement:  left ← NOT(right),  right ← left
                    Alternative: right ← NOT(left)
                    Next:        each child independent
                              |
                    Rule path → SufficientConditionSet(conditions)
                              |
                    Bucketed by conclusion value → ConclusionKnowledge
```

### Tree traversal semantics

| Selector | Left branch | Right branch |
|----------|-------------|--------------|
| ``Refinement(left, right)`` | ``NOT(right)`` (parent wins unless overridden) | ``left`` as positive guard (override requires parent) |
| ``Alternative(left, right)`` | no guard | ``NOT(left)`` (else-if: skip left) |
| ``Next`` | each child is a separate disjunct, no cross-guards | — |

At each leaf node the accumulated guards are prepended to the leaf's own condition.
The leaf condition is always added as a positive (non-negated) guard — the rule's own
condition expression.

---

## Key Design Decisions

### 1. ``GuardCondition`` with negated flag — no live tree mutation

Calling ``not_()`` on a live EQL expression node sets ``expression._parent_`` to the new
``Not`` wrapper, corrupting the original tree's parent references. The solution is a
``GuardCondition`` value object that stores ``(expression, negated: bool)`` — the
negation is a flag, not a tree operation. ``evaluate_against()`` applies the flag at
evaluation time by inverting the truth value.

### 2. Direct evaluation on live tree — no ``deepcopy``

``copy.deepcopy`` on ``SymbolicExpression`` objects crashes because
``MappedVariable.__hash__`` calls ``hash(self.hashable_key)``, which produces an
unhashable dict for the copy. ``evaluate_against()`` evaluates each guard condition
directly on the original live tree node by calling ``expression.evaluate()``.

The ``case_variable._update_domain_([case])`` call before evaluation binds the concrete
case to the shared variable — the same mechanism ``classify_case`` uses. This mutates
the shared variable but is a normal part of each evaluation cycle, not a permanent
structural change.

### 3. Lazy caching in ``BackwardInferenceIndex``

One full traversal builds the complete index for all conclusion values in a single pass.
Subsequent queries for any value are O(1) dict lookups. This is the right trade-off
because:

- Rule trees are mutated infrequently (only during fitting)
- Querying multiple values is common (e.g. checking "what do we know about" several
  species in a zoo RDR)
- Tree size is small enough that a full traversal is cheap

The cache is invalidated on every rule insertion via ``_backward_index.invalidate()``
in ``EQLSingleClassRDR.fit_case()``. The next query after invalidation rebuilds the
entire index.

### 4. SCRDR only; MCRDR deferred

``_collect_rule_paths`` handles ``Next`` nodes (MCRDR) without crashing — it iterates
each child as a separate disjunct with no cross-guards. This is correct for some MCRDR
patterns but does not capture the full MCRDR firing semantics (multiple refinements
from different parents). Full MCRDR support is deferred.

### 5. Free function + RDR method

The core backward inference logic lives in a free function ``what_do_we_know_about()``
in ``backward_inference.py``. ``EQLSingleClassRDR`` has a thin wrapper that delegates to
``BackwardInferenceIndex.query()``. This keeps the traversal logic independent of the
RDR class hierarchy.

---

## Extension Points

### Adding a new selector type

Add a new ``isinstance`` branch in ``_collect_rule_paths()`` — the current pattern uses
explicit dispatch. If selectors become numerous or third-party, a polymorphic ``visit``
method on each selector class would be cleaner.

### Supporting MCRDR

MCRDR uses ``Next`` nodes where multiple refinements can coexist. The current guard
accumulation for ``Next`` is a placeholder. Full MCRDR support would need:

- Tracking which refinement path a set of conclusions came from (multiple parent refs)
- Adding the appropriate guards for sibling refinements at the same level

### CNF subsumption

The current implementation returns DNF output — a disjunction of conjunctions. A future
CNF subsumption pass could reorganise the conditions into a decision tree structure for
more efficient checking.

---

## Code Reuse

| Utility | Source | Used in |
|---------|--------|---------|
| ``_extract_value(add_node)`` | ``rdr/utils.py`` | ``backward_inference.py``, ``observer.py`` |
| ``_conclusions_of(node)`` | ``rdr/utils.py`` | ``backward_inference.py``, ``rule_tree_view.py`` |
| ``format_condition(expr)`` | ``rdr/rule_tree_view.py`` | ``magics.py`` (``%knows`` output) |

These shared utilities were extracted from their original modules to eliminate
duplication.

---

## Test Coverage

40 tests in ``test_backward_inference.py`` covering:

- **Structure**: flat alternative trees, refinement trees, mixed trees, empty trees
- **Evaluate against**: correctness for concrete cases (cow, eagle, tuna, frog)
- **RDR integration**: ``EQLSingleClassRDR.what_do_we_know_about()`` method
- **Cache invalidation**: index rebuilds on refinement, alternative, and ``fit_case()``
- **Edge cases**: empty constructors, vacuous condition sets

7 tests in ``test_interactive_expert.py`` covering:

- Namespace key presence/absence
- Query results through the interactive fitting path
- ``%knows`` magic function: eval, bad arguments, empty line
