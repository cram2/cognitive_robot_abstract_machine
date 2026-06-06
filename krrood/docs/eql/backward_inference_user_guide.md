# Backward Inference User Guide

The EQL-RDR rule tree can answer two kinds of questions:

- **Forward inference**: given a case, what does it conclude? (``rdr.classify(case)``)
- **Backward inference**: given a conclusion value, what conditions would produce it?

Backward inference is the inverse of classification: instead of "what is this case?" it
answers "what do we know about *X*?" — enumerating every rule path in the tree that can
conclude a specific value.

---

## Quick Start

```python
from krrood.entity_query_language.rdr.single_class import EQLSingleClassRDR

# Build an RDR (or load one)
rdr = EQLSingleClassRDR(Animal, "species")
# ... fit it with cases ...

# Query: what do we know about Species.mammal?
knowledge = rdr.what_do_we_know_about(Species.mammal)

if knowledge.is_satisfiable():
    print(f"Found {len(knowledge.sufficient_condition_sets)} path(s)")
    for i, scs in enumerate(knowledge.sufficient_condition_sets, 1):
        print(f"\nPath {i}:")
        for gc in scs.conditions:
            prefix = "not " if gc.negated else ""
            print(f"  {prefix}{gc.expression}")
else:
    print("No rule path produces this conclusion.")
```

## Return Types

**``ConclusionKnowledge``** — the result of a backward-inference query:
- ``conclusion_value`` — the value you queried
- ``sufficient_condition_sets`` — a tuple of ``SufficientConditionSet`` objects, one per
  matching rule path
- ``is_satisfiable()`` — ``True`` when at least one rule path exists

**``SufficientConditionSet``** — one rule path's complete set of conditions:
- ``conditions`` — a tuple of ``GuardCondition`` objects
- ``evaluate_against(variable, case)`` — returns ``True`` when all conditions hold for a
  concrete case (useful for testing or simulation)

**``GuardCondition``** — a single guard on a rule path:
- ``expression`` — the original live EQL expression from the rule tree
- ``negated`` — when ``True``, the guard requires the expression to evaluate to ``False``

## API

### Free function

```python
from krrood.entity_query_language.rdr import what_do_we_know_about

knowledge = what_do_we_know_about(conditions_root, Species.mammal)
```

Takes the raw conditions root (``rdr.conditions_root``) and a conclusion value. Useful when
you have a tree reference but no RDR object.

### Method on ``EQLSingleClassRDR``

```python
knowledge = rdr.what_do_we_know_about(Species.mammal)
```

Convenience method on the RDR itself. Lazily cached by ``BackwardInferenceIndex`` —
the first query traverses the entire tree; subsequent queries for any value are O(1) until
the tree is mutated.

### IPython magic ``%knows``

During interactive fitting the expert can query backward inference directly from the shell:

```
%knows Species.mammal
→ 1 sufficient condition set(s) for Species.mammal:

  1.
    milk == true
```

The conclusion value is evaluated in the shell namespace, so enum values, imported
constants, and shell variables all work as arguments.

```
%knows Species.molusc
→ No rule path concludes Species.molusc.
```

The ``%knows`` magic is only available when the ``IPythonInterface`` is created with an
RDR reference:

```python
interface = IPythonInterface(rdr=rdr)
```

## Re-evaluating Conditions Against Cases

``SufficientConditionSet.evaluate_against()`` tests whether a concrete case satisfies all
the conditions of a rule path:

```python
animal, _, root = _flat_tree()
knowledge = what_do_we_know_about(root, Species.mammal)
scs = knowledge.sufficient_condition_sets[0]

# Returns True if the cow satisfies the conditions for Species.mammal
result = scs.evaluate_against(animal, my_cow)
```

This evaluates each ``GuardCondition`` directly against the case, respecting negations:
all conditions must be satisfied for the result to be ``True``.

## Scope

- **Supported**: ``EQLSingleClassRDR`` (SCRDR) — ``Refinement`` and ``Alternative`` selectors
- **Not yet semantically complete**: ``Next`` (MCRDR) — does not crash but may not capture
  all MCRDR firing semantics correctly
