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

# Execution Engine

The EQL execution engine is responsible for traversing the expression graph and producing results. It is built on a streaming architecture that uses Python generators to efficiently process data.

## `OperationResult` and Bindings

The fundamental unit of data in the engine is the {py:class}`~krrood.entity_query_language.core.base_expressions.OperationResult`. It wraps two critical pieces of information:
1.  **Bindings**: A mapping from symbolic variable IDs to their current values.
2.  **Truth Value**: A boolean indicating whether the current set of bindings satisfies the expression's constraints.

```{hint}
Every expression in EQL yields `OperationResult` objects. This allows the engine to pass the state of the query from one node to the next.
```

## `PerformsCartesianProduct` Mixin

Many complex operators, such as `Query`, `Comparator`, and `GroupedBy`, need to evaluate multiple children and combine their results. EQL uses the {py:class}`~krrood.entity_query_language.operators.set_operations.PerformsCartesianProduct` mixin to handle this logic.

### How it works:
1.  **Nested Evaluation**: It evaluates its children in sequence.
2.  **Binding Propagation**: The bindings produced by the first child are passed as "sources" to the second child, and so on.
3.  **Result Combination**: It produces a Cartesian product of all valid binding combinations from its children.

```{note}
The engine uses the {py:func}`~krrood.entity_query_language.utils.cartesian_product_while_passing_the_bindings_around` utility to implement this nested generator logic efficiently.
```

## Short-Circuiting and Truth Values

EQL optimizes execution by short-circuiting logical operations.

- **Conjunction (AND)**: If the left child returns a result with `is_false=True`, the right child is never evaluated for that particular binding set.
- **Disjunction (OR)**: If the left child returns `is_false=False`, the right child is skipped.

```{warning}
Because EQL is generator-based, the order of children in a `PerformsCartesianProduct` can significantly impact performance. EQL includes optimization logic to reorder children (e.g., putting more restrictive filters earlier).
```

## The Evaluation Loop

1.  **Root Trigger**: A call to `.evaluate()` starts at the top of the graph.
2.  **Recursive Call**: Each node calls `_evaluate_()` on its children.
3.  **Leaf Resolution**: Leaf nodes (like `Variable`) resolve their domains and yield initial bindings.
4.  **Back-propagation**: Results flow back up the graph, being transformed, filtered, or aggregated along the way.

## API Reference
- {py:class}`~krrood.entity_query_language.core.base_expressions.OperationResult`
- {py:class}`~krrood.entity_query_language.operators.set_operations.PerformsCartesianProduct`
- {py:func}`~krrood.entity_query_language.utils.cartesian_product_while_passing_the_bindings_around`
