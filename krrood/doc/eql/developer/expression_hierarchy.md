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

# Expression Hierarchy

At the core of EQL is a rich hierarchy of symbolic expressions. Every logical operation, variable, and aggregator inherits from {py:class}`~krrood.entity_query_language.core.base_expressions.SymbolicExpression`.

## The Root: `SymbolicExpression`

The {py:class}`~krrood.entity_query_language.core.base_expressions.SymbolicExpression` class defines the interface and life-cycle for all symbolic nodes in the execution graph.

### Key Responsibilities:
- **`_evaluate__`**: The internal method that subclasses must implement to yield {py:class}`~krrood.entity_query_language.core.base_expressions.OperationResult` objects.
- **Parent/Child Management**: Expressions maintain references to their parents and children to build a rooted DAG.
- **Public API**: Methods like `evaluate`, `tolist`, and `first` that trigger graph execution.

## Arity-based Specialization

To simplify implementation, EQL provides base classes for different numbers of children:

- **{py:class}`~krrood.entity_query_language.core.base_expressions.UnaryExpression`**: Operations with exactly one child (e.g., `Not`, `ResultQuantifier`, `Aggregator`).
- **{py:class}`~krrood.entity_query_language.core.base_expressions.BinaryExpression`**: Operations with two children (e.g., `AND`, `OR`, `Comparator`, `Conclusion`).
- **{py:class}`~krrood.entity_query_language.core.base_expressions.MultiArityExpression`**: Operations with N children (e.g., `Union`, `Query`).

```{hint}
When extending EQL, choose the base class that matches your operator's arity. This handles the child field management for you automatically.
```

## Behavioral Mixins

EQL uses mixins to define common behaviors across different parts of the hierarchy:

### 1. `TruthValueOperator`
Used for expressions that contribute to the truth value of a condition. Parents of these nodes only request evaluation when their truth value is relevant (e.g., short-circuiting logic).

### 2. `DerivedExpression`
Represents operations that transform the result stream (like sorting or quantification) without owning the primary data.

### 3. `Selectable`
A specialized expression that can be "selected" in a query's result set. Variables and aggregators are selectables.

```{note}
The hierarchy is designed to enable deeply nested expressions having subqueries as children of parent queries.
```

```{warning}
Directly overriding `_evaluate_` (single underscore) is dangerous. Always override `_evaluate__`
(double underscore) to ensure that parent/child tracking and result processing are handled correctly.
```

## API Reference
- {py:class}`~krrood.entity_query_language.core.base_expressions.SymbolicExpression`
- {py:class}`~krrood.entity_query_language.core.base_expressions.UnaryExpression`
- {py:class}`~krrood.entity_query_language.core.base_expressions.BinaryExpression`
- {py:class}`~krrood.entity_query_language.core.base_expressions.MultiArityExpression`
