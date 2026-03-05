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

# Architecture Overview

The Entity Query Language (EQL) is built on a clear separation between query definition and execution. This architectural pattern allows for complex query manipulation, optimization, and deferred evaluation.

## The Three Stages of a Query

Every EQL query passes through three distinct lifecycle stages:

1.  **Builder (Blueprint)**: The user interacts with {py:class}`~krrood.entity_query_language.query.query.Query`
that uses builders of type {py:class}`~krrood.entity_query_language.query.builders.ExpressionBuilder`s to define the 
query's structure (where, having, grouped_by, etc.).
2.  **Expression (Execution Graph)**: When `.build()` is called on the {py:class}`~krrood.entity_query_language.query.query.Query`
object, the builders materialize into nodes of type {py:class}`~krrood.entity_query_language.core.base_expressions.SymbolicExpression`,
forming a symbolic execution graph.
3.  **Execution (Evaluation)**: The graph is traversed using {py:meth}`~krrood.entity_query_language.core.base_expressions.SymbolicExpression.evaluate`, which yields results as a stream of bindings.

## Builders vs. Expressions

| Stage | Class                                   | Responsibility |
| :--- |:----------------------------------------| :--- |
| **Builder** | `WhereBuilder`, `GroupedByBuilder`, ... | Collects metadata, validates structure, and stores constraints. |
| **Expression** | `Where`, `GroupedBy`, ...               | Implements the actual evaluation logic (`_evaluate__`). |

```{hint}
The separation allows EQL to validate the query structure (e.g., checking if aggregators are used correctly in `having`) *before* the expensive evaluation process begins.
```

## The Query Lifecycle

1.  **Definition**: You call `entity(v).where(...)`. This populates the internal builders of the `Query` object.
2.  **Building**: When you call `evaluate()` method or use the query as part of another query, the `.build()` method is triggered.
It converts all builders into their corresponding expression nodes.
3.  **Evaluation**: The root expression (a `ResultQuantifier`) calls `_evaluate_()` on its children, starting the recursive evaluation process.

```{note}
The `@modifies_query_structure` decorator ensures that once a query has been built into an execution graph, its structure cannot be modified further.
```

```{warning}
Directly manipulating expression nodes is discouraged for end-users. Always use the `factories` and `Query` methods to build your queries.
```

## Simplified Visualization of the Pipeline

```mermaid
graph LR
    A[User Code] --> B[Builders]
    B -- .build() --> C[Expression Graph]
    C -- .evaluate() --> D[Result Stream]
```

## API Reference
- {py:class}`~krrood.entity_query_language.query.builders.ExpressionBuilder`
- {py:class}`~krrood.entity_query_language.core.base_expressions.SymbolicExpression`
- {py:class}`~krrood.entity_query_language.query.query.Query`
