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

# The Variable System

EQL variables are more than just placeholders; they are active components of the execution graph that know how to resolve data from object instances.

## `CanBehaveLikeAVariable`

The {py:class}`~krrood.entity_query_language.core.mapped_variable.CanBehaveLikeAVariable` mixin is what makes EQL so "pythonic." It overrides standard Python dunder methods to capture operations and turn them into symbolic expressions.

- **`__getattr__`**: Captures attribute access and returns an {py:class}`~krrood.entity_query_language.core.mapped_variable.Attribute`.
- **`__getitem__`**: Captures indexing and returns an {py:class}`~krrood.entity_query_language.core.mapped_variable.Index`.
- **`__call__`**: Captures method calls and returns a {py:class}`~krrood.entity_query_language.core.mapped_variable.Call`.

```{hint}
This is why you can write `robot.name == "R2D2"`. The `robot` variable captures the `.name` access and returns a symbolic `Attribute` node.
```

## Mapped Variables

A {py:class}`~krrood.entity_query_language.core.mapped_variable.MappedVariable` is a {py:class}`~krrood.entity_query_language.core.base_expressions.UnaryExpression` that transforms the value of its child.

### 1. `Attribute`
Represents access to a Python attribute. During evaluation, it uses `getattr(value, name)`.

### 2. `Index`
Represents access to a dictionary key or list index. It uses `value[key]`.

### 3. `Call`
Represents a symbolic method call. It stores the arguments and keyword arguments, and executes the call during evaluation.

```{note}
Mapped variables are cached internally to ensure that the same symbolic path (e.g., `robot.name`) always resolves to the same expression object within a query context.
```

## Flattening with `FlatVariable`

The {py:class}`~krrood.entity_query_language.core.mapped_variable.FlatVariable` is a special mapped variable used when an attribute returns an iterable, but you want to treat its elements as individual bindings in the result stream.

```{note}
Use `flat_variable()` explicitly when you want to iterate over elements of attributes that are iterable.
Standard attribute access on an iterable valued attribute will return the iterables as values, not their elements.
```

## API Reference
- {py:class}`~krrood.entity_query_language.core.mapped_variable.CanBehaveLikeAVariable`
- {py:class}`~krrood.entity_query_language.core.mapped_variable.MappedVariable`
- {py:class}`~krrood.entity_query_language.core.mapped_variable.Attribute`
- {py:class}`~krrood.entity_query_language.core.mapped_variable.FlatVariable`
