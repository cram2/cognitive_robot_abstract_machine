---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Quickstart
This is a quickstart guide to get you up and running with the `random_events` library.

## Installation

To install the library, run the following command

```bash
pip install random-events
```

Next, import the necessary functionality:

```{code-cell} ipython3
:tags: []


from random_events.variable import Symbolic, Continuous, Integer
from random_events.product_algebra import SimpleEvent, Event
from random_events.interval import SimpleInterval, Interval, closed, closed_open, open_closed, open
from random_events.set import SetElement, Set
import plotly
import plotly.graph_objects as go
```

## Preface

The `random_events` library is a Python library for working with sigma algebras. 
For every algebra, there are two kinds of sets, the simple sets and the composite sets.
Simple sets are sets that can be represented as single objects.
Composite sets are sets that are composed of multiple, sorted simple sets.


## Intervals

Intervals are a fundamental concept in the `random_events` library.
They are used to represent the range of possible values that a numeric variable can take.
There are two classes to interact with intervals: `SimpleInterval` and `Interval`.
However, it is **strongly recommended** to use the `Interval` class,
as it provides an API implementing all set operations.
First, create two simple intervals:


```{code-cell} ipython3
:tags: []

si1 = SimpleInterval(0, 1)
si2 = SimpleInterval(0.5, 1.5)
si1, si2
```

Simple sets of any kind can be converted to composite sets using the `.as_composite_set()` method.

```{code-cell} ipython3
:tags: []

i1 = si1.as_composite_set()
i2 = si2.as_composite_set()
i1, i2
```

All set operations are supported for composite sets.
All set operations automatically return simple and disjoint representations of the resulting set.

```{code-cell} ipython3
:tags: []

print(i1.intersection_with(i2))
print(i1.union_with(i2))
print(i1.difference_with(i2))
print(i1.complement())
```

Set operations can also be invoked by calling the built-in methods.

```{code-cell} ipython3
:tags: []

print(i1 & i2)
print(i1 | i2)
print(i1 - i2)
print(~i1)
```

Intervals can also be created using shortcuts for the most common types of intervals.

```{code-cell} ipython3
:tags: []

print(closed(0, 1))
print(closed_open(0, 1)) 
print(open_closed(0, 1))
print(open(0, 1))
```

## Sets

Since a sigma algebra requires more information than it is stored in regular python sets, the `SetElement` and `Set` 
classes are used.
A set element is a simple set, and a set is a composite set.
Whenever you want to create a set, just pass it some iterable, for example, an enum.

```{code-cell} ipython3
:tags: []
from enum import IntEnum

class Symbol(IntEnum):
    APPLE = 0
    DOG = 1
    RAIN = 2
```

Now you can interact with sets and set elements.

```{code-cell} ipython3
:tags: []

s1 = SetElement(Symbol.APPLE, Symbol).as_composite_set()
s2 = SetElement(Symbol.DOG, Symbol).as_composite_set()
print(s1)
print(~s1)
print(s1 & s2)
print(s1 | s2)
```

## Variables

Describing multidimensional algebras requires variables that describe the type of dimension.
Variables are either `Symbolic`, `Integer` or `Continuous`.
Symbolic variables need a class that inherits from `SetElement` to describe the possible values.
Integer and Continuous variables need no specification of their domain.

```{code-cell} ipython3
:tags: []

a = Symbolic("a", Set.from_iterable(Symbol))
x = Continuous("x")
y = Continuous("y")
z = Integer("z")
a, x, y, z
```

## Events

Describing events requires the use of the `SimpleEvent` and `Event` classes.
Events are elements of the product algebra.
Simple events are dictionary like objects that describe the values of the variables.

```{code-cell} ipython3
:tags: []

e1 = SimpleEvent({a: Symbol.APPLE, x: closed(0, 1), y: closed(2, 3), z: closed(0, 10)}).as_composite_set()
e2 = SimpleEvent({a: (Symbol.APPLE, Symbol.DOG), x: closed(0, 4), y: closed(0, 5), z: closed(0, 20)}).as_composite_set()
print(e1)
print(e1 & e2)
print(e1 | e2)
print(e2 - e1)
```

Events can also be plotted.

```{code-cell} ipython3
:tags: []

e3 = SimpleEvent({x: closed(0, 1) | closed(2, 2.5), y: closed(2, 3) | closed(4, 5)}).as_composite_set()
fig = go.Figure(e3.plot(), e3.plotly_layout())
fig.show()
```

JSON serialization is also supported for every object.

```{code-cell} ipython3
:tags: []

e4 = e2 - e1
print(e4.to_json())
e5 = Event.from_json(e4.to_json())
e5 == e4
```
