---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Role Pattern — User Guide

## The Problem Roles Solve

Sometimes an object needs to take on a new **semantic context** — extra, context-specific
attributes and behaviour — that does not belong to its permanent type. A `Person` may, over time,
act as a `CEO` or a `Professor`: each context adds its own data (the company they head, the courses
they teach), but none of it changes *who the person is*. Modelling this with subclassing or by
piling optional fields onto `Person` is awkward — the context is temporary, an entity may play
several contexts at once, and the extra data pollutes the base type.

The **Role pattern** solves this: it lets an existing object take on a new context — adding
context-specific attributes and behaviour — **without changing the original object's type or
identity**.

Reach for it when:

- an object needs context-specific attributes/behaviour that do not logically belong to the
  original type;
- the context is temporary or situational — the entity may gain and lose it over its lifetime;
- the same entity can play multiple semantic contexts simultaneously;
- any part of the system must be able to ask "does this entity have role X?" without scanning all
  instances;
- the role and the original object should be recognizable as the **same underlying entity** while
  remaining **distinct objects**.

When the pattern is the wrong fit, see *When Not to Use the Role Pattern* and the *Decision Guide*
at the end of this guide.

## What Is a Role?

A **role** adds a semantic context to an existing object — the **role taker** — without altering
the taker's type or identity. A role is an **ordinary object with its own identity**: it is equal
only to itself, never to its taker. When you need to ask whether two objects refer to the same
underlying entity, the `IsSameSemanticEntity` predicate answers that — it sees through role chains
to the root entity.

This guide uses a university domain as its running example. A `Person` is a persistent entity with
a name. Over time that person may become a `CEO` or a `Professor`. These are roles: they add
context-specific data, but the person's identity never changes.

## Classes and Instances: the Mental Model

Roles live on two layers — the **types** you define and the **objects** you create. Keeping them
straight removes most of the early confusion about "is a role taker a class or an instance?":

- There is one abstract base class, **`Role`** — you never use it directly.
- You **define a role type** by subclassing it as **`Role[T]`** — `CEO`, `Professor`,
  `Representative`. Each subclass is a *class*: "the CEO role type", "the Professor role type".
- A **role taker** is an *instance* of any class in your model — `alice = Person(name="Alice")`.
  It is a normal object and does not need to know anything about roles.
- A **role** is an *instance* of one of your `Role` subclasses, attached to a taker —
  `CEO(role_taker=alice, ...)`.
- One taker *instance* can hold **many role instances** at once: different types (`alice` as both a
  `CEO` and a `Professor`), several of the *same* type (`alice` heading two companies), and even
  **layered** ones — a role's taker can itself be a role (`Representative(role_taker=ceo)`).

So both "role taker" and "role" are *instances*; `Role` and its subclasses are the *classes*. The
sections below build this up one step at a time using `alice`.

## Defining a Role Taker

A role taker is any dataclass you define.

```{code-cell} ipython3
from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import List, Optional

from krrood.entity_query_language.predicate import Symbol
from krrood.patterns.role_predicates import IsSameSemanticEntity


@dataclass(eq=False)
class Person(Symbol):
    """A persistent human entity with a unique name."""

    name: str

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass(eq=False)
class Company(Symbol):
    """An organisation that people can head or work for."""

    name: str

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass(eq=False)
class Course(Symbol):
    """An academic course that a professor can teach."""

    name: str

    def __hash__(self) -> int:
        return hash(self.name)


alice = Person(name="Alice")
print(alice)
```

## Defining a Role

A role is a dataclass that inherits from `Role[T]`, where `T` is the type of its role taker.
The role taker is the inherited keyword-only `role_taker` field; a role declares only its own
fields and never re-declares the taker.

```{code-cell} ipython3
from krrood.patterns.role import Role


@dataclass(eq=False)
class CEO(Role[Person]):
    """The role of being the chief executive of a company."""

    head_of: Optional[Company] = None
```

Notice that `CEO` inherits from `Role[Person]` — it does **not** inherit from
`Person`. The role and its taker are entirely separate classes connected only through the
`role_taker` field.

## Constructing a Role

Role construction is explicit. You pass the role taker as the keyword-only `role_taker`
argument to the role's constructor, alongside any role-specific fields.

```{code-cell} ipython3
acme = Company(name="ACME")
ceo = CEO(role_taker=alice, head_of=acme)
print(ceo)
```

Constructing a role registers it against its taker, which is what makes the role discoverable by the
lookups below.

## Distinct Identity

A role and its role taker each carry **different, extra information** — the taker holds the
entity's persistent data, and each role adds its own context — so they must stay **distinct members
of sets and dictionaries**: a role, its taker, and multiple roles on one taker should never collapse
into a single entry.

Python ties equality and hashing together with one rule: *objects that compare equal must have the
same hash value*
([`object.__hash__`](https://docs.python.org/3/reference/datamodel.html#object.__hash__) in the
Python Data Model). Keeping a role and its taker distinct means they need different hashes, and by
that rule anything with a different hash must also compare unequal. So a `Role` deliberately uses
**identity** for both: it hashes by object identity (`__hash__ = object.__hash__`) and is equal only
to itself (`self is other`). (Defining `__eq__` on a class otherwise resets its `__hash__` to
`None`, making instances unhashable; the same Data Model page is why `Role` restores it explicitly.)

```{code-cell} ipython3
print("alice == ceo:", alice == ceo)
print("alice is ceo:", alice is ceo)
print("set size:", len({alice, ceo}))
```

Because "are these equal?" now means strictly "are these the very same object?", the separate
question "do these refer to the same underlying entity?" is answered by the `IsSameSemanticEntity`
predicate. It unwraps any role to its root entity before comparing, so a role and its taker count as
the same entity:

```{code-cell} ipython3
print("IsSameSemanticEntity(alice, ceo):", bool(IsSameSemanticEntity(alice, ceo)))
```

## Attribute Access

Reading an attribute that is not declared on the role is delegated to the role taker via
`__getattr__`, so you can read role-taker attributes through the role as if they were declared
there.

Assignments behave differently: they always target the role itself, never the role taker, and only
the role's own declared fields may be assigned. Assigning any other name raises
`RoleAttributeNotDeclaredError`, so a write can never silently shadow a role-taker attribute. To
change the role taker, assign through `role.role_taker`.

```{code-cell} ipython3
from krrood.patterns.exceptions import RoleAttributeNotDeclaredError

# Reading a role-taker attribute through the role delegates to the taker
print("ceo.name:", ceo.name)

# Assigning a name the role does not declare is rejected (it would shadow a taker attribute)
try:
    ceo.name = "Acting CEO"
except RoleAttributeNotDeclaredError as error:
    print("rejected:", error)

# To change the taker, assign through its reference explicitly
ceo.role_taker.name = "Alicia"
print("alice.name after writing through role_taker:", alice.name)

# Restore the original name
ceo.role_taker.name = "Alice"

# Role-native fields are read from the role, not the taker
print("ceo.head_of:", ceo.head_of)
print("hasattr(alice, 'head_of'):", hasattr(alice, "head_of"))
```

## IDE Support: Python Role Lens

Because a role's attributes come from two places — the fields declared on the role class and,
through delegation, the attributes of its role-taker chain — a standard IDE only sees the
fields declared directly on the role class. The **Python Role Lens** plugin for JetBrains IDEs
(such as PyCharm) closes that gap: it surfaces *all* attributes accessible on a role, including
the delegated ones, and lets you navigate straight to the source class that defines each
attribute — much like navigating an inheritance hierarchy.

- Plugin: [Python Role Lens on the JetBrains Marketplace](https://plugins.jetbrains.com/plugin/32276-python-role-lens)
- Install from your IDE: open **Settings → Plugins → Marketplace** and search for
  *Python Role Lens* (plugin id `io.github.abdelrhmanbassiouny.pyroles`).

<iframe width="384px" height="319px" src="https://plugins.jetbrains.com/embeddable/card/32276" frameborder="0"></iframe>

<iframe width="245px" height="48px" src="https://plugins.jetbrains.com/embeddable/install/32276" frameborder="0"></iframe>

A short demo of the plugin in action:

<iframe width="560" height="315" src="https://www.youtube.com/embed/qoPctQ9Z5Eo" frameborder="0" allowfullscreen></iframe>

## Querying Roles

`Role` provides class methods for checking whether a role taker has a role and for retrieving
role instances.

```{code-cell} ipython3
# Check whether alice has a CEO role
print("has CEO role:", Role.has_role(alice, CEO))

# Retrieve all CEO roles for alice
ceos = Role.roles_for(alice, CEO)
print("CEO roles:", ceos)

# Retrieve all roles of any type
all_roles = Role.roles_for(alice)
print("all roles:", all_roles)
```

## Parallel Roles

A single role taker can hold multiple roles simultaneously. Each role is a separate object with
its own identity, but all of them resolve to the same underlying entity.

```{code-cell} ipython3
@dataclass(eq=False)
class Professor(Role[Person]):
    """The role of being an academic who teaches courses."""

    teaches: List[Course] = field(default_factory=list, kw_only=True)


cs_101 = Course(name="CS 101")
professor = Professor(role_taker=alice, teaches=[cs_101])

print("ceo == professor:", ceo == professor)
print("IsSameSemanticEntity(ceo, professor):", bool(IsSameSemanticEntity(ceo, professor)))
print("IsSameSemanticEntity(professor, alice):", bool(IsSameSemanticEntity(professor, alice)))
print("set size:", len({alice, ceo, professor}))

# Each role preserves its own context-specific data
print("CEO heads:", ceo.head_of)
print("Professor teaches:", professor.teaches)

# The taker knows about every role it currently holds
print("all roles of alice:", Role.roles_for(alice))
```

## Multiple Roles of the Same Type

Because roles are distinct objects, a taker can hold several roles of the *same* type without
them collapsing into one. Each is retrievable from the taker and keeps its own data.

```{code-cell} ipython3
globex = Company(name="Globex")
ceo_of_globex = CEO(role_taker=alice, head_of=globex)

# Both CEO roles are kept and returned
same_type_roles = Role.roles_for(alice, CEO)
print("number of CEO roles:", len(same_type_roles))
print("companies headed:", sorted(role.head_of.name for role in same_type_roles))

# They are distinct objects, yet the same underlying entity
print("ceo is ceo_of_globex:", ceo is ceo_of_globex)
print("IsSameSemanticEntity(ceo, ceo_of_globex):", bool(IsSameSemanticEntity(ceo, ceo_of_globex)))
```

## Role Chaining

A role's role taker can itself be a role. This lets you layer multiple levels of semantic
context onto the same entity.

```{code-cell} ipython3
@dataclass(eq=False)
class Representative(Role[CEO]):
    """The role of a CEO acting as a representative of their company."""

    represents: Optional[Company] = None


rep = Representative(role_taker=ceo, represents=acme)

print("rep == ceo:", rep == ceo)
print("IsSameSemanticEntity(rep, ceo):", bool(IsSameSemanticEntity(rep, ceo)))
print("IsSameSemanticEntity(rep, alice):", bool(IsSameSemanticEntity(rep, alice)))
print("set size:", len({alice, ceo, professor, rep}))
```

The chain is transparent. Querying from any point in the chain resolves all the way back to
the root entity.

```{code-cell} ipython3
# Check roles from the perspective of the root taker
print("alice has Representative role:", Role.has_role(alice, Representative))
print(
    "Representative roles of alice:",
    Role.roles_for(alice, Representative),
)

# Navigate the chain programmatically
print("rep.role_taker:", rep.role_taker)
print("rep.root_persistent_entity:", rep.root_persistent_entity)
print("root type:", Representative.get_root_role_taker_type())
```
## Available Features

| Feature | How to use it |
|---|---|
| Define a role class | Inherit from `Role[T]` and add role-specific fields; the `role_taker` field is inherited |
| Construct a role | `MyRole(role_taker=taker_instance, ...)` — always explicit |
| Check whether a taker has a role | `Role.has_role(taker, MyRole)` |
| Retrieve roles of a type | `Role.roles_for(taker, MyRole)` |
| Retrieve all roles | `Role.roles_for(taker)` |
| Access the immediate role taker | `role.role_taker` |
| Access the root non-role entity | `role.root_persistent_entity` |
| Check two objects are the same entity | `IsSameSemanticEntity(a, b)` (sees through role chains) |
| Find the root role-taker type | `MyRole.get_root_role_taker_type()` |
| Read taker attributes via role | Use the role instance directly — read delegation is automatic |
| Change a taker attribute | Assign through `role.role_taker` |

## When Not to Use the Role Pattern

- The new class changes the permanent, identifying properties of the original. A `Room` that becomes a `Kitchen` does
not affect any of the things that makes it a `Room`, such as its size or location. But an `egg` that is boiled, changes
its original properties like its state of matter from liquid to solid.
- The new class has its own independent persistent identity.
- You never need to relate the two objects back to a shared underlying entity.

---

## Decision Guide

```{mermaid}
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '20px', 'lineColor': '#333333', 'primaryColor': '#ddeeff', 'primaryTextColor': '#111111', 'primaryBorderColor': '#3a7abf', 'edgeLabelBackground': '#ffffff'}, 'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80, 'padding': 20}}}%%
flowchart TD
    Q1{"Should the wrapper and<br/>the original object represent<br/>the same underlying persistent entity?"}
    Q2{"Is this a temporary<br/>contextual situation, or does it<br/>change persistent identity or properites?"}
    ROLE["<b>Role[T]</b><br/>Same-entity via IsSameSemanticEntity, role registry,<br/>optional chaining."]
    SUBCLASS["<b>Subclass</b><br/>The new class has its own<br/>persistent identifying properties."]
    ASSOCIATION["<b>Association</b><br/>The new class is a different kind<br/>of thing from the original."]

    Q1 -->|Yes| Q2
    Q1 -->|No| ASSOCIATION
    Q2 -->|Temporary / contextual| ROLE
    Q2 -->|Persistent identity change| SUBCLASS
```

---

## Learn More

- **Developer guide**: {doc}`developer/role` — design decisions, architecture, and extension points.
- **API reference**: `krrood.patterns.role` — `Role`; `krrood.patterns.role_registry` —
  `RoleRegistry`; `krrood.patterns.role_predicates` — `IsSameSemanticEntity`.
- **Source**: `krrood/src/krrood/patterns/role.py`
- **Tests**: `test/krrood_test/test_patterns/test_role.py` and the dataset under
  `test/krrood_test/dataset/role_and_ontology/`.
- **IDE tooling**: [Python Role Lens](https://plugins.jetbrains.com/plugin/32276-python-role-lens) (plugin id
  `io.github.abdelrhmanbassiouny.pyroles`) — a JetBrains plugin that displays a role's
  accessible (including delegated) attributes and lets you navigate to the classes that
  define them.
