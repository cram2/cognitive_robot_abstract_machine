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

The Role pattern lets an existing object take on a new semantic context — adding context-specific
attributes and behaviour — without changing the original object's type or identity. A role is an
**ordinary object with its own identity**: it is equal only to itself. When
you need to ask whether two objects refer to the same underlying entity, use the `IsSameEntity`
predicate, which sees through role chains to the root entity.

This guide uses a university domain as its running example. A `Person` is a persistent entity
with a name. Over time that person may become a `CEO` or a `Professor`. These are roles: they
add context-specific data, but the person's identity never changes.

## Defining a Role Taker

A role taker is any dataclass you define.

```{code-cell} ipython3
from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import List

from krrood.entity_query_language.predicate import Symbol
from krrood.patterns.role_predicates import IsSameEntity


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
The role taker is stored in a single field that you mark with `role_taker_field()`.

```{code-cell} ipython3
from krrood.patterns.role import Role, role_taker_field


@dataclass(eq=False)
class CEO(Role[Person]):
    """The role of being the chief executive of a company."""

    person: Person = role_taker_field()
    head_of: Company = None
```

Notice that `CEO` inherits from `Role[Person]` — it does **not** inherit from
`Person`. The role and its taker are entirely separate classes connected only through the
`role_taker_field`.

## Constructing a Role

Role construction is explicit. You pass the role taker as a keyword argument to the role's
constructor, just like any other dataclass field.

```{code-cell} ipython3
acme = Company(name="ACME")
ceo = CEO(person=alice, head_of=acme)
print(ceo)
```

The role registers itself with the `RoleRegistry` during `__post_init__`, which is what enables role lookup.

## Distinct Identity

A role is a distinct object from its role taker: they do not compare equal and do not share a
hash, so a role and its taker — and multiple roles on one taker — stay separate in sets and
dictionaries.

```{code-cell} ipython3
print("alice == ceo:", alice == ceo)
print("alice is ceo:", alice is ceo)
print("set size:", len({alice, ceo}))
```

To ask whether two objects refer to the same underlying entity, use the `IsSameEntity`
predicate. It unwraps any role to its root entity before comparing, so a role and its taker
count as the same entity:

```{code-cell} ipython3
print("IsSameEntity(alice, ceo):", bool(IsSameEntity(alice, ceo)))
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

    person: Person = role_taker_field()
    teaches: List[Course] = field(default_factory=list, kw_only=True)


cs_101 = Course(name="CS 101")
professor = Professor(person=alice, teaches=[cs_101])

print("ceo == professor:", ceo == professor)
print("IsSameEntity(ceo, professor):", bool(IsSameEntity(ceo, professor)))
print("IsSameEntity(professor, alice):", bool(IsSameEntity(professor, alice)))
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
ceo_of_globex = CEO(person=alice, head_of=globex)

# Both CEO roles are kept and returned
same_type_roles = Role.roles_for(alice, CEO)
print("number of CEO roles:", len(same_type_roles))
print("companies headed:", sorted(role.head_of.name for role in same_type_roles))

# They are distinct objects, yet the same underlying entity
print("ceo is ceo_of_globex:", ceo is ceo_of_globex)
print("IsSameEntity(ceo, ceo_of_globex):", bool(IsSameEntity(ceo, ceo_of_globex)))
```

## Role Chaining

A role's role taker can itself be a role. This lets you layer multiple levels of semantic
context onto the same entity.

```{code-cell} ipython3
@dataclass(eq=False)
class Representative(Role[CEO]):
    """The role of a CEO acting as a representative of their company."""

    ceo: CEO = role_taker_field()
    represents: Company = None


rep = Representative(ceo=ceo, represents=acme)

print("rep == ceo:", rep == ceo)
print("IsSameEntity(rep, ceo):", bool(IsSameEntity(rep, ceo)))
print("IsSameEntity(rep, alice):", bool(IsSameEntity(rep, alice)))
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

## The `from_role_taker` Factory

When you do not have extra role-specific data to supply at construction time, you can use the
`from_role_taker` class method. It constructs the role by passing only the role taker.

```{code-cell} ipython3
# Equivalent to CEO(person=alice) when no role-native fields need values up front
new_ceo = CEO.from_role_taker(alice)
print("new_ceo:", new_ceo)
```

## Available Features

| Feature | How to use it |
|---|---|
| Define a role class | Inherit from `Role[T]`, mark the taker field with `role_taker_field()` |
| Construct a role | `MyRole(taker_field=taker_instance, ...)` — always explicit |
| Construct without extra data | `MyRole.from_role_taker(taker_instance)` |
| Check whether a taker has a role | `Role.has_role(taker, MyRole)` |
| Retrieve roles of a type | `Role.roles_for(taker, MyRole)` |
| Retrieve all roles | `Role.roles_for(taker)` |
| Access the immediate role taker | `role.role_taker` |
| Access the root non-role entity | `role.root_persistent_entity` |
| Check two objects are the same entity | `IsSameEntity(a, b)` (sees through role chains) |
| Find the root role-taker type | `MyRole.get_root_role_taker_type()` |
| Read taker attributes via role | Use the role instance directly — read delegation is automatic |
| Change a taker attribute | Assign through `role.role_taker` |

## When to Use the Role Pattern

- An object needs context-specific attributes that do not logically belong to the original type.
- The additional context is temporary or situational — the entity may gain and lose the role
  over its lifetime.
- The same entity can play multiple semantic roles simultaneously.
- Any part of the system must be able to ask "does this entity have role X?" without scanning
  all instances.
- The role and the original object should be recognizable as the same underlying entity (via
  `IsSameEntity`) while remaining distinct objects.

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
    ROLE["<b>Role[T]</b><br/>Same-entity via IsSameEntity, role registry,<br/>optional chaining."]
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
- **API reference**: `krrood.patterns.role` — `Role`, `role_taker_field`,
  `RoleTakerFieldNotFound`; `krrood.patterns.role_registry` — `RoleRegistry`;
  `krrood.patterns.role_predicates` — `IsSameEntity`.
- **Source**: `krrood/src/krrood/patterns/role.py`
- **Tests**: `test/krrood_test/test_patterns/test_role.py` and the dataset under
  `test/krrood_test/dataset/role_and_ontology/`.
- **IDE tooling**: [Python Role Lens](https://plugins.jetbrains.com/plugin/32276-python-role-lens) (plugin id
  `io.github.abdelrhmanbassiouny.pyroles`) — a JetBrains plugin that displays a role's
  accessible (including delegated) attributes and lets you navigate to the classes that
  define them.
