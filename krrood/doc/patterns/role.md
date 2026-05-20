# Role Pattern

## What is it?

The **Role** pattern lets an existing object take on a new semantic context — adding context-specific
attributes and behaviour — while **remaining the same entity**. A role and its role taker share
identity: they compare equal, have the same hash, and point to each other through a role registry.

---

## Quick example

### Defining a role taker

```python
from dataclasses import dataclass, field

@dataclass(eq=False)
class Room(SemanticAnnotation):
    floor: Floor = field(kw_only=True)
```

### Defining a role

Inherit from `Role[T]` (where `T` is the role taker type), add the role taker as a field, and
implement `role_taker_attribute()` to point to that field:

```python
from krrood.patterns.role import Role
from krrood.entity_query_language.factories import variable_from


@dataclass(eq=False)
class Kitchen(Role[Room], Room): # The Role should also inherit from the role taker type (only for the IDE support).
    room: Room = field(kw_only=True) # the role taker should be a keyword-only argument
    utilities: list[str] = field(default_factory=list)

    @classmethod
    def role_taker_attribute(cls) -> Attribute[Room]:
        return variable_from(cls).room
```

### Using the role

```python
room = Room(floor=ground_floor)
kitchen = Kitchen(room=room)

# Identity is shared
assert kitchen == room
assert hash(kitchen) == hash(room)

# Role registry: the room knows about the kitchen
assert Role.roles_for(room, Kitchen)[0] is kitchen

# Room attributes are accessible directly on the kitchen (via delegation)
kitchen.floor   # → room.floor

# Role-specific attributes live on the kitchen
kitchen.utilities
```

---

## Examples from semantic_annotations

### Room roles — Kitchen, Bedroom, Bathroom, LivingRoom

A `Room` is a physical area with spatial bounds and a floor. Its semantic purpose — *kitchen*,
*bedroom*, *bathroom*, *living room* — is a temporary, contextual role. The same room could be a
kitchen today and converted to a bedroom tomorrow. The room's identity never changes; only its
semantic role does.

The `semantic_digital_twin` package defines four room roles, all following the same pattern:

```python
@dataclass(eq=False)
class Kitchen(Role[Room], Room):
    room: Room = field(kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> Attribute[Room]:
        return variable_from(cls).room

@dataclass(eq=False)
class Bedroom(Role[Room], Room):
    room: Room = field(kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> Attribute[Room]:
        return variable_from(cls).room

@dataclass(eq=False)
class Bathroom(Role[Room], Room):
    room: Room = field(kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> Attribute[Room]:
        return variable_from(cls).room

@dataclass(eq=False)
class LivingRoom(Role[Room], Room):
    room: Room = field(kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> Attribute[Room]:
        return variable_from(cls).room
```

A single `Room` can take on multiple roles simultaneously — a studio apartment's room might be a
`Kitchen`, `Bedroom`, and `LivingRoom` all at once:

```python
room = Room(floor=ground_floor)
kitchen = Kitchen(room=room)
bedroom = Bedroom(room=room)
living_room = LivingRoom(room=room)

# All roles share identity with the room
assert kitchen == bedroom == living_room == room
assert hash(kitchen) == hash(bedroom) == hash(living_room) == hash(room)

# The room's registry sees all active roles
assert len(Role.roles_for(room, Room)) == 4

# Room attributes are accessible on every role (via DelegatorForRoom)
kitchen.floor       # → room.floor
bedroom.floor       # → room.floor
living_room.floor   # → room.floor
```

This is the key insight of the Role pattern: the physical entity (`Room`) has a stable identity,
while its semantic roles come and go. Accessing of role-taker attributes is direct.

---

## Examples from ontology

### Person, CEO, Professor — parallel roles on the same entity

In an ontology, a `Person` is a persistent entity with a name and identity. Over time, that person
may take on various roles: they can become a `CEO` (temporarily heading a company), a `Professor`
(teaching courses), or both simultaneously. These roles are not permanent properties of the person —
they are contextual relationships that come and go.

```python
@dataclass(eq=False)
class Person(Symbol):
    name: str

@dataclass(eq=False)
class CEO(Role[Person], Person):
    person: Person = field(kw_only=True)
    head_of: Company = None

    @classmethod
    def role_taker_attribute(cls) -> Attribute[Person]:
        return variable_from(cls).person

@dataclass(eq=False)
class Professor(Role[Person], Person):
    person: Person = field(kw_only=True)
    teacher_of: List[Course] = field(default_factory=list)

    @classmethod
    def role_taker_attribute(cls) -> Attribute[Person]:
        return variable_from(cls).person
```

A single person can hold both roles at once:

```python
alice = Person(name="Alice")
ceo = CEO(person=alice, head_of=acme_corp)
prof = Professor(person=alice, teacher_of=[cs101, math201])

# Both roles share identity with Alice
assert ceo == prof == alice
assert hash(ceo) == hash(prof) == hash(alice)

# Alice knows about all her roles
assert Role.roles_for(alice, CEO)[0] is ceo
assert Role.roles_for(alice, Professor)[0] is prof

# Each role carries its own context
ceo.head_of         # → acme_corp
prof.teacher_of     # → [cs101, math201]
```

### CEO, Representative — role chaining

A role's role taker can itself be a role, creating a **role chain**. A `CEO` can temporarily act as a
`Representative` for a company, adding another layer of context on top:

```python
@dataclass(eq=False)
class Representative(Role[CEO], CEO):
    ceo: CEO = field(kw_only=True)
    represents: Company = None

    @classmethod
    def role_taker_attribute(cls) -> Attribute[CEO]:
        return variable_from(cls).ceo
```

```python
alice = Person(name="Alice")
ceo = CEO(person=alice, head_of=acme_corp)
rep = Representative(ceo=ceo, represents=acme_corp)

# All three are the same entity
assert rep == ceo == alice
assert hash(rep) == hash(ceo) == hash(alice)

# The root entity sees the full chain
assert Role.roles_for(alice, CEO)[0] is ceo
assert Role.roles_for(alice, Representative)[0] is rep

# The chain is transparent: each role taker sees all roles
Role.roles_for(ceo, Representative)[0] is rep   # CEO knows about its Representative role
```

### Why roles, not subclasses?

A `CEO` is not a subclass of `Person` — a person does not *become* a different kind of thing when
they take a job. They gain temporary properties (`head_of`) and relationships that may later be
removed, but their identity as a person persists. Subclassing would imply that a `CEO` is a
permanently different kind of entity with distinct identifying properties, which is not how these
concepts work in the real world.

Similarly, `Kitchen` is not a subclass of `Room` — a room can be a kitchen one year and a bedroom
the next. If `Kitchen` were a subclass, that transition would be impossible to model without
replacing the room object entirely. A Role captures the temporary, contextual nature of these
concepts.

---

## Identity sharing

When a `Role` is constructed, `__post_init__` automatically registers it in the role taker's `roles`
dict and links the two objects in the symbol graph. From that point on:

- `role == role_taker` (and vice versa)
- `hash(role) == hash(role_taker)`
- `role_taker.roles[RoleClass]` returns the role instance

This means any collection, set, or dictionary keyed by the original object automatically reflects the
role, and code that receives either object can navigate to the other:

```python
room = Room(floor=ground_floor)
kitchen = Kitchen(room=room)

# Both references refer to "the same thing" in sets and dicts
s = {room}
assert kitchen in s

# Navigate from either direction
kitchen.role_taker         # → room
room.roles[Kitchen]        # → kitchen
```

---

## Role chaining

A role's role taker can itself be a role. This is called **role chaining** and lets you layer multiple
levels of semantic context onto a single physical entity.

As shown in the ontology examples above, a `Person` can become a `CEO`, and that `CEO` can become a
`Representative` — three layers of context, one identity. The chain can be arbitrarily deep: a
`Representative` could become a `Delegate`, and so on.

All objects in the chain are equal and share the same hash:

```python
assert rep == ceo == alice
assert hash(rep) == hash(ceo) == hash(alice)

# Each role taker in the chain sees all roles
assert Role.roles_for(alice, CEO)[0] is ceo
assert Role.roles_for(alice, Representative)[0] is rep
```

The `Role` class provides helpers to navigate chains:

| Method / Property                                | Description                                                      |
|--------------------------------------------------|------------------------------------------------------------------|
| `role.role_taker`                                | The immediate role taker (may itself be a role).                 |
| `Role.get_root_role_taker_type()`                | Walks the chain to find the non-Role base type.                  |
| `role.role_taker_roles`                          | All roles registered on the root role taker.                     |
| `Role.has_role(entity, RoleType)`                | Checks if `entity` (or anything equal to it) has the given role. |
| `Role.roles_for(entity, RoleType)`               | Returns all roles of a given type.                               |
---

## How it works

### `__post_init__`

When a role is constructed, `Role.__init__` runs automatically. It:

1. Retrieves the role taker from the declared field (`role_taker_attribute_name()`).
2. Updates the symbol graph to link the role and role taker.

### `role_taker_attribute()`

This abstract classmethod must be implemented by every role subclass. It returns a symbolic
`Attribute` reference to the role-taker field. This is how `Role` knows which field holds the role taker:

```python
@classmethod
def role_taker_attribute(cls) -> Attribute[Room]:
    return variable_from(cls).room
```

### Shared fields

If the role class and the role taker class share a common base (e.g. both inherit `Symbol`), the
shared fields — such as a persistent `id` — are not duplicated. The role's fields for those
attributes are set to `init=False` and delegate to the role taker.

---

## When to use the Role pattern

- An object needs **context-specific attributes** that do not logically belong to the original type
  but are tightly coupled to a specific usage of that object.
- The additional context is **temporary or situational** — the same entity may gain and lose this
  role over its lifetime (a room that is a kitchen today becomes a bedroom tomorrow; a person becomes
  a CEO and later steps down).
- The same physical entity can play **multiple semantic roles simultaneously** (a room that is both
  a kitchen and a living room; a person who is both a CEO and a professor).
- The extended object and the original object must be **considered the same entity** throughout the
  system (same hash, equality, shared identity in collections).
- You want a **role registry**: any part of the system should be able to ask "does this room have a
  kitchen role?" without scanning all kitchen instances.

## When NOT to use the Role pattern

- **The new class changes persistent, identifying properties of the original** — if a `Fruit` could
  not exist without being an `Apple`, or if being a `Apple` fundamentally altered what the fruit *is*
  rather than what it *does*, subclassing is the right choice.
- **The new class has its own persistent identity** — if the concept is a permanently different kind
  of thing that exists independently to its parent class (e.g., a `MustardBottle` is a kind of `Bottle` that has a permenant
  shape and functionality that identifies it from any other borrle.), it is a subclass, not a role.
- **You do not need identity sharing**.
- **You need more than one role of the same type at the same time** — the `roles` dict is keyed by
  type, so only one instance of each role type per role taker is supported.
- **The role taker type changes after construction** — `role_taker` is a `cached_property`; the role
  taker is fixed.

---

## Decision guide

```{mermaid}
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '20px', 'lineColor': '#333333', 'primaryColor': '#ddeeff', 'primaryTextColor': '#111111', 'primaryBorderColor': '#3a7abf', 'edgeLabelBackground': '#ffffff'}, 'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80, 'padding': 20}}}%%
flowchart TD
    Q1{"Should the wrapper and<br/>the original object be<br/>the same entity?"}
    Q2{"Is this a temporary<br/>contextual role, or does it<br/>change persistent identity?"}
    ROLE["<b>Role[T]</b><br/>Identity-sharing, role registry,<br/>optional chaining."]
    SUBCLASS["<b>Subclass / Inheritance</b><br/>The new class has its own<br/>persistent identifying properties."]
    ASSOCIATION["<b>Association</b><br/>The new class is a different kind<br/>of thing from the original."]

    Q1 -->|Yes| Q2
    Q1 -->|No| Association
    Q2 -->|Temporary / contextual| ROLE
    Q2 -->|Persistent identity change| SUBCLASS
```

---