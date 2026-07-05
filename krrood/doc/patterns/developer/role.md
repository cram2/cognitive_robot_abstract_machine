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

# Role Pattern — Developer Guide

This document describes the design goals, architectural decisions, and extension points of the
Role pattern as implemented in `krrood.patterns.role`.

## Architectural Goals and Constraints

This guide assumes the {doc}`user guide <../role>`, which introduces what a role is and when to
choose it over subclassing or association. The design optimises for four properties:

1. **Distinct identity, explicit equivalence.** A role is an ordinary object with its own
   identity (equal only to itself), so multiple roles — even of the same type — on one taker stay
   distinct. "Same underlying entity?" is answered explicitly by the `IsSameEntity` predicate
   rather than by overloading `==`/`hash`.
2. **Explicit, auditable construction.** The system that creates a role must pass the role taker
   explicitly. There must be no implicit or automatic role creation.
3. **Pure composition, not inheritance.** A role class must not inherit from its role taker type.
   Role membership is expressed through registry queries, not `isinstance` checks.
4. **Transparent attribute reads.** A consumer of a role that does not know which attributes belong
   to the role and which belong to the taker should be able to read all of them through the role
   instance without special handling. Writes are deliberately not transparent: an assignment always
   targets the role, so changing the taker is an explicit operation through `role.role_taker`. This
   keeps writes unambiguous and prevents mutating a shared entity as a side effect of writing
   through one of its roles.

These goals shape every significant decision in the implementation.

## Architecture Overview

The implementation centres on three collaborating components:

**`Role[T]`** (`role.py`) is the base dataclass that every role class inherits. It provides
identity-based equality/hashing (`__hash__`, `__eq__`), attribute delegation (`__getattr__`,
`__setattr__`), and all querying class methods. It inherits from `Generic[T], SubClassSafeGeneric` so that
fields whose annotation uses the generic parameter are rewritten to the bound concrete type on each
role subclass (keeping generic roles introspectable by the class diagram), and from `Symbol` so a
role participates in the class diagram and ORM like any other entity. A role sets
`_cache_instances_ = False`, so role *instances* are not registered as nodes in the `SymbolGraph`:
membership uses the `RoleRegistry` and persistence uses the role's own ORM mapping, so a role never
needs to be a graph node. Role *classes* still take part in the class diagram, which is built from
`Symbol` subclasses, not from cached instances.

**`role_taker`** is the keyword-only field, declared once on `Role[T]` and inherited by every role
class, that holds the entity the role is attached to (see *Fixed-Name Discovery of the Role Taker*).

**`RoleRegistry`** (`role_registry.py`) is the runtime inverse index from a taker to its roles that
backs all role lookup (see *The Role Registry*).

## Key Design Decisions and Rationale

### Pure Composition: No Inheritance from the Taker Type

The role class does not inherit from the taker class.

The reason is that inheritance implies substitutability: if `CEO` inherited from `Person`, every
`isinstance(ceo, Person)` check would return `True`, and every API that accepted a `Person`
would silently accept a `CEO`. This causes hidden coupling between systems that should not need
to know about each other's roles. Role membership must be queried explicitly through
`Role.has_role` or `Role.roles_for`, which makes the dependency visible.

Pure composition also keeps the role independent of the taker's construction logic.

### No `__init__` Manipulation

The `Role` class uses the standard dataclass `__post_init__` hook for all post-construction work.
There is no override of `__init__`, no metaclass manipulation of `__init__`, and no
`InitVar`-based workarounds.

This means that the constructor signature of every concrete role class is exactly what
`@dataclass` generates from its fields. A caller can always read a role class's field
declarations and know exactly what arguments its constructor accepts, with no hidden
transformations.

### Fixed-Name Discovery of the Role Taker

The role taker is the inherited `role_taker` field, so it is identified by its fixed name
(`role_taker_field_name()`) on any `Role` subclass. The taker *type* comes from the role's
`Role[Taker]` generic argument, resolved by `get_role_taker_type()` (a bounded `TypeVar`
resolves to its bound). The class diagram derives the role-taker association from that generic
argument because the inherited field's own annotation is the unbound generic parameter.

### Distinct Identity and the `IsSameEntity` Predicate

Each role is equal only to itself and distinct from its taker and from
sibling roles. This keeps two concerns separate: *object identity* (am I this exact object?) and
*semantic equivalence* (do we refer to the same underlying entity?). The latter is expressed explicitly by the `IsSameEntity` predicate in
`krrood/src/krrood/patterns/role_predicates.py`, which unwraps each operand to its
`root_persistent_entity` (walking the taker chain to the non-`Role` object) and compares the
roots by identity. Two roles at different levels of a chain, and a role versus its root taker,
are therefore *not* `==` but *are* `IsSameEntity`. A direct consequence is that multiple roles,
even of the same type, on one taker stay distinct in sets and dicts.

### The Role Registry

Role lookup (checking whether a taker has a role, retrieving roles of a given type) is a lookup in
`RoleRegistry`, an inverse index from a taker to the roles attached to it. The role registers itself
during `__post_init__`. Entries are keyed by taker identity and held through weak references, so the
index neither keeps roles or takers alive nor conflates two distinct takers that compare equal.

A role is indexed under every taker in its chain, not only its immediate taker. When the taker is
itself a role, the new role is therefore reachable from every entity beneath it, so a query from any
point in the chain resolves to all roles layered above it. The opposite direction (a role's takers)
needs no index: `all_role_takers` walks the taker chain directly.

The registry depends on nothing in `krrood.class_diagrams` or `krrood.symbol_graph`, so a role's
membership behaviour is independent of whether its class has been added to the class diagram.

## Attribute Access

The mechanics (read delegation via `__getattr__`, write-local via `__setattr__`) are covered in the
{doc}`user guide <../role>`. The design decision worth recording here is the deliberate **asymmetry**:
reads are transparent so a consumer need not know which attributes belong to the role and which to
the taker, but writes are not — an assignment always targets the role and rejects any undeclared
name with `RoleAttributeNotDeclaredError`. That asymmetry is what prevents a write through a role
from mutating the shared taker as a side effect or silently shadowing one of its attributes; changing
the taker is therefore the explicit `role.role_taker` operation rather than an implicit consequence
of assignment.

## Role-Taker Associations and Property Descriptors

For every role, the class diagram infers an `AssociationThroughRoleTaker` (in
`krrood.class_diagrams.class_diagram`): an association that connects the role class to each target
reachable through its role taker's associations, transitively along the taker chain. This is how a
role participates in the class model in the associations its taker already has, without inheriting
from the taker.

The property-descriptor module (`krrood.ontomatic.property_descriptor`) builds on this model:
`PropertyDescriptor.get_superproperties_associations` and `PropertyDescriptorRelation` resolve
property and super-property relations across role takers through these associations (using
`ClassDiagram.get_outgoing_associations_with_condition` and
`AssociationThroughRoleTaker.get_original_source_instance_given_this_relation_source_instance`). The
property-descriptor code therefore depends on the role-taker association model defined here, so the
two evolve together: a change to how roles project their takers' associations is reflected in both
the class diagram and the property descriptors.

## Source References

- `krrood/src/krrood/patterns/role.py` — `Role` (and its inherited `role_taker` field)
- `krrood/src/krrood/patterns/role_registry.py` — `RoleRegistry`
- `krrood/src/krrood/patterns/exceptions.py` — `DelegatedFactoryMethodError`,
  `RoleAttributeNotDeclaredError`
- `krrood/src/krrood/patterns/subclass_safe_generic.py` — `SubClassSafeGeneric`
- `krrood/src/krrood/symbol_graph/symbol_graph.py` — `Symbol`
- `test/krrood_test/test_patterns/test_role.py` — behavioural tests for the role pattern
- `test/krrood_test/dataset/role_and_ontology/university_ontology_like_classes_without_descriptors.py`
  — canonical fixture for role pattern tests
