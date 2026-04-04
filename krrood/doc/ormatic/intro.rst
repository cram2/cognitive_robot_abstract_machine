.. _ormatic-overview:

ORMatic Overview
========

ORMatic is a subpackage of KRROOD designed to integrate Python **dataclasses** with the **SQLAlchemy ORM**.
It operates by generating a SQLAlchemy declarative model layer, offering utilities for translating between
domain objects (dataclasses) and database representations (rows).

This approach promotes a separation between the domain model and the persistence mechanism, allowing developers
to concentrate on business logic while abstracting underlying database implementation and SQLAlchemy specifics.

Core Assumptions and Modeling Rules
-----------------------------------
To ensure unambiguous translation and an ergonomic generated ORM, ORMatic adheres to the following rules for dataclass modeling:

- **Protected Fields:** Any dataclass field name beginning with ``_`` is ignored for persistence. These protected fields are suitable for transient or derived runtime data.
- **Optional Types:** The only supported union is :py:class:`Optional[T] <typing.Optional>`. Other unions should be modeled using a shared superclass.
- **Collections:** Iterables must be non-optional and non-nested. For an "optional list," use a default factory that returns an empty collection.
- **Inheritance and Polymorphism:** Inheritance is supported and generates `joined-table <https://docs.sqlalchemy.org/en/20/orm/inheritance.html>`_ inheritance. Note that only the *first* base class in a multiple inheritance structure is considered for queries using abstract classes.
- **Type Discipline:** Dataclass fields require concrete, non-ambiguous type annotations. Prefer small dataclasses or value objects over primitive types when modeling relationships.

When dataclasses cannot conform to these patterns, the following alternatives are available:

- **Alternative Mapping:** Explicit mapping classes can be provided to control how a dataclass is persisted.
- **Type Decorator:** Custom type decorators can be supplied for specialized value types.
Detailed information on these options is provided in the :ref:`alternative_mapping` section.

What ORMatic Generates
----------------------
Execution of ORMatic on a set of classes produces a module containing:

- ``Base``: An SQLAlchemy :py:class:`DeclarativeBase <sqlalchemy.orm.DeclarativeBase>` that manages metadata and a ``type_mappings`` registry for custom types.
- ``*DAO`` Classes: One class per dataclass (and per explicit mapping). Each is a SQLAlchemy declarative model that includes:
  - Columns for fields with built-in or custom types
  - Foreign keys and relationships inferred from nested dataclasses and collections
  - ``__mapper_args__`` for inheritance and polymorphic configuration

ORMatic analyzes the dataclass structure to identify scalar fields, one-to-one, one-to-many, and many-to-many associations. This information is then used to generate a full, decoupled SQLAlchemy declarative layer via a Jinja template.


ORMatic Compatible Class Pattern
--------------------------------

If you want to generally know how to write classes that work out of the box with ORMatic, follow these simple rules:
 1. Everything is a dataclass
 2. Never write you own ``__init__``.
 3. Apply side effects (e. g. setting back references) in the ``__post_init__``
 4. Use protected fields, those whose name begins with ``_`` only for computational state that doesnt need persistence (e. g. caching, indexing, ...)
 5. Don't use nested containers as public fields
 6. Don't use dicts as public fields
 7. Never use optional containers, use empty containers by default.

Persisting Objects
------------------------
- :py:func:`krrood.ormatic.dao.to_dao`: Converts a dataclass instance into its corresponding DAO object, including recursive conversion of nested elements.
- :py:func:`krrood.ormatic.dao.DataAccessObject.from_dao`: Converts a loaded DAO instance back into the original dataclass, including nested components and collections.