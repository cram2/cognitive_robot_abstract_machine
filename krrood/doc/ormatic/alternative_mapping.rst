.. _alternative_mapping:

Dealing with unmappable dataclasses
===================================

ORMatic can infer a mapping for a dataclass from its fields and relationships. This is the default behavior.
However, sometimes you may need to customize the mapping. There are two ways to do this:

Type Decorators
---------------
You can provide ORMatic a type mapping that maps a Python type to a `SQLAlchemy TypeDecorator <https://docs.sqlalchemy.org/en/20/core/custom_types.html>`_.
This decorator is then used as a type mapping every time ORMatic encounters the type.
I recommend using this approach when dealing with meshes, NumPy arrays, etc. Basically anything
that does not contain any fields that need to be referenced by other classes and/or are very messy to store in a
database.

Alternative Mapping (Explicit Control)
--------------------------------------
If a dataclass does not fit the standard rules or requires a more specialized schema, and a referencing of its fields
define an explicit mapping by subclassing :class:`krrood.ormatic.dao.AlternativeMapping`. The generic variable
has to be the class you create the alternative mapping for. This lets you:

- Override column names or shapes
- Introduce association tables explicitly
- Write reusable mappings that can be shared across multiple applications
- Filter information that should not persist

You also have to implement the methods :func:`krrood.ormatic.dao.AlternativeMapping.create_instance` and
:func:`krrood.ormatic.dao.AlternativeMapping.create_from_dao`.