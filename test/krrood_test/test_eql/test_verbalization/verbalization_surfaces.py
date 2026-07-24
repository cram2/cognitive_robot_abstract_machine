"""
Committed verbalization surfaces: the sentence each covered symbolic callable renders.

This is the snapshot for :mod:`test_verbalization_surfaces` -- one
:class:`~krrood.entity_query_language.verbalization.surface_verification.VerbalizationSurface` per
covered symbolic callable, referencing the class itself (so a rename or removal breaks this import)
and its approved sentence. Update an entry when an intentional wording change makes the surface-match
test print a new sentence.
"""

from __future__ import annotations

from typing_extensions import Tuple

from krrood.entity_query_language.factories import (
    AttributeOwnerClass,
    IsClass,
    IsSubclass,
    NodeChildren,
    NodeDescendants,
    NodeId,
    NodeParents,
    NodeType,
    RuntimeType,
)
from krrood.entity_query_language.predicate import (
    HasType,
    HasTypes,
    Is,
    Length,
    SymbolicCallable,
)
from krrood.entity_query_language.verbalization._example_domain import (
    IsReachable,
    WorksIn,
)
from krrood.entity_query_language.testing.surface_verification import (
    VerbalizationSurface,
)
from krrood.inheritance_path_length import InheritancePathLength
from krrood.patterns.role_predicates import IsSameSemanticEntity

SURFACES: Tuple[VerbalizationSurface, ...] = (
    VerbalizationSurface(
        AttributeOwnerClass, "the attribute owner class of an Attribute"
    ),
    VerbalizationSurface(IsClass, "an object is a class"),
    VerbalizationSurface(IsSubclass, "a subclass is a subclass of a parent or parents"),
    VerbalizationSurface(NodeChildren, "the node children of a CanBehaveLikeAVariable"),
    VerbalizationSurface(
        NodeDescendants, "the node descendants of a SymbolicExpression"
    ),
    VerbalizationSurface(NodeId, "the node id of a SymbolicExpression"),
    VerbalizationSurface(NodeParents, "the node parents of a SymbolicExpression"),
    VerbalizationSurface(NodeType, "the node type of a Selectable"),
    VerbalizationSurface(RuntimeType, "the runtime type of an object"),
    VerbalizationSurface(HasType, "a variable is of type Integer"),
    VerbalizationSurface(HasTypes, "a variable is of type Integer or Text"),
    VerbalizationSurface(Is, "a first entity is the same object as a second entity"),
    VerbalizationSurface(Length, "the length of an iterable"),
    VerbalizationSurface(IsReachable, "a location is reachable for a body"),
    VerbalizationSurface(WorksIn, "an employee works in a department"),
    VerbalizationSurface(
        InheritancePathLength,
        "the inheritance path length between a child class and a parent class",
    ),
    VerbalizationSurface(
        IsSameSemanticEntity, "an entity 1 is the same entity as an entity 2"
    ),
)
