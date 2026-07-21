"""
Committed verbalization surfaces: the sentence each covered symbolic callable renders.

This is the snapshot for :mod:`test_verbalization_surfaces` -- one
:class:`VerbalizationSurface` per covered symbolic callable, referencing the class
itself (so a rename or removal breaks this import) and its approved sentence. Update an
entry when an intentional wording change makes the surface-match test print a new
sentence.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Tuple, Type

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
from krrood.entity_query_language.verbalization.example_domain import (
    IsReachable,
    WorksIn,
)
from krrood.inheritance_path_length import InheritancePathLength
from krrood.patterns.role_predicates import IsSameSemanticEntity


@dataclass(frozen=True)
class VerbalizationSurface:
    """
    One symbolic callable and the sentence it verbalizes to.
    """

    callable_class: Type[SymbolicCallable]
    """
    The symbolic function or predicate whose surface this records.
    """

    sentence: str
    """
    The approved sentence it renders with the test's placeholder operands.
    """


SURFACES: Tuple[VerbalizationSurface, ...] = (
    VerbalizationSurface(
        AttributeOwnerClass, "the attribute owner class of an Attribute"
    ),
    VerbalizationSurface(IsClass, "an object is a class"),
    VerbalizationSurface(IsSubclass, "object 1 is a subclass of object 2"),
    VerbalizationSurface(NodeChildren, "the node children of a CanBehaveLikeAVariable"),
    VerbalizationSurface(
        NodeDescendants, "the node descendants of a SymbolicExpression"
    ),
    VerbalizationSurface(NodeId, "the node id of a SymbolicExpression"),
    VerbalizationSurface(NodeParents, "the node parents of a SymbolicExpression"),
    VerbalizationSurface(NodeType, "the node type of a Selectable"),
    VerbalizationSurface(RuntimeType, "the runtime type of an object"),
    VerbalizationSurface(HasType, "an object is of type Integer"),
    VerbalizationSurface(HasTypes, "an object is of type Integer or Text"),
    VerbalizationSurface(Is, "object 1 is the same object as object 2"),
    VerbalizationSurface(Length, "the length of an object"),
    VerbalizationSurface(IsReachable, "an object is reachable"),
    VerbalizationSurface(WorksIn, "object 1 works in object 2"),
    VerbalizationSurface(
        InheritancePathLength,
        "the inheritance path length between object 1 and object 2",
    ),
    VerbalizationSurface(
        IsSameSemanticEntity, "object 1 is the same entity as object 2"
    ),
)
