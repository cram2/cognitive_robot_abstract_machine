from __future__ import annotations

import datetime
from dataclasses import dataclass

from typing_extensions import List, Optional

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import (
    Attribute,
    Call,
    FlatVariable,
    Index,
    MappedVariable,
)
from krrood.entity_query_language.core.variable import Literal, Variable
from krrood.entity_query_language.verbalization.fragments.source_reference import (
    SourceReference,
)


@dataclass(frozen=True)
class PathStep:
    """
    One hop of a navigation chain — a display name and the source reference it links to.

    Replaces a bare ``(name, source_reference)`` tuple so the two halves are named (``.name`` /
    ``.source_reference``) rather than positional.
    """

    name: str
    """The display text for this hop (e.g. ``"amount"``, ``"handle[0]"``, ``"()"``)."""

    source_reference: Optional[SourceReference] = None
    """The attribute's source reference, or ``None`` for composite / index / call hops."""


def walk_chain(
    expression: SymbolicExpression,
) -> tuple[list[MappedVariable], SymbolicExpression]:
    """
    Walk a ``MappedVariable`` chain outward-first.

    Example: for ``robot.arm.joint`` the chain is
    ``[Attribute('joint'), Attribute('arm')]`` and root is the ``robot`` variable.

    :param expression: Any expression; non-``MappedVariable`` expressions return an empty
        chain with *expression* as the root.
    :return: Tuple ``(chain, root)`` — the access path (root-adjacent first, terminal last)
        and the chain base.
    """
    if isinstance(expression, MappedVariable):
        return list(expression._access_path_), expression._chain_root_
    return [], expression


def is_temporal(expression: SymbolicExpression) -> bool:
    """
    :param expression: Any EQL expression.
    :return: ``True`` when *expression* denotes a ``datetime`` value or variable.
    """
    if isinstance(expression, Literal):
        return isinstance(expression._value_, datetime.datetime)
    if isinstance(expression, Variable):
        return getattr(expression, "_type_", None) is datetime.datetime
    if isinstance(expression, MappedVariable):
        chain, _ = walk_chain(expression)
        return bool(chain) and getattr(chain[-1], "_type_", None) is datetime.datetime
    return False


def chain_root(expression: SymbolicExpression) -> SymbolicExpression:
    """
    :param expression: Any expression.
    :return: The non-``MappedVariable`` root of *expression* (the deepest non-``MappedVariable``
        node, or *expression* itself when it is not a ``MappedVariable``), found without building
        the full chain list.
    """
    return (
        expression._chain_root_
        if isinstance(expression, MappedVariable)
        else expression
    )


def chain_ends_in_boolean_attribute(chain: list[MappedVariable]) -> bool:
    """
    :param chain: A walked chain (root-adjacent first).
    :return: ``True`` when the walked *chain* ends in a ``bool``-typed attribute (the
        predicative *"<navigation> is <attribute>"* form).
    """
    return bool(chain) and isinstance(chain[-1], Attribute) and chain[-1]._type_ is bool


def build_path_parts(chain: list[MappedVariable]) -> list[PathStep]:
    """
    Convert a walked chain into :class:`PathStep` hops.

    Merging rules:

    * Consecutive ``Attribute → Index`` pairs are merged into ``"attribute[key]"`` with no source
      ref (composite indexed access has no clean single-symbol anchor).
    * Standalone ``Index`` nodes appear as ``"[key]"`` with no source ref.
    * ``Call`` nodes appear as ``"()"`` with no source ref.
    * ``FlatVariable`` nodes are skipped.

    :param chain: Outermost-first chain list.
    :return: Ordered list of :class:`PathStep`, outermost attribute first.
    """
    parts: list[PathStep] = []
    i = 0
    while i < len(chain):
        node = chain[i]
        if isinstance(node, Attribute):
            name = node._attribute_name_
            owner = node._owner_class_
            ref: Optional[SourceReference] = SourceReference.for_attribute(owner, name)
            while i + 1 < len(chain) and isinstance(chain[i + 1], Index):
                i += 1
                name += f"[{repr(chain[i]._key_)}]"
                ref = None  # composite indexed access has no clean single-line anchor
            parts.append(PathStep(name, ref))
        elif isinstance(node, Index):
            parts.append(PathStep(f"[{repr(node._key_)}]", None))
        elif isinstance(node, Call):
            parts.append(PathStep("()", None))
        elif isinstance(node, FlatVariable):
            pass
        i += 1
    return parts


@dataclass(frozen=True)
class ChainAnalysis:
    """
    A ``MappedVariable`` chain analysed once into the values its rendering needs: the walked
    chain, its root, the display path-parts, and whether it ends in a boolean attribute.
    """

    chain: List[MappedVariable]
    """The access path, root-adjacent first."""

    root: SymbolicExpression
    """The chain root (first non-``MappedVariable`` node)."""

    parts: List[PathStep]
    """The display path-parts."""

    is_boolean_terminal: bool
    """``True`` when the chain ends in a ``bool``-typed attribute (predicative form)."""

    @classmethod
    def of(cls, expression: SymbolicExpression) -> ChainAnalysis:
        """
        :param expression: A ``MappedVariable`` chain, or any root expression.
        :return: The chain analysis of *expression*.
        """
        chain, root = walk_chain(expression)
        return cls(
            chain=chain,
            root=root,
            parts=build_path_parts(chain),
            is_boolean_terminal=chain_ends_in_boolean_attribute(chain),
        )
