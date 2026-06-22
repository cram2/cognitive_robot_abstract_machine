from __future__ import annotations

from dataclasses import replace

from typing_extensions import TYPE_CHECKING, Optional, Sequence

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.verbalization.fragments.base import Fragment
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    RenderOptions,
    RuleContext,
    PhraseRule,
    select,
)
from krrood.entity_query_language.verbalization.grammar.framework.registry import RULES
from krrood.entity_query_language.verbalization.exceptions import (
    UnverbalizableExpressionError,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.context import MicroplanningServices


def root_context(
    services: MicroplanningServices,
    rules: Sequence[PhraseRule],
    options: Optional[RenderOptions] = None,
) -> RuleContext:
    """
    Build the :class:`RuleContext` for one node — the **single** definition of the fold
    continuation (``recurse``), shared by :func:`fold` and the match assembler's root context, so a
    new render flag is one field on :class:`RenderOptions` rather than two duplicated lambdas.

    :param services: The pass-wide microplanning services.
    :param rules: The grammar to dispatch over.
    :param options: The render flags for this node (defaults to all reset).
    :return: The context whose ``child`` re-enters :func:`fold`.
    """
    return RuleContext(
        recurse=lambda child_node, child_options: fold(
            child_node, services, rules, child_options
        ),
        services=services,
        options=options if options is not None else RenderOptions(),
    )


def fold(
    node: SymbolicExpression,
    services: MicroplanningServices,
    rules: Optional[Sequence[PhraseRule]] = None,
    options: Optional[RenderOptions] = None,
) -> Fragment:
    """
    Verbalize *node* by dispatching it to its matching grammar rule and recursing — the single
    catamorphism (fold) over the EQL expression tree.

    A node carrying a pre-built binding override is returned directly, before any dispatch.
    When no rule covers the node, an ``UnverbalizableExpressionError`` is raised rather than
    degrading silently to the class name.

    The recursion is an F-algebra fold over the EQL algebra, with the grammar as the algebra
    (Meijer, Fokkinga & Paterson 1991, "Functional Programming with Bananas, Lenses, Envelopes
    and Barbed Wire"; Bird & de Moor 1997, "Algebra of Programming").

    :param node: Any EQL expression.
    :param services: The pass-wide microplanning services (and render configuration).
    :param rules: Grammar to dispatch over; defaults to the standard rule set.
    :param options: The render flags to build *node* under (defaults to all reset).
    :return: The fragment for *node*.
    :raises UnverbalizableExpressionError: when no grammar rule covers *node*.
    """
    rules = RULES if rules is None else rules

    node_id = getattr(node, "_id_", None)
    if node_id is not None:
        override = services.binding.binding_overrides.get(node_id)
        if override is not None:
            return override

    context = root_context(services, rules, options)

    rule = select(node, rules, context)
    if rule is None:
        raise UnverbalizableExpressionError(node=node)
    if rule.enters_query_scope:
        # The rule's construct is a query body: everything built inside sees query_depth >= 1,
        # so a nested Entity renders as a noun phrase. Declared on the rule (not pushed by hand
        # in assemblers) so the policy lives in one place. ``when`` already ran outside.
        with services.configuration.query_depth_scope():
            return _with_source(rule.build(node, context), node)
    return _with_source(rule.build(node, context), node)


def _with_source(fragment: Fragment, node: SymbolicExpression) -> Fragment:
    """
    Stamp *node* as the fragment's provenance, so later passes can follow it back to the read
    model. The innermost producer wins: a transparent wrapper (``An(Entity)``) returns its child's
    fragment, which already carries the child's (the ``Entity``'s) source, and is left untouched.

    :param fragment: The fragment a rule produced for *node*.
    :param node: The EQL node dispatched.
    :return: *fragment* with ``source`` set when it was not already (a fresh copy; never mutates a
        possibly-shared instance).
    """
    return fragment if fragment.source is not None else replace(fragment, source=node)
