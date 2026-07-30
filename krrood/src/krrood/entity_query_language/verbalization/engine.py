from __future__ import annotations

from dataclasses import replace

from typing_extensions import TYPE_CHECKING, Optional, Sequence

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
)
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
    from krrood.entity_query_language.verbalization.microplanning.coordination import (
        FoldNode,
    )


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

    >>> from krrood.entity_query_language.verbalization.context import MicroplanningServices
    >>> robot = variable(Robot, [])
    >>> context = root_context(MicroplanningServices.from_expression(robot), RULES)
    >>> type(context.child(robot)).__name__
    'NounPhrase'
    """
    return RuleContext(
        recurse=lambda child_node, child_options: fold(
            child_node, services, rules, child_options
        ),
        services=services,
        options=options if options is not None else RenderOptions(),
    )


def fold(
    node: FoldNode,
    services: MicroplanningServices,
    rules: Optional[Sequence[PhraseRule]] = None,
    options: Optional[RenderOptions] = None,
) -> VerbalizationFragment:
    """
    Verbalize *node* by dispatching it to its matching grammar rule and recursing — the
    single catamorphism (fold) over the EQL expression tree.

    A node carrying a pre-built binding override is returned directly, before any dispatch.
    When no rule covers the node, an ``UnverbalizableExpressionError`` is raised rather than
    degrading silently to the class name.

    The recursion is an F-algebra fold (catamorphism) over the EQL algebra, with the grammar as
    the algebra — see :func:`~krrood.entity_query_language.verbalization.fragments.base.fold_fragment`
    for the catamorphism / F-algebra definition (:cite:t:`meijer1991bananas`; :cite:t:`bird1997algebra`).

    :param node: Any EQL expression, or a synthetic coordination artifact produced by conjunct
        reduction (:class:`RangeFold` / :class:`CoindexedFold`).
    :param services: The pass-wide microplanning services (and render configuration).
    :param rules: Grammar to dispatch over; defaults to the standard rule set.
    :param options: The render flags to build *node* under (defaults to all reset).
    :return: The fragment for *node*.
    :raises UnverbalizableExpressionError: when no grammar rule covers *node*.

    The catamorphism is observable through :func:`~krrood.entity_query_language.verbalization.pipeline.verbalize_expression`,
    which runs this fold and then lowers the produced tree — a two-hop chain dispatches the
    attribute rule, which recurses on its variable child:

    >>> verbalize_expression(variable(Robot, []).battery)
    'the battery of a Robot'
    """
    rules = RULES if rules is None else rules

    if isinstance(node, SymbolicExpression):
        override = services.binding.binding_overrides.get(node._id_)
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


def _with_source(
    fragment: VerbalizationFragment, node: FoldNode
) -> VerbalizationFragment:
    """
    Stamp *node* as the fragment's provenance, so later passes can follow it back to the read
    model. The innermost producer wins: a transparent wrapper (``An(Entity)``) returns its child's
    fragment, which already carries the child's (the ``Entity``'s) source, and is left untouched.

    :param fragment: The fragment a rule produced for *node*.
    :param node: The EQL node dispatched.
    :return: *fragment* with ``source`` set when it was not already (a fresh copy; never mutates a
        possibly-shared instance).

    >>> from krrood.entity_query_language.verbalization.fragments.base import RoleFragment
    >>> robot = variable(Robot, [])
    >>> _with_source(RoleFragment.for_variable("Robot", robot), robot).source is robot
    True
    >>> stamped = _with_source(RoleFragment.for_variable("Robot", robot), robot)
    >>> _with_source(stamped, variable(Robot, [])).source is robot
    True
    """
    return fragment if fragment.source is not None else replace(fragment, source=node)
