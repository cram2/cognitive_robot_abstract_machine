"""
EQL verbalizer — coordinator and one-shot convenience entry point.

:class:`EQLVerbalizer` dispatches an EQL expression tree to the rule engine and
returns a :class:`~krrood.entity_query_language.verbalization.fragments.base.VerbFragment`
tree.  :func:`verbalize_expression` is the simplest entry point — it returns a plain
English string with no colour markup.

For coloured / hierarchical output use
:class:`~krrood.entity_query_language.verbalization.pipeline.VerbalizationPipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.query.query import Query
from krrood.entity_query_language.verbalization.context import VerbalizationContext
from krrood.entity_query_language.verbalization.fragments.base import VerbFragment
from krrood.entity_query_language.verbalization.rule_engine import RuleEngine
from krrood.entity_query_language.verbalization.rules.registry import ALL_RULES
from krrood.entity_query_language.verbalization.utils import _str

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.rendering.renderer import FragmentRenderer


@dataclass
class EQLVerbalizer:
    """
    Coordinator that maps an EQL expression tree to a :class:`~krrood.entity_query_language.verbalization.fragments.base.VerbFragment` tree.

    Dispatches via a :class:`~krrood.entity_query_language.verbalization.rule_engine.RuleEngine` of
    :class:`~krrood.entity_query_language.verbalization.rule_engine.VerbalizationRule` classes.
    Each rule declares its guard in :meth:`~krrood.entity_query_language.verbalization.rule_engine.VerbalizationRule.applies`
    and its rendering in :meth:`~krrood.entity_query_language.verbalization.rule_engine.VerbalizationRule.transform`.
    More-specific subclasses are tried before their parents (MRO-depth priority).

    For simple plain-text output use :func:`verbalize_expression`.
    For coloured / formatted output build a
    :class:`~krrood.entity_query_language.verbalization.pipeline.VerbalizationPipeline`.
    """

    _engine: RuleEngine = field(init=False, repr=False)
    """Rule dispatcher; sorts rules by MRO depth before first call."""

    def __post_init__(self) -> None:
        self._engine = RuleEngine(ALL_RULES)

    def build(
        self,
        expression: SymbolicExpression,
        context: Optional[VerbalizationContext] = None,
    ) -> VerbFragment:
        """
        Translate *expression* into a :class:`~krrood.entity_query_language.verbalization.fragments.base.VerbFragment` tree.

        A fresh :class:`~krrood.entity_query_language.verbalization.context.VerbalizationContext`
        (with a pre-built disambiguation map) is created when *context* is ``None``.

        :param expression: Any EQL symbolic expression.
        :type expression: ~krrood.entity_query_language.core.base_expressions.SymbolicExpression
        :param context: Shared verbalization state; created automatically when omitted.
        :type context: ~krrood.entity_query_language.verbalization.context.VerbalizationContext or None
        :returns: Root of the fragment tree representing *expression* in natural language.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        if context is None:
            context = VerbalizationContext.from_expression(expression)
        return self._engine.build(expression, context, self)

    def verbalize(
        self,
        expression: SymbolicExpression,
        context: Optional[VerbalizationContext] = None,
    ) -> str:
        """
        Translate *expression* into a plain-text English string.

        Equivalent to ``_str(self.build(expression, context))`` — no colour markup.
        Prefer :class:`~krrood.entity_query_language.verbalization.pipeline.VerbalizationPipeline`
        when colour or hierarchical layout is needed.

        :param expression: Any EQL symbolic expression.
        :type expression: ~krrood.entity_query_language.core.base_expressions.SymbolicExpression
        :param context: Shared verbalization state; created automatically when omitted.
        :type context: ~krrood.entity_query_language.verbalization.context.VerbalizationContext or None
        :returns: Plain-text natural-language representation of *expression*.
        :rtype: str
        """
        return _str(self.build(expression, context))


_default_verbalizer = EQLVerbalizer()


def verbalize_expression(
    expression,
    *,
    renderer: "FragmentRenderer | None" = None,
) -> str:
    """
    Verbalize any EQL expression into a human-readable English phrase.

    This is the simplest entry point.  With no arguments it returns plain text
    (no colour markup, paragraph prose).  Pass a *renderer* to control output
    format and layout:

    * :class:`~krrood.entity_query_language.verbalization.rendering.renderer.ParagraphRenderer`
      (:class:`~krrood.entity_query_language.verbalization.rendering.formatter.PlainFormatter`) —
      plain prose (default).
    * ``ParagraphRenderer(ANSIFormatter())`` — ANSI-coloured prose.
    * ``ParagraphRenderer(HTMLFormatter())`` — HTML-coloured prose.
    * :class:`~krrood.entity_query_language.verbalization.rendering.renderer.HierarchicalRenderer`
      — indented bullet list; pair with any formatter.

    For source hyperlinks pass a configured *renderer* with a *link_resolver*,
    e.g. ``HierarchicalRenderer(HTMLFormatter(), resolver)``.

    :param expression: Any EQL expression or :class:`~krrood.entity_query_language.query.query.Query`.
    :param renderer: Optional
        :class:`~krrood.entity_query_language.verbalization.rendering.renderer.FragmentRenderer`
        instance.  When ``None`` the default plain-text output is produced.
    :returns: Natural-language string (format depends on *renderer*).
    :rtype: str
    """
    if isinstance(expression, Query):
        expression.build()
    if renderer is None:
        return _default_verbalizer.verbalize(expression)
    # Lazy import avoids circular dependency (pipeline imports verbalizer).
    from krrood.entity_query_language.verbalization.pipeline import VerbalizationPipeline
    return VerbalizationPipeline(renderer).verbalize(expression)
