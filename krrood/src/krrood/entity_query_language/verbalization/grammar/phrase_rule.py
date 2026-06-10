"""
The grammar as first-class rule objects — a :class:`PhraseRule` subclass per EQL
construct, the dispatch primitive :func:`select`, and the per-node :class:`Ctx`
handed to a rule.

This realises the **rule-to-rule** mapping of Montague grammar: each construct of
the source algebra (an EQL :class:`~krrood.entity_query_language.core.base_expressions.SymbolicExpression`)
has one rule describing how it composes into the target (English) algebra.  Rules
are registered as *instances* (see
:mod:`~krrood.entity_query_language.verbalization.grammar.registry`), so the grammar
is still queryable with EQL (``entity(r).where(r.construct == Comparator)``) while a
rule's behaviour lives in an overridable :meth:`PhraseRule.build` method — the
natural home for the planner/assembler split of the more complex constructs.

Specificity is driven by the ``construct`` class attribute (not by the rule-class
hierarchy), so rules stay flat and dispatch is honest.

References:

* Montague, R. (1970), "Universal Grammar", *Theoria* 36 — syntax algebra →
  semantics algebra as a homomorphism.
* Bach, E. (1976) — the *rule-to-rule hypothesis* (one syntactic rule ↔ one
  semantic rule).
* Stanford Encyclopedia of Philosophy, "Montague Semantics" / "Compositionality".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing_extensions import Any, Callable, ClassVar, Optional, Sequence

from krrood.entity_query_language.verbalization.fragments.base import VerbFragment
from krrood.entity_query_language.verbalization.fragments.features import Number
from krrood.entity_query_language.verbalization.grammar.selection import most_specific

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.context import VerbalizationContext
    from krrood.entity_query_language.verbalization.microplanning.binding_scope import (
        BindingScope,
    )
    from krrood.entity_query_language.verbalization.microplanning.config import (
        RenderConfig,
    )
    from krrood.entity_query_language.verbalization.microplanning.referring import (
        ReferringExpressions,
    )


@dataclass
class Ctx:
    """
    Per-node context handed to :meth:`PhraseRule.build`.

    Bundles the single recursion entry (:attr:`child`, the fold continuation) with
    the microplanning services, so a ``build`` never recurses by hand and never
    reaches for cross-cutting state directly.
    """

    child: Callable[..., VerbFragment]
    """Recurse on a sub-expression — the fold continuation bound to this pass.  Accepts an
    optional ``number=`` to request a plural realisation of the child (consumed by that
    child's rule, not inherited further down)."""

    context: "VerbalizationContext"
    """The owning verbalization context (services accessed via the properties below)."""

    number: Number = Number.SINGULAR
    """Grammatical number *requested* for **this** node (set by the parent's ``child(...,
    number=…)``).  A number-aware rule (variable / chain / flat-variable) reads it to build
    the bare plural noun-phrase shape and tag its leaves; every other rule ignores it
    (renders singular).  Per-edge: a rule's own ``ctx.child`` calls default back to singular."""

    @property
    def refer(self) -> "ReferringExpressions":
        """Referring-expression service (articles, coreference, pronouns)."""
        return self.context.referring

    @property
    def scope(self) -> "BindingScope":
        """Binding-scope service (deferred constraints + field overrides)."""
        return self.context.binding

    @property
    def config(self) -> "RenderConfig":
        """Render-mode flags (query depth, compact predicates)."""
        return self.context.config


class PhraseRule(ABC):
    """
    One Montague rule-to-rule clause: *for this construct, build this phrase.*

    Subclasses set :attr:`construct` (and optionally :attr:`name` / :attr:`tiebreak`),
    may override :meth:`when` to add a guard beyond the ``isinstance`` test, and must
    implement :meth:`build`.  Rules are flat (all direct subclasses) — specificity
    comes from :attr:`construct`, never from the rule-class hierarchy.

    :cvar construct: The EQL node class this rule handles (the ``isinstance`` gate).
    :cvar name: Stable identifier for querying / tracing the grammar.
    :cvar tiebreak: Explicit ordering for the rare case of two rules with the same
        ``construct`` that are *both* guarded and overlap (e.g. inference vs.
        top-level entity); higher wins.
    """

    construct: ClassVar[type]
    name: ClassVar[str] = ""
    tiebreak: ClassVar[int] = 0

    def when(self, node, ctx: Ctx) -> bool:
        """
        Extra precondition beyond ``isinstance(node, construct)``.

        The default accepts everything; override to express the non-``isinstance``
        part of the rule's applicability (a *guarded* rule outranks an unguarded one
        on the same construct).  Receives the same :class:`Ctx` as :meth:`build`, so a
        guard may consult the microplanning services (e.g. ``ctx.config.query_depth``).

        :param node: The candidate EQL expression.
        :param ctx: The per-node context (recursion + services).
        :rtype: bool
        """
        return True

    @abstractmethod
    def build(self, node, ctx: Ctx) -> VerbFragment:
        """
        Build the fragment for *node*, delegating recursion to ``ctx.child`` and
        cross-cutting decisions to ``ctx`` services / morphology / coordination /
        the lexicon.

        :param node: The EQL expression to verbalize.
        :param ctx: The per-node context (recursion + services).
        :return: The fragment for *node*.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """


def _mro_depth(cls: type) -> int:
    """Specificity of a construct: deeper in the MRO ⇒ more specific (``Literal`` > ``Variable``)."""
    return len(cls.__mro__)


def _is_guarded(rule: PhraseRule) -> bool:
    """``True`` when *rule* overrides :meth:`PhraseRule.when` (a guarded rule)."""
    return type(rule).when is not PhraseRule.when


def select(node, rules: Sequence[PhraseRule], ctx: Ctx) -> Optional[PhraseRule]:
    """
    Return the most-specific :class:`PhraseRule` whose ``construct`` and :meth:`~PhraseRule.when`
    match *node*, or ``None`` when none apply.

    Specificity key, highest wins: ``(construct MRO depth, guarded over unguarded,
    explicit tiebreak)`` — reproducing the previous engine's MRO-depth ordering and
    ``applies`` guards without a rule-class hierarchy.

    :param node: The EQL expression being dispatched.
    :param rules: The grammar (e.g. ``RULES``).
    :param ctx: The per-node context, passed to each rule's :meth:`~PhraseRule.when`.
    :return: The chosen rule, or ``None`` (caller supplies the fallback).
    """
    candidates = [
        rule
        for rule in rules
        if isinstance(node, rule.construct) and rule.when(node, ctx)
    ]
    return most_specific(
        candidates,
        key=lambda rule: (
            _mro_depth(rule.construct),
            _is_guarded(rule),
            rule.tiebreak,
        ),
    )
