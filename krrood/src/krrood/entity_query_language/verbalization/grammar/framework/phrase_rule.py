from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing_extensions import TYPE_CHECKING, Callable, ClassVar, Optional, Sequence

from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    GrammaticalNumber,
)
from krrood.entity_query_language.verbalization.exceptions import AmbiguousRuleError
from krrood.patterns.specificity_ranking import mro_depth, sole_maximum

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.context import MicroplanningServices
    from krrood.entity_query_language.verbalization.microplanning.coordination import (
        FoldNode,
    )
    from krrood.entity_query_language.verbalization.microplanning.binding_scope import (
        BindingScope,
    )
    from krrood.entity_query_language.verbalization.microplanning.config import (
        RenderConfiguration,
    )
    from krrood.entity_query_language.verbalization.microplanning.microplan import (
        Microplan,
    )
    from krrood.entity_query_language.verbalization.microplanning.referring import (
        ReferringExpressions,
    )


@dataclass(frozen=True)
class RenderOptions:
    """The per-recursion render flags, bundled so a new one is a single field here rather than a
    new positional parameter threaded through every recursion site.

    Each flag is *per-fold*: it applies to the node being folded and resets for its children (so
    :meth:`RuleContext.child` builds a fresh ``RenderOptions`` from its keyword overrides, never
    inheriting the parent's).
    """

    number: GrammaticalNumber = GrammaticalNumber.SINGULAR
    """Grammatical number requested for this node.  A number-aware rule reads it to build a
    plural noun-phrase shape; every other rule ignores it (renders singular)."""

    inline: bool = False
    """``True`` when this node is being folded in chain-root position, so an ``Entity`` renders as
    an inline noun rather than a *"Find …"* / nested phrase."""

    as_value: bool = False
    """``True`` when this node is being folded in *value* position (a comparator's right side, a
    match assignment's value), so a domain-constrained value-type ``Variable`` renders as its
    candidate set (*"one of A, B, or C"*) rather than as a subject noun (*"an int"*)."""


@dataclass
class RuleContext:
    """
    Per-node context handed to a rule's ``build``.

    Bundles the recursion entry (:meth:`child`, the fold continuation) with the microplanning
    services and the per-node :class:`RenderOptions`, so a rule never recurses by hand nor reaches
    for cross-cutting state directly.
    """

    recurse: Callable[["FoldNode", "RenderOptions"], VerbalizationFragment]
    """The fold continuation — given a sub-expression and its render options, returns its fragment.
    Rules do not call this directly; they call :meth:`child`."""

    services: MicroplanningServices
    """The owning microplanning services."""

    options: RenderOptions = field(default_factory=RenderOptions)
    """The render flags for *this* node (number / inline / value position)."""

    def child(
        self,
        node: FoldNode,
        *,
        number: GrammaticalNumber = GrammaticalNumber.SINGULAR,
        inline: bool = False,
        as_value: bool = False,
    ) -> VerbalizationFragment:
        """Recurse on a sub-expression, requesting its render flags (all reset by default — they do
        not inherit from this node).

        :param node: The sub-expression to fold.
        :param number: Grammatical number to build the child under.
        :param inline: Fold the child in chain-root (inline-noun) position.
        :param as_value: Fold the child in value position (domain variables list their candidates).
        :return: The child's fragment.

        >>> verbalize_expression(variable(BankTransaction, []).amount_details.amount)
        'the amount of the amount_details of a BankTransaction'
        """
        return self.recurse(
            node, RenderOptions(number=number, inline=inline, as_value=as_value)
        )

    @property
    def number(self) -> GrammaticalNumber:
        """:return: The grammatical number requested for this node."""
        return self.options.number

    @property
    def inline(self) -> bool:
        """:return: Whether this node is folded in inline-noun position."""
        return self.options.inline

    @property
    def as_value(self) -> bool:
        """:return: Whether this node is folded in value position."""
        return self.options.as_value

    @property
    def refer(self) -> ReferringExpressions:
        """:return: The referring-expression service (articles, coreference, pronouns)."""
        return self.services.referring

    @property
    def scope(self) -> BindingScope:
        """:return: The binding-scope service (deferred constraints + field overrides)."""
        return self.services.binding

    @property
    def configuration(self) -> RenderConfiguration:
        """:return: The render-mode flags (query depth, compact predicates)."""
        return self.services.configuration

    @property
    def microplan(self) -> Microplan:
        """:return: The plan read model (each node's plan computed once and shared)."""
        return self.services.microplan


class PhraseRule(ABC):
    """
    One Montague rule-to-rule clause: *for this construct, build this phrase.*

    This realises the rule-to-rule mapping of Montague grammar: each construct of the source
    algebra (an EQL expression) has one rule describing how it composes into the target
    (English) algebra. Specificity comes primarily from ``construct``; rules are otherwise flat,
    except that a rule which is a *special case* of another (its guard implies the other's) may
    subclass it, and ``select`` then prefers the more-derived class. Rules whose guards merely
    overlap have no such is-a relationship and must instead keep their guards mutually exclusive.

    References:

    * :cite:t:`montague1970universal` — syntax algebra → semantics algebra as a homomorphism.
    * :cite:t:`bach1976extension` — the rule-to-rule hypothesis (one syntactic rule ↔ one semantic rule).
    * :cite:t:`janssen2021montague`, :cite:t:`szabo2022compositionality` — Montague semantics and compositionality.
    """

    construct: ClassVar[type]
    """The EQL node class this rule handles (the ``isinstance`` gate)."""
    enters_query_scope: ClassVar[bool] = False
    """``True`` on a rule whose construct is itself a query body, so an entity found anywhere
    within it renders as a nested noun phrase."""

    def when(self, node: FoldNode, context: RuleContext) -> bool:
        """
        Extra precondition beyond ``isinstance(node, construct)``.

        The default accepts everything; override to express the non-``isinstance`` part of the
        rule's applicability (a guarded rule outranks an unguarded one on the same construct).

        :param node: The candidate EQL expression.
        :param context: The per-node context (recursion and services).
        :return: ``True`` when the rule applies to *node*.
        """
        return True

    @abstractmethod
    def build(self, node: FoldNode, context: RuleContext) -> VerbalizationFragment:
        """
        Build the fragment for *node*.

        :param node: The EQL expression to verbalize.
        :param context: The per-node context (recursion and services).
        :return: The fragment for *node*.
        """


def _is_guarded(rule: PhraseRule) -> bool:
    """
    :param rule: A phrase rule instance.
    :return: ``True`` when *rule* overrides ``when`` (a guarded rule).

    >>> from krrood.entity_query_language.verbalization.grammar.query.rules import (
    ...     SetOfRule, TopLevelEntityRule)
    >>> _is_guarded(TopLevelEntityRule())
    True
    >>> _is_guarded(SetOfRule())
    False
    """
    return type(rule).when is not PhraseRule.when


def select(
    node: FoldNode, rules: Sequence[PhraseRule], context: RuleContext
) -> Optional[PhraseRule]:
    """
    Specificity key, highest wins: ``(construct MRO depth, guarded over unguarded, rule-class MRO
    depth)``.

    The last component lets a rule that is a *special case* of another express that by subclassing
    it: when both guards hold, the more-derived rule class wins (e.g. an inference-rule entity is a
    refinement of a top-level entity). This only models *subsumption* — a guard that implies
    another's. Rules whose guards merely *overlap* (neither implies the other) have no is-a
    relationship to express; their guards must be mutually exclusive, since an equal-specificity
    tie is a collision raised as :class:`~krrood.entity_query_language.verbalization.exceptions.AmbiguousRuleError`.

    :param node: The EQL expression being dispatched.
    :param rules: The grammar.
    :param context: The per-node context, passed to each rule's ``when``.
    :return: The most-specific rule whose ``construct`` and ``when`` match *node*, or ``None``
        when none apply (the caller supplies the fallback).
    :raises AmbiguousRuleError: When two rules are equally specific for *node*.

    The winning rule decides the phrasing: an inference-rule entity is dispatched to the
    more-derived ``InferenceRuleRule`` (an ``IF … THEN …`` block) rather than the plain
    top-level entity form.

    >>> from krrood.entity_query_language.factories import inference
    >>> fixed = variable(FixedConnection, [])
    >>> verbalize_expression(entity(inference(Drawer)(
    ...     container=fixed.parent, handle=fixed.child))).startswith('If')
    True
    """
    candidates = [
        rule
        for rule in rules
        if isinstance(node, rule.construct) and rule.when(node, context)
    ]
    return sole_maximum(
        candidates,
        key=lambda rule: (
            mro_depth(rule.construct),
            _is_guarded(rule),
            mro_depth(type(rule)),
        ),
        collision_error=lambda tied: AmbiguousRuleError(
            subject=node, candidates=[type(rule) for rule in tied]
        ),
    )
