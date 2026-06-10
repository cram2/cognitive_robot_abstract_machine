"""
Verbalization context — a thin facade composing the microplanning services
threaded through a single
:meth:`~krrood.entity_query_language.verbalization.verbalizer.EQLVerbalizer.verbalize`
call.

The per-pass state is split by concern into three single-responsibility services
(mirroring the microplanning subtasks of Reiter & Dale 2000):

* :class:`~krrood.entity_query_language.verbalization.microplanning.referring.ReferringExpressions`
  — coreference, articles, disambiguation, pronouns.
* :class:`~krrood.entity_query_language.verbalization.microplanning.binding_scope.BindingScope`
  — deferred-constraint frames and field-reference overrides.
* :class:`~krrood.entity_query_language.verbalization.microplanning.config.RenderConfig`
  — render-mode flags (query depth, compact predicates).

:class:`VerbalizationContext` keeps a backward-compatible surface (the fields the
rules read directly and the methods they call) by delegating to these services,
so each concern can be reasoned about and tested in isolation.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from typing_extensions import Any, Optional

from krrood.entity_query_language.verbalization.fragments.base import VerbFragment
from krrood.entity_query_language.verbalization.microplanning.binding_scope import (
    BindingScope,
)
from krrood.entity_query_language.verbalization.microplanning.config import RenderConfig
from krrood.entity_query_language.verbalization.fragments.features import Definiteness
from krrood.entity_query_language.verbalization.microplanning.referring import (
    ReferringExpressions,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import SymbolicExpression

__all__ = ["VerbalizationContext"]


@dataclass
class VerbalizationContext:
    """
    Facade over the microplanning services for one verbalization pass.

    Holds a :class:`~krrood.entity_query_language.verbalization.microplanning.referring.ReferringExpressions`,
    a :class:`~krrood.entity_query_language.verbalization.microplanning.binding_scope.BindingScope`,
    and a :class:`~krrood.entity_query_language.verbalization.microplanning.config.RenderConfig`,
    delegating to each.  Create via :meth:`from_expression` to pre-load the
    disambiguation map.
    """

    referring: ReferringExpressions = field(default_factory=ReferringExpressions)
    """Coreference / article / disambiguation / pronoun service."""

    binding: BindingScope = field(default_factory=BindingScope)
    """Deferred-constraint frames and field-reference overrides."""

    config: RenderConfig = field(default_factory=RenderConfig)
    """Render-mode flags (query depth, compact predicates)."""

    @classmethod
    def from_expression(cls, expression) -> VerbalizationContext:
        """
        Create a context with the disambiguation map pre-built for *expression*.

        :param expression: Root EQL expression or Query to scan.
        :return: A fresh context whose referring service has its disambiguation map populated.
        :rtype: VerbalizationContext
        """
        return cls(referring=ReferringExpressions.from_expression(expression))

    # ── Referring-expression delegation ──────────────────────────────────────

    @property
    def seen(self) -> dict:
        """Coreference map (``_id_`` → label); see :attr:`ReferringExpressions.seen`."""
        return self.referring.seen

    @property
    def disambiguation_map(self) -> dict:
        """Pre-computed disambiguation labels; see :attr:`ReferringExpressions.disambiguation_map`."""
        return self.referring.disambiguation_map

    def push_subject(self, var) -> None:
        """Delegate to :meth:`ReferringExpressions.push_subject`."""
        self.referring.push_subject(var)

    def pop_subject(self) -> None:
        """Delegate to :meth:`ReferringExpressions.pop_subject`."""
        self.referring.pop_subject()

    @property
    def current_subject_id(self):
        """``_id_`` of the current coreference subject, or ``None``."""
        return self.referring.current_subject_id

    def register(self, expression, label: VerbFragment) -> None:
        """Delegate to :meth:`ReferringExpressions.register`."""
        self.referring.register(expression, label)

    def register_label(self, expression, text: str) -> None:
        """Delegate to :meth:`ReferringExpressions.register_label`."""
        self.referring.register_label(expression, text)

    def alias(self, target, source) -> None:
        """Delegate to :meth:`ReferringExpressions.alias`."""
        self.referring.alias(target, source)

    def seen_reference(self, expression) -> Optional[VerbFragment]:
        """Delegate to :meth:`ReferringExpressions.seen_reference`."""
        return self.referring.seen_reference(expression)

    def pronoun_for(self, root) -> Optional[VerbFragment]:
        """Delegate to :meth:`ReferringExpressions.pronoun_for`."""
        return self.referring.pronoun_for(root)

    def noun_for_parts(self, var) -> tuple[Definiteness, str]:
        """Delegate to :meth:`ReferringExpressions.noun_for_parts`."""
        return self.referring.noun_for_parts(var)

    # ── Binding-scope delegation ─────────────────────────────────────────────

    @property
    def binding_overrides(self) -> dict:
        """Field-reference override map; see :attr:`BindingScope.binding_overrides`."""
        return self.binding.binding_overrides

    @property
    def constraint_exprs(self) -> list:
        """Deferred-constraint frame stack; see :attr:`BindingScope.constraint_exprs`."""
        return self.binding.constraint_exprs

    def push_constraint_frame(self) -> None:
        """Delegate to :meth:`BindingScope.push_constraint_frame`."""
        self.binding.push_constraint_frame()

    def pop_constraint_frame(self):
        """Delegate to :meth:`BindingScope.pop_constraint_frame`."""
        return self.binding.pop_constraint_frame()

    def defer_constraint(self, expression: SymbolicExpression) -> None:
        """Delegate to :meth:`BindingScope.defer_constraint`."""
        self.binding.defer_constraint(expression)

    # ── Render-config delegation ─────────────────────────────────────────────

    @property
    def query_depth(self) -> int:
        """Current query/noun nesting depth; see :attr:`RenderConfig.query_depth`."""
        return self.config.query_depth

    @property
    def compact_predicates(self) -> bool:
        """Whether comparators drop the copula; see :attr:`RenderConfig.compact_predicates`."""
        return self.config.compact_predicates

    def query_depth_scope(self):
        """Delegate to :meth:`RenderConfig.query_depth_scope`."""
        return self.config.query_depth_scope()

    def compact_predicates_scope(self):
        """Delegate to :meth:`RenderConfig.compact_predicates_scope`."""
        return self.config.compact_predicates_scope()

    # ── Value lexicalisation ─────────────────────────────────────────────────

    def type_name_of_value(self, value: Any) -> str:
        """
        Render a Python value as a human-readable string.

        * A bare ``type`` → its ``__name__`` (``Apple`` → ``"Apple"``).
        * A tuple of types → ``"A or B or C"``.
        * A ``datetime`` with no time → ``"May 23, 2026"``; with a time →
          ``"May 23, 2026 at 14:30"``.
        * Anything else → ``repr(value)``.

        :param value: Python value from a
            :class:`~krrood.entity_query_language.core.variable.Literal` node.
        :return: Human-readable string representation.
        :rtype: str
        """
        if isinstance(value, type):
            return value.__name__
        if isinstance(value, tuple) and all(
            isinstance(variable, type) for variable in value
        ):
            return " or ".join(variable.__name__ for variable in value)
        if isinstance(value, datetime.datetime):
            if value.time() == datetime.time.min:
                return value.strftime("%B %-d, %Y")
            return value.strftime("%B %-d, %Y at %H:%M")
        return repr(value)
