from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING, Optional

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.verbalization.microplanning.binding_scope import (
    BindingScope,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.vocabulary.english import Directive
from krrood.entity_query_language.verbalization.microplanning.config import (
    RenderConfiguration,
)
from krrood.entity_query_language.verbalization.microplanning.microplan import (
    Microplan,
)
from krrood.entity_query_language.verbalization.microplanning.referring import (
    ReferringExpressions,
)


@dataclass
class MicroplanningServices:
    """
    The three microplanning services for one verbalization pass: the referring-
    expression service, the binding scope, and the render configuration.

    The split mirrors the microplanning subtasks of :cite:t:`reiter2000building`.
    """

    referring: ReferringExpressions = field(default_factory=ReferringExpressions)
    """
    Coreference / article / disambiguation / pronoun service.
    """

    binding: BindingScope = field(default_factory=BindingScope)
    """
    Deferred-constraint frames and field-reference overrides.
    """

    configuration: RenderConfiguration = field(default_factory=RenderConfiguration)
    """
    Render-mode flags (query depth, compact predicates).
    """

    microplan: Microplan = field(default_factory=Microplan)
    """
    The plan read model — each node's plan computed once and shared (lazy / memoised).
    """

    performative_override: Optional[Directive] = None
    """
    An explicit opening directive (``Find`` / ``Generate``) chosen by the caller —
    typically resolved from the evaluating backend (generative → ``Generate``, selective
    → ``Find``).

    When set it overrides the query-type default; ``None`` keeps the default derived
    from the query.
    """

    @classmethod
    def from_expression(cls, expression: SymbolicExpression) -> MicroplanningServices:
        """
        Create a context with the disambiguation map pre-built for *expression*.

        :param expression: Root EQL expression or query to scan.
        :return: A fresh context whose referring service has its disambiguation map
            populated.
        """
        return cls(referring=ReferringExpressions.from_expression(expression))
