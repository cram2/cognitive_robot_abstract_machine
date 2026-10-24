"""
Rows of bindings together with how likely a probabilistic model finds them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from krrood.entity_query_language.core.base_expressions import OperationResult


@dataclass(eq=False)
class ProbabilisticOperationResult(OperationResult):
    """
    A result whose bindings a probabilistic model assigns a likelihood to.
    """

    log_likelihood: float = field(kw_only=True)
    """
    The natural logarithm of the likelihood of the bindings under the model.
    """
