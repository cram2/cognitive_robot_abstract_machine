"""
Tests of expressions an expert writes as source code.
"""

from __future__ import annotations

import pytest

from krrood.ripple_down_rules.datastructures.callable_expression import (
    CallableExpression,
)
from krrood.ripple_down_rules.exceptions import (
    NoUserInputOrConclusionForCallableExpression,
)

# %% constructing an expression


def test_an_expression_needs_user_input_or_a_conclusion() -> None:
    """
    An expression with neither source code nor a conclusion has nothing to evaluate.
    """
    with pytest.raises(NoUserInputOrConclusionForCallableExpression):
        CallableExpression()
