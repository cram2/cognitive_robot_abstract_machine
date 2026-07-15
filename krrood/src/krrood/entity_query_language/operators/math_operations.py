"""
Arithmetic operators for the Entity Query Language.

An arithmetic node (see :mod:`krrood.entity_query_language.operators.arithmetic`) delegates its
computation to the :class:`MathOperator` it carries: each operator owns both its rendered symbol and the
Python callable that performs it, so the node stays decoupled from the concrete operation.
"""

from __future__ import annotations

import numbers
import operator
from dataclasses import dataclass
from enum import Enum

from typing_extensions import Callable


@dataclass(frozen=True)
class MathOperatorSpecification:
    """
    The symbol and callable that make up one :class:`MathOperator`.

    Mixed into ``MathOperator`` itself (an Enum mix-in type), so each member's
    ``symbol``/``function`` are its own attributes directly, with no separate value
    object to forward through.
    """

    symbol: str
    """
    The mathematical symbol used when rendering the operator.
    """

    function: Callable[..., numbers.Number]
    """
    The callable that performs the operation over already-resolved operand values.
    """


class MathOperator(MathOperatorSpecification, Enum):
    """
    An arithmetic operator usable inside a query.

    Each member carries the symbol it renders as and the callable that computes it over
    already-resolved operand values.
    """

    ADD = ("+", operator.add)
    SUBTRACT = ("-", operator.sub)
    MULTIPLY = ("*", operator.mul)
    DIVIDE = ("/", operator.truediv)
    FLOOR_DIVIDE = ("//", operator.floordiv)
    MODULO = ("%", operator.mod)
    POWER = ("**", operator.pow)
    NEGATE = ("-", operator.neg)
