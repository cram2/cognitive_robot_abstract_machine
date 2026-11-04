"""
Common interface for fitted relational distribution templates.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from krrood.entity_query_language.query.match import AbstractMatchExpression, Match

if TYPE_CHECKING:
    from probabilistic_model.probabilistic_circuit.relational.relational_probabilistic_circuit import (
        RelationalProbabilisticCircuit,
    )


@dataclass
class RelationalDistributionTemplate(ABC):
    """
    A fitted distribution template of one relation, grounded once per query part.
    """

    template_distribution: RelationalProbabilisticCircuit
    """
    The fitted ``RelationalProbabilisticCircuit`` shared by every grounded part.
    """

    @staticmethod
    def _prefix_for_part(part: Match, index: int) -> str:
        """
        The namespace prefix a grounded part's variables are renamed under.

        :param part: The query part being grounded.
        :param index: Position of the part in its parent list; used as fallback prefix
            when ``part`` does not carry a symbolic variable.
        :return: The prefix taken from ``part``'s variable, or ``str(index)`` when it
            carries none.
        """
        return (
            str(part._variable_)
            if isinstance(part, AbstractMatchExpression)
            else str(index)
        )
