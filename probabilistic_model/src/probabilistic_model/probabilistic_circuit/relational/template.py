"""
Common interface for fitted relational distribution templates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from krrood.entity_query_language.query.match import AbstractMatchExpression, Match
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    Unit,
)

if TYPE_CHECKING:
    from probabilistic_model.probabilistic_circuit.relational.rspn import (
        RelationalProbabilisticCircuit,
    )


@dataclass
class RelationalDistributionTemplate(ABC):
    """
    Common interface for a fitted distribution template that grounds one
    ``RelationalProbabilisticCircuit`` per query part and combines the grounded
    copies into a single circuit -- under a product for
    :class:`~probabilistic_model.probabilistic_circuit.relational.rspn.ExchangeableDistributionTemplate`'s
    exchangeable relation, into a chain for
    :class:`~probabilistic_model.probabilistic_circuit.relational.markov_chain.MarkovChainDistributionTemplate`'s
    sequential one.
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
        :return: ``str(part.variable)`` if ``part`` carries one, else ``str(index)``.
        """
        return (
            str(part.variable)
            if isinstance(part, AbstractMatchExpression)
            else str(index)
        )

    @staticmethod
    def _ground_and_rename_part(
        template_distribution: RelationalProbabilisticCircuit,
        part: Match,
        index: int,
        excluded_variables: list,
    ) -> ProbabilisticCircuit:
        """
        Ground one part and rename its variables under its namespace prefix.

        :param template_distribution: The fitted circuit grounded once per part.
        :param part: The query part being grounded.
        :param index: Position of the part in its parent list; used as fallback prefix
            when ``part`` does not carry a symbolic variable.
        :param excluded_variables: Variables left unrenamed, e.g. latent variables
            shared with a parent circuit.
        :return: A self-contained circuit ready to be mounted into a parent circuit.
        """
        part_circuit = template_distribution.ground(part)
        prefix = RelationalDistributionTemplate._prefix_for_part(part, index)
        part_circuit.rename_variables_with_prefix(prefix, excluded_variables)
        return part_circuit

    @staticmethod
    def _mount_part(
        result: ProbabilisticCircuit, part_circuit: ProbabilisticCircuit
    ) -> Unit:
        """
        Mount an already-grounded, already-renamed part circuit into ``result``.

        :param result: The circuit ``part_circuit`` is mounted into.
        :param part_circuit: A self-contained circuit for one grounded part.
        :return: The root of the mounted part, owned by ``result``.
        """
        part_root_index = part_circuit.root.index
        node_index_map = result.mount(part_circuit.root)
        return node_index_map[part_root_index]

    @abstractmethod
    def ground(
        self, parts_to_ground: list[Match], *args, **kwargs
    ) -> ProbabilisticCircuit:
        """
        Ground one copy of the template per part and combine them into a single circuit.

        :param parts_to_ground: The query parts, one per position/child in the relation.
        :return: A circuit over all variables implied by the query.
        """
