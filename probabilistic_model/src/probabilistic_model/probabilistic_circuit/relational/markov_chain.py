"""
A relational, EQL-grounded template for hidden Markov models expressed as probabilistic
circuits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing_extensions import TYPE_CHECKING

from probabilistic_model.distributions.distributions import SymbolicDistribution
from probabilistic_model.distributions.multinomial import MultinomialDistribution
from probabilistic_model.exceptions import ShapeMismatchError
from probabilistic_model.probabilistic_circuit.relational.template import (
    RelationalDistributionTemplate,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.query.match import Match


@dataclass
class MarkovChainDistributionTemplate(RelationalDistributionTemplate):
    """
    A fitted distribution template for one sequential (Markov chain) relation.

    Grounds one ``RelationalProbabilisticCircuit`` per position in a sequence, the
    same way :class:`~probabilistic_model.probabilistic_circuit.relational.rspn.ExchangeableDistributionTemplate`
    grounds one per exchangeable child, and chains the grounded copies into a
    hidden Markov model instead of combining them under a product.

    Each grounded copy's root is a :class:`~probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit.SumUnit`;
    its latent-variable interpretation (see the "Latent Variable Interpretation"
    theorem in ``doc/circuits.md``) is the chain's hidden state at that position:
    the root's children are the states, and ``root.subcircuits[s]`` is
    ``P(observation | state = s)``.
    """

    starting_distribution: SymbolicDistribution
    """
    Distribution over the chain's hidden state for the first grounded part. Its
    ``variable``'s domain enumerates the hidden states as ``0, 1, ..., K - 1``
    (``K`` = number of hidden states), the same indices ``root.subcircuits`` uses.
    """

    transition_model: MultinomialDistribution
    """
    Joint table over ``(state, next_state)`` pairs, row-stochastic given ``state``;
    applied between every consecutive pair of grounded parts. Both of its variables'
    domains must match :attr:`starting_distribution`'s.
    """

    def __post_init__(self):
        state_domain = self.starting_distribution.variable.domain
        transition_domains = tuple(
            variable.domain for variable in self.transition_model.variables
        )
        expected_domains = (state_domain, state_domain)
        if transition_domains != expected_domains:
            raise ShapeMismatchError(transition_domains, expected_domains)

    @property
    def _state_count(self) -> int:
        return len(self.starting_distribution.variable.domain.simple_sets)

    def _starting_probability(self, state: int) -> float:
        return self.starting_distribution.probabilities[hash(state)]

    def _transition_probability(self, state: int, next_state: int) -> float:
        return self.transition_model.probabilities[state, next_state]

    def ground(self, parts_to_ground: list[Match]) -> ProbabilisticCircuit:
        """
        Ground one copy of the template per part and chain them into a hidden Markov
        model.

        :param parts_to_ground: The query parts, one per position in the sequence, in
            sequence order.
        :return: A circuit over all variables implied by the query, with the per-
            position hidden states marginalized out. Empty if ``parts_to_ground`` is.
        """
        if len(parts_to_ground) == 0:
            return ProbabilisticCircuit()

        result = ProbabilisticCircuit()
        roots = [
            self._ground_and_mount_part(result, index, part)
            for index, part in enumerate(parts_to_ground)
        ]

        if len(roots[0].subcircuits) != self._state_count:
            raise ShapeMismatchError(
                (len(roots[0].subcircuits),), (self._state_count,)
            )

        # Read every position's emission branches once, up front, since the chain
        # built below mutates the circuit -- if it re-read a position's branches
        # after mutating an earlier one, "state s" could stop meaning the same thing
        # at every position.
        emission_given_state = [list(root.subcircuits) for root in roots]

        self._point_root_at_hidden_markov_chain(result, roots[0], emission_given_state)

        result.remove_unreachable_nodes(roots[0])
        result.simplify()
        return result

    def _ground_and_mount_part(
        self, result: ProbabilisticCircuit, index: int, part: Match
    ):
        """
        Ground one part of the sequence and mount it into ``result``.

        :param result: The circuit the grounded part is mounted into.
        :param index: Position of the part in the sequence; used as fallback prefix when
            ``part`` does not carry a symbolic variable.
        :param part: The query part being grounded.
        :return: The root of the mounted part, owned by ``result``.
        """
        part_circuit = self._ground_and_rename_part(
            self.template_distribution, part, index, []
        )
        return self._mount_part(result, part_circuit)

    def _point_root_at_hidden_markov_chain(
        self,
        result: ProbabilisticCircuit,
        first_root: SumUnit,
        emission_given_state: list,
    ) -> None:
        """
        Rebuild ``first_root``'s edges so it becomes the root of the whole chain.

        Builds ``P(observation_t, ..., observation_{T-1} | state_t = s)`` from the
        last position back to the first: at each step it sums the next position's
        already-built continuations, weighted by :attr:`transition_model`, and pairs
        that sum with the current position's own emission. ``first_root``'s own
        children are then replaced by these length-``T`` continuations, weighted by
        :attr:`starting_distribution`.

        Working right to left means every node this method adds is only ever
        referenced once it is fully built, so nothing here depends on the order
        :meth:`~probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit.InnerUnit.subcircuits`
        happens to report for a node whose edges were changed earlier in the method
        -- unlike ``SumUnit.mount_with_interaction_terms``, which is built for a
        single pairwise combination and does not preserve a node's own children
        order once it has been used as its first argument, this never re-reads the
        children of a node it has already rebuilt.

        :param result: The circuit every new node is added to.
        :param first_root: The mounted root of the first grounded part; ends up as
            the root of the whole chain.
        :param emission_given_state: ``emission_given_state[t][s]`` is
            ``P(observation_t | state_t = s)``, read once before any mutation.
        """
        continuation_given_state = emission_given_state[-1]
        for position in reversed(range(len(emission_given_state) - 1)):
            continuation_given_state = self._continuation_given_state_at(
                result, position, continuation_given_state, emission_given_state
            )

        for state, own_subcircuit in enumerate(emission_given_state[0]):
            result.remove_edge(first_root, own_subcircuit)
            first_root.add_subcircuit(
                continuation_given_state[state],
                np.log(self._starting_probability(state)),
            )

    def _continuation_given_state_at(
        self,
        result: ProbabilisticCircuit,
        position: int,
        next_continuation_given_state: list,
        emission_given_state: list,
    ) -> list:
        """
        Build ``P(observation_position, ..., observation_{T-1} | state_position = s)``
        for every state ``s``, from the next position's already-built continuations.

        :param result: The circuit every new node is added to.
        :param position: The position this continuation is built for.
        :param next_continuation_given_state: Position ``position + 1``'s already-built
            continuations, indexed by state.
        :param emission_given_state: ``emission_given_state[t][s]`` is
            ``P(observation_t | state_t = s)``.
        :return: The new continuations for ``position``, indexed by state.
        """
        return [
            self._joint_of_emission_and_transition(
                result, position, state, next_continuation_given_state, emission_given_state
            )
            for state in range(self._state_count)
        ]

    def _joint_of_emission_and_transition(
        self,
        result: ProbabilisticCircuit,
        position: int,
        state: int,
        next_continuation_given_state: list,
        emission_given_state: list,
    ) -> ProductUnit:
        """
        Build ``P(observation_position, ..., observation_{T-1} | state_position = state)``:
        the current position's own emission, paired with a sum over the next state of
        its continuation weighted by the transition probability.

        :param result: The circuit every new node is added to.
        :param position: The position this joint is built for.
        :param state: The state of ``position`` this joint is conditioned on.
        :param next_continuation_given_state: Position ``position + 1``'s already-built
            continuations, indexed by state.
        :param emission_given_state: ``emission_given_state[t][s]`` is
            ``P(observation_t | state_t = s)``.
        :return: The joint's root.
        """
        transition = SumUnit(probabilistic_circuit=result)
        for next_state in range(self._state_count):
            transition.add_subcircuit(
                next_continuation_given_state[next_state],
                np.log(self._transition_probability(state, next_state)),
            )
        joint = ProductUnit(probabilistic_circuit=result)
        joint.add_subcircuit(emission_given_state[position][state])
        joint.add_subcircuit(transition)
        return joint
