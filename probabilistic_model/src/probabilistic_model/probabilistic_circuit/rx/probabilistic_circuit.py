from __future__ import annotations
import numpy.typing as npt

import copy
import functools
import itertools
import math
import queue
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import rustworkx as rx
import rustworkx.visualization
from sortedcontainers import SortedSet
from typing_extensions import (
    List,
    Optional,
    Any,
    Self,
    Dict,
    Tuple,
    Iterable,
    Callable,
    Union,
)

from probabilistic_model.distributions.distributions import (
    UnivariateDistribution,
    IntegerDistribution,
    SymbolicDistribution,
    DiscreteDistribution,
    ContinuousDistribution,
)
from probabilistic_model.distributions.helper import make_dirac
from probabilistic_model.exceptions import IntractableError
from probabilistic_model.probabilistic_circuit.groundable_circuit import (
    GroundableCircuit,
    ProductsOfCopies,
)
from probabilistic_model.probabilistic_circuit.probabilistic_circuit_representation import (
    MixtureOverVariable,
    ProbabilisticCircuitRepresentation,
    UnnormalizedMixture,
)
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    DenseEdgeValues,
)
from probabilistic_model.probabilistic_model import (
    PartialPointType,
    ProbabilisticModel,
    OrderType,
    CenterType,
    MomentType,
)
from probabilistic_model.query_results import LogConditioning, LogMode, LogTruncation
from probabilistic_model.utils import MissingDict, logsumexp
from random_events.interval import SimpleInterval, Interval
from random_events.product_algebra import VariableMap, SimpleEvent, Event
from random_events.set import Set
from krrood.adapters.json_serializer import SubclassJSONSerializer, to_json, from_json
from random_events.variable import Variable, Symbolic, Continuous, Integer


def invalidates_topology_cache(method):
    """
    Decorator for :class:`ProbabilisticCircuit` methods that change the graph topology.

    After the wrapped method has run, the circuit's cached root and layers are
    invalidated so that they are recomputed on the next access. Use this only for
    methods whose effect on the topology is unconditional and complete by the time they
    return; methods that invalidate conditionally (e.g. only when a new edge is actually
    created) or that need a valid cache part-way through their own body invalidate the
    cache explicitly instead.

    :param method: The method to wrap.
    :return: The wrapped method.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        self._invalidate_topology_cache()
        return result

    return wrapper


class PlotAlignment(IntEnum):
    HORIZONTAL = 0
    VERTICAL = 1


@dataclass
class WeightedLeafProduct:
    """
    A product of leaves with the logarithmic weight it carries in a mixture.
    """

    log_weight: float
    """
    The logarithmic weight of the product.
    """

    leaves: list[LeafUnit]
    """
    The leaves whose distributions are multiplied.
    """


@dataclass
class Unit(SubclassJSONSerializer, ABC):
    """
    Class for all units of a probabilistic circuit.

    This class should not be used by users directly. Use :class:`ProbabilisticCircuit`
    as interface to users.
    """

    probabilistic_circuit: ProbabilisticCircuit | None = field(
        kw_only=True, repr=False, default=None
    )
    """
    The circuit this component is part of.
    """

    result_of_current_query: Any = field(init=False, default=None, repr=False)
    """
    The result of the current query.
    """

    index: int | None = field(kw_only=True, default=None, repr=False)
    """
    The index of the node in the graph of its circuit.
    """

    def __post_init__(self):
        if self.probabilistic_circuit is not None:
            self.probabilistic_circuit.add_node(self)

    @property
    def subcircuits(self) -> list[Unit]:
        """
        :return: The subcircuits of this unit.
        """
        return self.probabilistic_circuit.successors(self)

    @property
    def parents(self) -> list[InnerUnit]:
        """
        :return: The parents of this unit.
        """
        return self.probabilistic_circuit.predecessors(self)

    @abstractmethod
    def support(self):
        """
        Calculate the support of this unit.
        """
        raise NotImplementedError

    @abstractmethod
    def weighted_leaf_products(self) -> list[WeightedLeafProduct]:
        """
        :return: The weighted products of leaves whose mixture is the distribution of
            this unit.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_leaf(self):
        """
        :return: If this unit is a leaf unit.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def variables(self) -> SortedSet:
        raise NotImplementedError

    def is_product_over(self, variables: SortedSet[Variable]) -> bool:
        """
        :param variables: Variables.
        :return: Whether this unit is a product unit that models all of the variables.
        """
        return False

    def mixture_components(self) -> list[Unit]:
        """
        :return: Nothing, since a unit that does not mix has no components. A unit that
            mixes answers the subcircuits it mixes.
        """
        return []

    @property
    def log_normalization_constant(self) -> float:
        """
        :return: ``0.0``, since a unit that mixes nothing has no weights to normalize.
            A unit that mixes answers what its weights sum to.
        """
        return 0.0

    @property
    def leaves(self) -> list[LeafUnit]:
        """
        :return: The leaves of the circuit that are descendants of this node.
        """
        return [
            unit
            for unit in self.probabilistic_circuit.descendants(self)
            if unit.is_leaf
        ]

    def update_variables(self, new_variables: VariableMap):
        """
        Update the variables of this unit and its descendants.

        :param new_variables: A map that maps the variables that should be replaced to
            their new variable.
        """
        for leaf in self.leaves:
            for variable in leaf.variables:
                if variable in new_variables:
                    leaf.distribution.variable = new_variables[leaf.variable]

    def connect_incoming_edges_to(self, other: Unit):
        """
        Connect all incoming edges to this unit to another unit.

        :param other: The other unit to connect the incoming edges to.
        """
        [
            self.probabilistic_circuit.add_edge(parent, other, data)
            for parent, _, data in self.probabilistic_circuit.in_edges(self)
        ]

    def filter_variable_map_by_self(self, variable_map: VariableMap):
        """
        Filter a variable map by the variables of this unit.

        :param variable_map: The map to filter
        :return: The map filtered by the variables of this unit.
        """
        variables = self.variables
        return variable_map.__class__(
            {
                variable: value
                for variable, value in variable_map.items()
                if variable in variables
            }
        )

    @property
    def impossible_condition_result(self) -> LogTruncation[Unit]:
        """
        :return: The result of an impossible truncated query.
        """
        return LogTruncation.impossible()

    def log_mode(self):
        raise NotImplementedError

    def __hash__(self):
        if self.probabilistic_circuit is not None and self.index is not None:
            return hash((self.index, id(self.probabilistic_circuit)))
        else:
            return id(self)

    def copy_without_graph(self, memo: dict[int, Any] | None = None) -> Self:
        """
        :param memo: The objects already copied during the deep copy this copy is part
            of, by their id.
        :return: A deep copy of this unit that belongs to no circuit.
        """
        result = self.empty_copy()
        result.result_of_current_query = self.result_of_current_query
        return result

    def empty_copy(self) -> Self:
        """
        Creat a copy of this circuit without any subcircuits.

        Only the parameters should be copied. This is used whenever a new circuit has to
        be created during inference.

        :return: A copy of this circuit without any subcircuits that is not in this
            units graph.
        """
        return self.__class__()

    def simplify(self):
        """
        Simplify the circuit by removing nodes and redirected edges that have no impact
        in-place. Essentially, this method transforms the circuit into an alternating
        order of sum and product units.

        :return: The simplified circuit.
        """
        raise NotImplementedError()

    def sample(self, *args, **kwargs):
        """
        Draw samples from the circuit.

        For sampling, a node gets requested a number of samples from all his parents.
        The parents write into the `result_of_current_query` attribute a tuple
        describing the beginning index of the sampling and how many samples are
        requested.
        """
        raise NotImplementedError

    def marginal(self, *args, **kwargs) -> Self | None:
        """
        Remove nodes that are not part of the marginal distribution.
        """
        raise NotImplementedError

    @abstractmethod
    def moment(self, *args, **kwargs):
        raise NotImplementedError


@dataclass(eq=False)
class LeafUnit(Unit):
    """
    Class for Leaf units.
    """

    distribution: ProbabilisticModel | None
    """
    The distribution contained in this leaf unit.
    """

    def __repr__(self):
        return f"leaf({repr(self.distribution)}"

    @property
    def variables(self) -> Iterable[Variable]:
        return SortedSet(self.distribution.variables)

    @property
    def subcircuits(self) -> list[Unit]:
        return []

    @property
    def is_leaf(self):
        return True

    @property
    def leaves(self) -> list[LeafUnit]:
        return [self]

    def weighted_leaf_products(self) -> list[WeightedLeafProduct]:
        return [WeightedLeafProduct(log_weight=0.0, leaves=[self])]

    def log_likelihood(self, events: npt.NDArray):
        self.result_of_current_query = self.distribution.log_likelihood(events)

    def cumulative_distribution(self, events: npt.NDArray):
        self.result_of_current_query = (
            self.distribution.cumulative_distribution_function(events)
        )

    def probability_of_simple_event(self, event: SimpleEvent):
        self.result_of_current_query = self.distribution.probability_of_simple_event(
            event
        )

    def support(self):
        self.result_of_current_query = self.distribution.support

    def simplify(self):
        if self.distribution is None:
            self.probabilistic_circuit.remove_node(self)

    def log_truncated_of_simple_event_in_place(
        self, event: SimpleEvent, singleton_allowed: bool = False
    ):
        self.adopt_truncation(
            self.distribution.log_truncated(event.as_composite_set(), singleton_allowed)
        )

    def adopt_truncation(self, truncation: LogTruncation) -> None:
        """
        Replace the distribution of this leaf by its truncation, whose log-probability
        becomes the result of the current query.

        :param truncation: The truncation of the distribution of this leaf.
        """
        self.distribution = truncation.distribution
        self.result_of_current_query = truncation.log_probability

    def adopt_conditioning(self, conditioning: LogConditioning) -> None:
        """
        Replace the distribution of this leaf by its conditioning, whose log-likelihood
        becomes the result of the current query.

        :param conditioning: The conditioning of the distribution of this leaf.
        """
        self.distribution = conditioning.distribution
        self.result_of_current_query = conditioning.log_likelihood

    def moment(self, order, center, variable_to_index_map):
        result = np.zeros(len(variable_to_index_map))

        # check if this nodes distribution is queried
        requested_variables = set(order.keys())
        if set(self.distribution.variables).issubset(requested_variables):
            moment = self.distribution.moment(order, center)
            for variable in self.variables:
                result[variable_to_index_map[variable]] = moment[variable]
        self.result_of_current_query = result

    def sample(self, samples: npt.NDArray, variable_to_index_map: dict[Variable, int]):
        """
        Sample from the distribution and write the samples into the samples array.

        During sampling each node accumulates, in ``result_of_current_query``, the
        indices of the rows in ``samples`` that are routed to it (as a list of index
        arrays, one per parent contribution). This leaf draws all of its samples in a
        single batched call instead of once per request.

        :param samples: The array to write the samples into.
        :param variable_to_index_map: The map from variables to column indices in the
            samples array.
        """
        # a subcircuit legitimately receives no rows when an ancestor mixture
        # assigns it zero samples; there is then nothing for this leaf to draw
        if not self.result_of_current_query:
            return
        rows = np.concatenate(self.result_of_current_query)
        column_indices = [
            variable_to_index_map[variable] for variable in self.distribution.variables
        ]
        samples[rows[:, None], column_indices] = self.distribution.sample(len(rows))

    def marginal(self, variables: Iterable[Variable]) -> Self | None:
        marginal = self.distribution.marginal(variables)
        if marginal is None:
            self.probabilistic_circuit.remove_node(self)
            return None
        else:
            self.distribution = marginal
            return self

    def log_mode(self):
        self.result_of_current_query = self.distribution.log_mode()

    def to_json(self):
        result = super().to_json()
        result["distribution"] = to_json(self.distribution)
        return result

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        distribution = from_json(data["distribution"])
        return cls(distribution)

    def log_conditional_in_place(self, point: dict[Variable, Any]):
        if any(variable for variable in self.variables if variable in point):
            self.adopt_conditioning(self.distribution.log_conditional(point))
        else:
            self.result_of_current_query = 0.0

    def copy_without_graph(self, memo: dict[int, Any] | None = None) -> Self:
        return self.__class__(distribution=copy.deepcopy(self.distribution, memo))

    def replace_by_mixture(
        self, truncations: Iterable[LogTruncation]
    ) -> SumUnit | None:
        """
        Replace this leaf by the mixture of the given truncations of its distribution,
        each weighted by its probability.

        :param truncations: Each truncated distribution with its log-probability.
        :return: The sum unit that replaced this leaf, or nothing if every truncation is
            impossible, in which case this leaf is left without a distribution.
        """
        result = SumUnit(probabilistic_circuit=self.probabilistic_circuit)
        total_probability = 0.0

        for truncation in truncations:
            probability = np.exp(truncation.log_probability)
            if probability == 0:
                continue
            result.add_subcircuit(
                self.__class__(
                    distribution=truncation.distribution,
                    probabilistic_circuit=self.probabilistic_circuit,
                ),
                truncation.log_probability,
            )
            total_probability += probability

        if total_probability == 0:
            self.result_of_current_query = -np.inf
            self.distribution = None
            self.probabilistic_circuit.remove_node(result)
            return None

        self.connect_incoming_edges_to(result)
        self.probabilistic_circuit.remove_node(self)

        result.normalize()
        result.result_of_current_query = np.log(total_probability)
        return result


@dataclass(eq=False)
class InnerUnit(Unit, ABC):
    """
    Class for inner units.
    """

    @property
    def is_leaf(self):
        return False

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def log_forward(self, *args, **kwargs):
        raise NotImplementedError

    def marginal(self, *args, **kwargs) -> Self | None:
        if len(self.subcircuits) == 0:
            self.probabilistic_circuit.remove_node(self)
            return None
        return self

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        return cls()

    def add_subcircuit(self, subcircuit: Unit, log_weight: float = None):
        """
        Add a subcircuit to the subcircuits of this unit.

        .. note::

            This method does not normalize the edges to the subcircuits.


        :param subcircuit: The subcircuit to add.
        :param log_weight: The logarithmic weight of the subcircuit.
        Only needed if this is a sum unit
        """
        self.probabilistic_circuit.add_edge(self, subcircuit, log_weight)


@dataclass(eq=False)
class SumUnit(InnerUnit):
    _latent_variable: Symbolic | None = None
    """
    The latent variable of this unit.

    This has to be here due to the rvalue/lvalue problem in random events.

    TODO remove this when RE is fixed
    """

    def __repr__(self):
        return "⊕"

    __hash__ = Unit.__hash__

    @property
    def representation(self) -> str:
        return "+"

    def mixture_components(self) -> list[Unit]:
        return self.subcircuits

    @property
    def log_normalization_constant(self) -> float:
        """
        :return: ``log(sum(exp(w)))`` over the weights of the components this unit
            mixes, which is ``0.0`` while it mixes nothing.
        """
        log_weights = self.log_weights
        if len(log_weights) == 0:
            return 0.0
        return float(logsumexp(log_weights))

    @property
    def log_weighted_subcircuits(self) -> list[tuple[float, Unit]]:
        """
        :return: The weighted subcircuits of this unit.
        """
        return [
            (
                self.probabilistic_circuit.graph.get_edge_data(
                    self.index, subcircuit.index
                ),
                subcircuit,
            )
            for subcircuit in self.subcircuits
        ]

    def weighted_leaf_products(self) -> list[WeightedLeafProduct]:
        return [
            WeightedLeafProduct(
                log_weight=log_weight + product.log_weight, leaves=product.leaves
            )
            for log_weight, subcircuit in self.log_weighted_subcircuits
            for product in subcircuit.weighted_leaf_products()
        ]

    @property
    def variables(self) -> SortedSet:
        return self.subcircuits[0].variables

    @property
    def latent_variable(self) -> Symbolic:
        name = f"{hash(self)}.latent"
        subcircuit_enum = IntEnum(
            name,
            {
                str(hash(subcircuit)): index
                for index, subcircuit in enumerate(self.subcircuits)
            },
        )
        result = Symbolic(name=name, domain=Set.from_iterable(subcircuit_enum))
        self._latent_variable = result
        return result

    def forward(self, *args, **kwargs):
        # streamed instead of building the full subcircuit-results list at once
        result = 0.0
        for weight, subcircuit in self.log_weighted_subcircuits:
            result = result + np.exp(weight) * subcircuit.result_of_current_query
        self.result_of_current_query = result

    def log_forward(self, *args, **kwargs):
        # streamed via logaddexp instead of building the full list for logsumexp
        result = None
        for log_weight, subcircuit in self.log_weighted_subcircuits:
            value = log_weight + subcircuit.result_of_current_query
            result = value if result is None else np.logaddexp(result, value)
        self.result_of_current_query = result

    moment = forward

    def log_forward_conditioning(self, *args, **kwargs):
        result = [
            lw + s.result_of_current_query for lw, s in self.log_weighted_subcircuits
        ]

        # update weights according to bayes rule
        for new_weight, subcircuit in zip(result, self.subcircuits):
            self.probabilistic_circuit.add_edge(self, subcircuit, log_weight=new_weight)

        self.result_of_current_query = logsumexp(result, axis=0)

    def support(self):
        support = self.subcircuits[0].result_of_current_query
        for subcircuit in self.subcircuits[1:]:
            support |= copy.deepcopy(subcircuit.result_of_current_query)
        self.result_of_current_query = support

    @property
    def log_weights(self) -> npt.NDArray:
        """
        :return: The log_weights of the subcircuits.
        """
        return np.array([weight for weight, _ in self.log_weighted_subcircuits])

    def sample(self, *args, **kwargs):
        """
        Route the sample rows accumulated from this unit's parents to its subcircuits.

        Every row routed to a mixture is assigned to exactly one subcircuit, drawn
        according to the subcircuit weights. The rows are partitioned in a single
        multinomial draw rather than once per parent request, which keeps the work
        proportional to the number of samples instead of the number of paths through the
        circuit.
        """
        # a subcircuit legitimately receives no rows when an ancestor mixture
        # assigns it zero samples; there is then nothing to route onward
        if not self.result_of_current_query:
            return

        # all sample rows routed to this unit by its parents
        rows = np.concatenate(self.result_of_current_query)

        # fetch weights and subcircuits together to keep them aligned
        log_weighted_subcircuits = self.log_weighted_subcircuits
        weights = np.exp([log_weight for log_weight, _ in log_weighted_subcircuits])

        # assign every row to one subcircuit according to the weights
        counts = np.random.multinomial(len(rows), pvals=weights)

        # shuffle the rows so the contiguous chunks handed to the subcircuits are
        # an unbiased partition
        np.random.shuffle(rows)

        offset = 0
        for count, (_, subcircuit) in zip(counts, log_weighted_subcircuits):
            if not count:
                continue
            subcircuit.result_of_current_query.append(rows[offset : offset + count])
            offset += count

    def mount_with_interaction_terms(
        self, other: Self, interaction_model: ProbabilisticModel
    ):
        r"""
        Create a distribution that factorizes as follows:

        .. math::
            p(self.latent\_variable) \cdot p(self.variables | self.latent\_variable) \cdot
            p(other.latent\_variable | self.latent\_variable) \cdot p(other.variables | other.latent\_variable)

        where `self.latent_variable` and `other.latent_variable` are the results of the latent variable interpretation
        of mixture models.

        :param other: The other distribution to mount at this distribution children level.
        :param interaction_model: The interaction probabilities between both latent variables
        """
        assert set(self.variables).intersection(set(other.variables)) == set()
        assert set(interaction_model.variables) == {
            self.latent_variable,
            other.latent_variable,
        }

        # load latent variables
        own_latent_variable = self.latent_variable
        other_latent_variable = other.latent_variable

        # load subircuits
        own_subcircuits = self.subcircuits
        other_subcircuits = other.subcircuits

        for own_index, own_subcircuit in enumerate(own_subcircuits):

            # create denominator of weight
            own_index = own_latent_variable.domain.simple_sets[0].element.__class__(
                own_index
            )
            condition = SimpleEvent.from_data(
                {own_latent_variable: own_index}
            ).as_composite_set()
            p_condition = interaction_model.probability(condition)

            # skip iterations that are impossible
            if p_condition == 0:
                continue

            # create proxy nodes for mounting
            proxy_product_node = ProductUnit()
            proxy_sum_node = other.empty_copy()
            self.probabilistic_circuit.add_nodes_from(
                [proxy_product_node, proxy_sum_node]
            )

            # remove edge to old child and replace it by product proxy
            self.probabilistic_circuit.remove_edge(self, own_subcircuit)
            self.add_subcircuit(proxy_product_node, np.log(p_condition))

            # mount current child on the product proxy
            proxy_product_node.add_subcircuit(own_subcircuit)

            # mount the proxy for the children from other in the product proxy
            proxy_product_node.add_subcircuit(proxy_sum_node)

            for other_index, other_subcircuit in enumerate(other_subcircuits):

                # create numerator of weight
                other_index = other_latent_variable.domain.simple_sets[
                    0
                ].element.__class__(other_index)
                query = (
                    SimpleEvent.from_data(
                        {other_latent_variable: other_index}
                    ).as_composite_set()
                    & condition
                )
                p_query = interaction_model.probability(query)

                # skip iterations that are impossible
                if p_query == 0:
                    continue

                # calculate truncated probability
                weight = p_query / p_condition

                # create edge from proxy to subcircuit
                proxy_sum_node.add_subcircuit(
                    other_subcircuit, log_weight=np.log(weight)
                )
            proxy_sum_node.normalize()

    def mount_from_bayesian_network(self, other: SumUnit):
        """
        Mount a distribution from tge `to_probabilistic_circuit` method in bayesian networks.
        The distribution is mounted as follows:


        :param other: The other distribution to mount at this distribution children level.
        :return:
        """
        assert set(self.variables).intersection(set(other.variables)) == set()
        assert len(self.subcircuits) == len(other.subcircuits)
        # mount the other subcircuit

        for (own_weight, own_subcircuit), other_subcircuit in zip(
            self.log_weighted_subcircuits, other.subcircuits
        ):
            # create proxy nodes for mounting
            proxy_product_node = ProductUnit()
            self.probabilistic_circuit.add_node(proxy_product_node)

            # remove edge to old child and replace it by product proxy
            self.probabilistic_circuit.remove_edge(self, own_subcircuit)
            self.add_subcircuit(proxy_product_node, np.log(own_weight))
            proxy_product_node.add_subcircuit(own_subcircuit)
            proxy_product_node.add_subcircuit(other_subcircuit)

    def simplify(self):

        # if this has only one child
        if len(self.subcircuits) == 1:

            # redirect every incoming edge to the child
            incoming_edges = list(self.probabilistic_circuit.in_edges(self))
            for parent, _, data in incoming_edges:
                self.probabilistic_circuit.add_edge(parent, self.subcircuits[0], data)

            # remove this node
            self.probabilistic_circuit.remove_node(self)

            return

        # for every subcircuit
        for weight, subcircuit in self.log_weighted_subcircuits:

            # if the weight is 0, skip this subcircuit
            if weight == -np.inf:
                # remove the edge
                self.probabilistic_circuit.remove_edge(self, subcircuit)

            # if the simplified subcircuit is of the same type as this
            if type(subcircuit) is type(self):

                # type hinting
                subcircuit: Self

                # mount the children of that circuit directly
                for sub_weight, sub_subcircuit in subcircuit.log_weighted_subcircuits:
                    new_weight = sub_weight + weight

                    # add an edge to that subcircuit
                    self.add_subcircuit(sub_subcircuit, new_weight)

                if self.probabilistic_circuit.graph.has_edge(
                    self.index, subcircuit.index
                ):
                    self.probabilistic_circuit.remove_edge(self, subcircuit)
                if not self.probabilistic_circuit.in_edges(subcircuit):
                    self.probabilistic_circuit.remove_node(subcircuit)

    def normalize(self):
        """
        Normalize the log_weights of the subcircuits such that they sum up to 1 inplace.
        """
        total_weight = logsumexp(self.log_weights)
        for log_weight, subcircuit in self.log_weighted_subcircuits:
            self.probabilistic_circuit.graph.add_edge(
                self.index, subcircuit.index, log_weight - total_weight
            )

    def is_normalized(self, tolerance: float = 1e-6) -> bool:
        """
        Return True iff this SumUnit's log-weights sum to log(1) == 0.

        Uses logsumexp for numerical stability, matching normalize(). An empty SumUnit
        (no subcircuits) is considered normalized.

        :param tolerance: Maximum absolute deviation from 0.0 permitted.
        :returns: True if the weights are normalised within tolerance.
        """
        log_weights = self.log_weights
        if len(log_weights) == 0:
            return True
        return abs(float(logsumexp(log_weights))) < tolerance

    def is_deterministic(self) -> bool:
        """
        :return: If this unit is deterministic or not.
        """
        # for every unique combination of subcircuits
        for subcircuit_a, subcircuit_b in itertools.combinations(self.subcircuits, 2):
            # check if they intersect
            if not subcircuit_a.result_of_current_query.intersection_with(
                subcircuit_b.result_of_current_query
            ).is_empty():
                return False

        # if none intersect, the subcircuit is deterministic
        return True

    def log_mode(self):
        log_maxima = [
            log_weight + subcircuit.result_of_current_query.log_likelihood
            for log_weight, subcircuit in self.log_weighted_subcircuits
        ]
        log_max = max(log_maxima)
        arg_log_maxima = [
            subcircuit.result_of_current_query.event
            for lm, subcircuit in zip(log_maxima, self.subcircuits)
            if lm == log_max
        ]
        arg_log_max = arg_log_maxima[0]
        for event in arg_log_maxima[1:]:
            arg_log_max |= event
        self.result_of_current_query = LogMode(arg_log_max, log_max)

    def subcircuit_index_of_samples(self, samples: npt.NDArray) -> npt.NDArray:
        """
        :return: the index of the subcircuit where p(sample) > 0 and None if p(sample) = 0 for all subcircuits.
        """
        result = np.full(len(samples), np.nan)
        for index, subcircuit in enumerate(self.subcircuits):
            likelihood = subcircuit.log_likelihood(samples)
            result[likelihood > -np.inf] = index
        return result


@dataclass(eq=False)
class ProductUnit(InnerUnit):
    """
    Decomposable Product Units for Probabilistic Circuits.
    """

    representation = "×"
    __hash__ = Unit.__hash__

    def __repr__(self):
        return "⊗"

    def forward(self, *args, **kwargs):
        self.result_of_current_query = math.prod(
            [subcircuit.result_of_current_query for subcircuit in self.subcircuits]
        )

    def log_forward(self, *args, **kwargs):
        self.result_of_current_query = np.sum(
            [subcircuit.result_of_current_query for subcircuit in self.subcircuits],
            axis=0,
        )

    moment = log_forward

    @property
    def variables(self) -> SortedSet:
        result = SortedSet()
        for subcircuit in self.subcircuits:
            result = result.union(subcircuit.variables)
        return result

    def is_product_over(self, variables: SortedSet[Variable]) -> bool:
        return variables.issubset(self.variables)

    def support(self):
        support: Event = self.subcircuits[0].result_of_current_query
        support.fill_missing_variables(self.variables)

        for subcircuit in self.subcircuits[1:]:
            support &= subcircuit.result_of_current_query

        self.result_of_current_query = support

    def weighted_leaf_products(self) -> list[WeightedLeafProduct]:
        return [
            WeightedLeafProduct(
                log_weight=sum(factor.log_weight for factor in factors),
                leaves=[leaf_unit for factor in factors for leaf_unit in factor.leaves],
            )
            for factors in itertools.product(
                *(
                    subcircuit.weighted_leaf_products()
                    for subcircuit in self.subcircuits
                )
            )
        ]

    def is_decomposable(self):
        for index, subcircuit in enumerate(self.subcircuits):
            variables = subcircuit.variables
            for subcircuit_ in self.subcircuits[index + 1 :]:
                if len(set(subcircuit_.variables).intersection(set(variables))) > 0:
                    return False
        return True

    def log_mode(self):
        first_mode = self.subcircuits[0].result_of_current_query
        arg_log_max, log_max = first_mode.event, first_mode.log_likelihood
        arg_log_max.fill_missing_variables(self.variables)
        for subcircuit in self.subcircuits[1:]:
            arg_log_max = arg_log_max.intersection_with(
                subcircuit.result_of_current_query.event
            )
            log_max += subcircuit.result_of_current_query.log_likelihood
        self.result_of_current_query = LogMode(arg_log_max, log_max)

    def simplify(self):

        # if this has only one child
        if len(self.subcircuits) == 1:
            self.connect_incoming_edges_to(self.subcircuits[0])
            self.probabilistic_circuit.remove_node(self)
            return

        # for every subcircuit
        for subcircuit in self.subcircuits:

            # if the simplified subcircuit is of the same type as this
            if type(subcircuit) is type(self):

                # type hinting
                subcircuit: Self

                # mount the children of that circuit directly onto this one
                for sub_subcircuit in subcircuit.subcircuits:
                    self.add_subcircuit(sub_subcircuit)

                if self.probabilistic_circuit.graph.has_edge(
                    self.index, subcircuit.index
                ):
                    self.probabilistic_circuit.remove_edge(self, subcircuit)
                if not self.probabilistic_circuit.in_edges(subcircuit):
                    self.probabilistic_circuit.remove_node(subcircuit)

    def sample(self, *args, **kwargs):
        """
        Route the sample rows accumulated from this unit's parents to its subcircuits.

        A decomposable product factorizes over disjoint variables, so every sample row
        is forwarded unchanged to each subcircuit; the subcircuits then fill in their
        respective columns of the same rows.
        """
        # a subcircuit legitimately receives no rows when an ancestor mixture
        # assigns it zero samples; there is then nothing to route onward
        if not self.result_of_current_query:
            return
        # a decomposable product routes every one of its rows to each subcircuit
        rows = np.concatenate(self.result_of_current_query)
        for subcircuit in self.subcircuits:
            subcircuit.result_of_current_query.append(rows)

    def attach_marginal_circuit(
        self,
        marginal_circuit: ProbabilisticCircuit,
        target_circuit: ProbabilisticCircuit,
    ) -> None:
        """
        Attach the root of marginal_circuit as a child of this ProductUnit, constructing
        fresh nodes owned by target_circuit.

        marginal() and log_truncated_in_place() return flat circuits (SumUnit -> leaves,
        or a single leaf), so one level of recursion suffices to copy all nodes into
        target_circuit.

        :param marginal_circuit: The marginal or truncated circuit whose root to attach
            as a child of this ProductUnit.
        :param target_circuit: The owning circuit for all newly created nodes.
        """
        root = marginal_circuit.root
        if isinstance(root, SumUnit):
            new_sum_unit = SumUnit(probabilistic_circuit=target_circuit)
            for child_log_weight, child_subcircuit in root.log_weighted_subcircuits:
                new_sum_unit.add_subcircuit(
                    leaf(copy.deepcopy(child_subcircuit.distribution), target_circuit),
                    child_log_weight,
                )
            self.add_subcircuit(new_sum_unit)
        else:
            self.add_subcircuit(leaf(copy.deepcopy(root.distribution), target_circuit))


@dataclass
class ProbabilisticCircuit(GroundableCircuit[Unit], SubclassJSONSerializer):
    """
    Probabilistic Circuits as a directed, rooted, acyclic graph.

    The nodes of the graph are the units of the circuit. The edges of the graph indicate
    how the units are connected. The outgoing edges of a sum unit contain the log-
    log_weights of the subcircuits.
    """

    graph: rx.PyDAG[Unit] = field(default_factory=lambda: rx.PyDAG(multigraph=False))
    """
    The graph to check connectivity from.
    """

    _root_cache: Unit | None = field(
        init=False, default=None, repr=False, compare=False
    )
    """
    Cached root unit.

    Invalidated whenever the topology of the graph changes.
    """

    _layers_cache: list[list[Unit]] | None = field(
        init=False, default=None, repr=False, compare=False
    )
    """
    Cached layers of the graph.

    Invalidated whenever the topology of the graph changes.
    """

    def _invalidate_topology_cache(self):
        """
        Invalidate the cached root and layers.

        This must be called whenever nodes or edges are added to or removed from the
        graph. Pure edge-weight updates (which do not change the topology) do not
        require invalidation. The :func:`invalidates_topology_cache` decorator calls
        this automatically after a mutator; call it directly only from mutators whose
        invalidation is conditional (:meth:`add_node`, :meth:`add_edge`).
        """
        self._root_cache = None
        self._layers_cache = None

    def __len__(self):
        """
        Return the number of nodes in the graph.

        :return: The number of nodes in the graph.
        """
        return len(self.graph)

    def __iter__(self):
        """
        Return an iterator over the nodes in the graph.

        :return: An iterator over the nodes in the graph.
        """
        return iter(self.graph.nodes())

    @classmethod
    def from_other(cls, other: Self) -> Self:
        result = cls()
        result.add_edges_and_nodes_from_circuit(other)
        return result

    @property
    def variables(self) -> SortedSet:
        return self.root.variables

    @property
    def variable_to_index_map(self) -> dict[Variable, int]:
        return {variable: index for index, variable in enumerate(self.variables)}

    @property
    def layers(self) -> list[list[Unit]]:
        if self._layers_cache is None:
            self._layers_cache = rx.layers(
                self.graph, [self.root.index], index_output=False
            )
        return self._layers_cache

    @property
    def leaves(self) -> list[LeafUnit]:
        return self.root.leaves

    def is_valid(self) -> bool:
        """
        Check if this graph is:

        - acyclic
        - connected

        :return: True if the graph is valid, False otherwise.
        """
        return rx.is_directed_acyclic_graph(self.graph) and self.root

    def add_node(self, node: Unit):
        # invalidation is conditional here (a node that already belongs to this
        # circuit is a no-op), so it is done inline rather than via
        # ``@invalidates_topology_cache``
        if node.probabilistic_circuit is self and node.index is not None:
            return
        elif (
            node.probabilistic_circuit is not None
            and node.probabilistic_circuit is not self
        ):
            raise NotImplementedError(
                "Cannot add a node that already belongs to another circuit."
            )

        node.index = self.graph.add_node(node)

        # write self as the nodes' circuit
        node.probabilistic_circuit = self
        self._invalidate_topology_cache()

    def add_nodes_from(self, units: Iterable[Unit]):
        [self.add_node(node) for node in units]

    def add_edge(self, parent: Unit, child: Unit, log_weight: float | None = None):
        self.add_node(parent)
        self.add_node(child)
        # invalidation is conditional here: only creating a new edge changes the
        # topology, while updating the weight of an existing edge leaves root and
        # layers untouched. It is therefore done inline rather than via
        # ``@invalidates_topology_cache``
        if not self.graph.has_edge(parent.index, child.index):
            self._invalidate_topology_cache()
        self.graph.add_edge(parent.index, child.index, log_weight)

    def add_edges_from(
        self, edges: Iterable[tuple[Unit, Unit] | tuple[Unit, Unit, float]]
    ):
        [self.add_edge(*edge) for edge in edges]

    def successors(self, unit: Unit) -> list[Unit]:
        return self.graph.successors(unit.index)

    def descendants(self, unit: Unit) -> Set[Unit]:
        return {self.graph[unit] for unit in rx.descendants(self.graph, unit.index)}

    def predecessors(self, unit: Unit) -> list[InnerUnit]:
        return self.graph.predecessors(unit.index)

    @invalidates_topology_cache
    def remove_node(self, unit: Unit):
        self.graph.remove_node(unit.index)
        unit.index = None
        unit.probabilistic_circuit = None

    def remove_nodes_from(self, units: Iterable[Unit]):
        [self.remove_node(unit) for unit in units]

    @invalidates_topology_cache
    def remove_edge(self, parent: Unit, child: Unit):
        self.graph.remove_edge(parent.index, child.index)

    def remove_edges_from(self, edges: Iterable[tuple[Unit, Unit]]):
        [self.remove_edge(*edge) for edge in edges]

    def in_edges(self, unit: Unit) -> list[tuple[Unit, Unit, float | None]]:
        return [
            (
                self.graph.get_node_data(parent_index),
                unit,
                edge_data,
            )
            for parent_index, _, edge_data in self.graph.in_edges(unit.index)
        ]

    @invalidates_topology_cache
    def add_from_subgraph(self, subgraph: rx.PyDAG[Unit]) -> dict[int, Unit]:
        """
        Add nodes and edges from a subgraph to this circuit.

        :param subgraph: The subgraph to add nodes from.
        :return: A dictionary mapping the node indices in the subgraph to the new units
            in this circuit.
        """
        new_nodes = {node.index: node.copy_without_graph() for node in subgraph.nodes()}
        self.add_nodes_from(new_nodes.values())

        [
            self.graph.add_edge(
                new_nodes[subgraph.get_node_data(parent).index].index,
                new_nodes[subgraph.get_node_data(child).index].index,
                subgraph.get_edge_data(parent, child),
            )
            for parent, child in subgraph.edge_list()
        ]
        return new_nodes

    def nodes(self) -> list[Unit]:
        """
        Return an iterator over the nodes.

        :return: An iterator over the nodes.
        """
        return self.graph.nodes()

    def edges(self) -> list[tuple[Unit, Unit, float | None]]:
        return [
            (
                self.graph[parent],
                self.graph[child],
                self.graph.get_edge_data(parent, child),
            )
            for parent, child in self.graph.edge_list()
        ]

    def in_degree(self, unit: Unit):
        return self.graph.in_degree(unit.index)

    def has_edge(self, parent: Unit, child: Unit) -> bool:
        return self.graph.has_edge(parent.index, child.index)

    @property
    def root(self) -> Unit:
        """
        The root of the circuit is the node with in-degree 0.

        This is the output node, that will perform the final computation.

        :return: The root of the circuit.
        """
        if self._root_cache is not None:
            return self._root_cache

        # find all nodes with in-degree 0 at the index level to avoid building
        # intermediate Unit lists for the whole graph
        possible_roots = [
            index
            for index in self.graph.node_indices()
            if self.graph.in_degree(index) == 0
        ]
        if len(possible_roots) == 1:
            self._root_cache = self.graph[possible_roots[0]]
            return self._root_cache
        elif len(possible_roots) > 1:
            raise ValueError(
                f"More than one root found. Possible roots are "
                f"{[self.graph[index] for index in possible_roots]}"
            )
        else:
            raise ValueError(f"No root found.")

    def log_likelihood(self, events: npt.NDArray) -> npt.NDArray:
        variable_to_index_map = self.variable_to_index_map
        for layer in reversed(self.layers):
            for unit in layer:  # open all the procesess
                if unit.is_leaf:
                    unit: LeafUnit
                    unit.log_likelihood(
                        events[
                            :,
                            [
                                variable_to_index_map[variable]
                                for variable in unit.distribution.variables
                            ],
                        ]
                    )
                else:
                    unit: InnerUnit
                    unit.log_forward()  # Synch trheads 1
        return self.root.result_of_current_query

    def cumulative_distribution_function(self, events: npt.NDArray) -> npt.NDArray:
        variable_to_index_map = self.variable_to_index_map
        for layer in reversed(self.layers):
            for unit in layer:
                unit: LeafUnit
                if unit.is_leaf:
                    unit.cumulative_distribution(
                        events[
                            :,
                            [
                                variable_to_index_map[variable]
                                for variable in unit.distribution.variables
                            ],
                        ]
                    )
                else:
                    unit: InnerUnit
                    unit.forward()
        return self.root.result_of_current_query

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        for layer in reversed(self.layers):
            for unit in layer:
                if unit.is_leaf:
                    unit: LeafUnit
                    unit.probability_of_simple_event(event)
                else:
                    unit: InnerUnit
                    unit.forward()
        return self.root.result_of_current_query

    def log_mode(self, check_determinism: bool = True) -> LogMode:
        if check_determinism:
            if not self.is_deterministic():
                raise IntractableError(self)
        [unit.log_mode() for layer in reversed(self.layers) for unit in layer]
        return self.root.result_of_current_query

    def remove_unreachable_nodes(self, root: Unit):
        """
        Remove all nodes that are not reachable from the root.
        """
        reachable_nodes = self.descendants(root)
        unreachable_nodes = set(self.graph.nodes()) - (reachable_nodes | {root})
        self.remove_nodes_from(unreachable_nodes)

    def log_truncated_of_simple_event_in_place(
        self, simple_event: SimpleEvent, singleton_allowed: bool = False
    ) -> LogTruncation[Self]:
        """
        Construct the truncated circuit from a simple event.

        :param simple_event: The simple event to condition on.
        :param singleton_allowed: Whether to allow singletons in the simple sets of the
            event.
        :return: The truncated circuit and the log-probability of the event
        """
        for layer in reversed(self.layers):
            for unit in layer:
                if unit.is_leaf:
                    unit: LeafUnit
                    unit.log_truncated_of_simple_event_in_place(
                        simple_event, singleton_allowed
                    )
                else:
                    unit: InnerUnit
                    if isinstance(unit, SumUnit):
                        unit.log_forward_conditioning()
                    else:
                        unit.log_forward()

        root = self.root
        [
            self.remove_node(node)
            for node in self.nodes()
            if node.result_of_current_query == -np.inf
        ]

        if root not in set(self.graph.nodes()):
            return LogTruncation.impossible()

        # clean the circuit up
        self.remove_unreachable_nodes(root)
        self.simplify()
        self.normalize()

        return LogTruncation(self, root.result_of_current_query)

    def log_truncated_in_place(
        self, event: Event, singleton_allowed: bool = False
    ) -> LogTruncation[Self]:
        """
        Truncate the circuit to an Event in place.

        A composite event is handled by truncating a deep copy of the circuit to each of
        its (disjoint) simple sets and combining the results into a normalized mixture.

        :param event: The event to condition on.
        :param singleton_allowed: Whether to allow singletons in the simple sets of the
            event.
        :return: The truncated circuit and the log-probability of the event, or ``(None,
            -inf)`` if the event has zero probability.
        """
        # skip trivial case
        if event.is_empty():
            self.graph.remove_nodes_from(list(self.graph.node_indices()))
            self._invalidate_topology_cache()
            return LogTruncation.impossible()

        # if the event is easy, don't create a proxy node
        elif len(event.simple_sets) == 1:
            return self.log_truncated_of_simple_event_in_place(
                event.simple_sets[0], singleton_allowed
            )

        # truncate a deep copy of the circuit to each simple set
        truncations = list(
            copy.deepcopy(self).log_truncated_of_simple_event_in_place(
                simple_event, singleton_allowed
            )
            for simple_event in event.simple_sets
        )

        # clear this circuit
        self.remove_nodes_from(list(self.graph.nodes()))

        # filter out impossible truncations
        truncations = [
            truncation
            for truncation in truncations
            if truncation.log_probability > -np.inf
        ]

        # if all truncations are impossible
        if len(truncations) == 0:
            return LogTruncation.impossible()

        # create a new sum unit
        result = SumUnit(probabilistic_circuit=self)

        # add the truncations to the sum unit
        for truncation in truncations:
            root = truncation.distribution.root
            new_nodes = result.probabilistic_circuit.add_from_subgraph(
                truncation.distribution.graph
            )
            result.add_subcircuit(new_nodes[root.index], truncation.log_probability)

        # the simple sets of an event are disjoint, so P(E) = sum_k P(E_k)
        total_log_probability = logsumexp(
            np.array([truncation.log_probability for truncation in truncations])
        )
        result.normalize()
        return LogTruncation(self, total_log_probability)

    def log_truncated(
        self, event: Event, singleton_allowed: bool = False
    ) -> LogTruncation[Self]:
        result = copy.deepcopy(self)
        return result.log_truncated_in_place(event, singleton_allowed)

    def marginal_in_place(self, variables: Iterable[Variable]) -> Self | None:
        result = self.restrict_to_variables_in_place(variables)
        if result is None:
            return None
        self.simplify()
        return self

    def restrict_to_variables_in_place(
        self, variables: Iterable[Variable]
    ) -> Self | None:
        """
        Restrict the circuit to variables in place, without ``simplify()``'s same-type
        merge.

        :meth:`marginal_in_place` is this plus a trailing ``simplify()`` call: that call
        flattens nested SumUnits into their parent, which leaves the represented
        distribution unchanged but can erase branch boundaries a caller relies on -- for
        instance ``CausalCircuit.verify_support_determinism`` inspecting whether a
        support-deterministic circuit's own branches stay disjoint.

        :param variables: The variables to keep.
        :return:``self``, or ``None`` if none of ``variables`` are modeled.
        """
        result = [
            node.marginal(variables)
            for layer in reversed(self.layers)
            for node in layer
        ][-1]
        if result is None:
            return None
        self.remove_unreachable_nodes(result)
        return self

    def _condition_leaves_in_place(self, point: dict[Variable, Any]) -> Unit | None:
        """
        Push ``point`` into every leaf and sum unit's own forward pass, in place, and
        drop whatever that leaves with zero probability. Every unit that survives stays
        exactly where it was: this alone neither marginalizes, merges nor moves a unit.

        :param point: The value to condition each of the given variables on.
        :return: The surviving root, or ``None`` if ``point`` has no probability.
        """
        for layer in reversed(self.layers):
            for unit in layer:
                if unit.is_leaf:
                    unit: LeafUnit
                    unit.log_conditional_in_place(point)
                elif isinstance(unit, SumUnit):
                    unit.log_forward_conditioning()
                elif isinstance(unit, ProductUnit):
                    unit.log_forward()
                else:
                    raise NotImplementedError()

        root = self.root
        [
            self.graph.remove_node(node.index)
            for layer in reversed(self.layers)
            for node in layer
            if node.result_of_current_query == -np.inf
        ]
        self._invalidate_topology_cache()

        if root not in self.graph.nodes():
            return None

        self.remove_unreachable_nodes(root)
        return root

    def log_conditional_in_place(
        self, point: dict[Variable, Any], preserve_structure: bool = False
    ) -> LogConditioning[Self]:
        """
        Condition the circuit on a point, in place.

        :param point: The value to condition each of the given variables on.
        :param preserve_structure: Skip the rebuild below, which marginalizes the
            conditioned variables out and simplifies -- like :meth:`marginal_in_place`,
            unlike :meth:`restrict_to_variables_in_place`, flattening nested sums and
            erasing which branch a unit belonged to. Keeps that structure, at the cost
            of leaving the conditioned variables' point leaves in place.
        :return: The conditioned circuit, or ``None`` if the point has no probability,
            and the point's log-likelihood.
        """
        root = self._condition_leaves_in_place(point)
        if root is None:
            return LogConditioning.impossible()
        # read before marginalizing, which may simplify the root away
        log_probability = root.result_of_current_query

        if preserve_structure:
            self.normalize()
            return LogConditioning(self, log_probability)

        # simplify dirac parts
        remaining_variables = [v for v in self.variables if v not in point]

        self.marginal_in_place(remaining_variables)

        if len(remaining_variables) > 0:
            root = self.root

        # add dirac parts
        new_root = ProductUnit(probabilistic_circuit=self)

        if len(remaining_variables) > 0:
            new_root.add_subcircuit(root)

        for variable, value in point.items():
            new_root.add_subcircuit(leaf(make_dirac(variable, value), self))

        new_root.result_of_current_query = log_probability

        self.simplify()
        self.normalize()

        return LogConditioning(self, log_probability)

    def log_conditional(self, point: dict[Variable, Any]) -> LogConditioning[Self]:
        result = copy.deepcopy(self)
        return result.log_conditional_in_place(point)

    # %% operations on nodes

    def condition_on_point_in_place(self, point: PartialPointType) -> bool:
        point = {
            variable: value
            for variable, value in point.items()
            if variable in self.variables
        }
        if not point:
            return True
        conditioned = self.marginal(point).log_conditional_in_place(point).distribution
        if conditioned is None:
            return False
        self.log_conditional_in_place(point, preserve_structure=True)
        return True

    def lowest_product_nodes_over(
        self, variables: SortedSet[Variable]
    ) -> list[ProductUnit]:
        found_nodes: list[ProductUnit] = []
        ancestor_indices: set[int] = set()
        for layer in reversed(self.layers):
            for node in layer:
                if node.index in ancestor_indices:
                    continue
                if not node.is_product_over(variables):
                    continue
                found_nodes.append(node)
                ancestor_indices.add(node.index)
                ancestor_indices.update(rx.ancestors(self.graph, node.index))
        return found_nodes

    def subcircuit_at(self, node: Unit) -> Self:
        result = self.__class__()
        result.mount(node)
        return result

    def remove_variables_in_place(
        self, variables: SortedSet[Variable], nodes: list[ProductUnit]
    ) -> list[ProductUnit]:
        self.restrict_to_variables_in_place(SortedSet(self.variables) - variables)
        return nodes

    def root_mixture_components(self) -> list[Self]:
        return [
            self.subcircuit_at(subcircuit)
            for subcircuit in self.root.mixture_components()
        ]

    def multiply_nodes_by_mixtures(
        self,
        nodes: list[ProductUnit],
        products: ProductsOfCopies,
        log_weights: DenseEdgeValues | None,
    ):
        roots = [
            self._mounted_product(products, index)
            for index in range(len(products.points))
        ]
        if log_weights is None:
            for node in nodes:
                node.add_subcircuit(roots[0])
            return
        for node, node_log_weights in zip(nodes, log_weights):
            sum_unit = SumUnit(probabilistic_circuit=self)
            node.add_subcircuit(sum_unit)
            for root, log_weight in zip(roots, node_log_weights):
                if log_weight > -np.inf:
                    sum_unit.add_subcircuit(root, float(log_weight))
            sum_unit.normalize()

    def _mounted_product(self, products: ProductsOfCopies, index: int) -> ProductUnit:
        """
        :param products: The products.
        :param index: The index of a product.
        :return: The root of the product, mounted into this circuit.
        """
        root = ProductUnit(probabilistic_circuit=self)
        for copies in products.conditioned_copies:
            conditioned = copy.deepcopy(copies.circuit)
            conditioned.condition_on_point_in_place(products.points[index])
            conditioned.restrict_to_variables_in_place(copies.variables)
            conditioned.update_variables(copies.renaming)
            root.add_subcircuit(self.mount(conditioned.root)[conditioned.root.index])
        for variable, values in products.point_masses.items():
            root.add_subcircuit(leaf(make_dirac(variable, values[index]), self))
        if products.components:
            component = products.components[index]
            root.add_subcircuit(self.mount(component.root)[component.root.index])
        return root

    # %% reading the mixtures

    def unnormalized_mixtures(
        self, tolerance: float = 1e-6
    ) -> list[UnnormalizedMixture[Unit]]:
        return [
            UnnormalizedMixture(node, node.log_normalization_constant)
            for node in self.nodes()
            if abs(node.log_normalization_constant) >= tolerance
        ]

    def mixtures_over(
        self, variables: SortedSet[Variable]
    ) -> list[MixtureOverVariable[Unit]]:
        _ = self.support
        return [
            report
            for layer in self.layers
            for node in layer
            for report in self._mixtures_of_unit_over(node, variables)
        ]

    def _mixtures_of_unit_over(
        self, unit: Unit, variables: SortedSet[Variable]
    ) -> list[MixtureOverVariable[Unit]]:
        """
        :param unit: The unit to read.
        :param variables: The variables to read it through.
        :return: One report per variable the unit models, or nothing if it mixes fewer
            than two components.
        """
        components = unit.mixture_components()
        if len(components) < 2:
            return []
        return [
            MixtureOverVariable(
                unit,
                variable,
                [
                    component.result_of_current_query.marginal([variable])
                    for component in components
                ],
            )
            for variable in variables
            if variable in unit.variables
        ]

    # %% composition

    @classmethod
    def mixture_of(cls, components: list[Self], log_weights: Iterable[float]) -> Self:
        log_weights = list(log_weights)
        ProbabilisticCircuitRepresentation.validate_mixture(components, log_weights)
        result = cls()
        root = SumUnit(probabilistic_circuit=result)
        for component, log_weight in zip(components, log_weights):
            root.add_subcircuit(result._mounted_copy_of(component), log_weight)
        root.normalize()
        return result

    @classmethod
    def product_of(cls, factors: list[Self]) -> Self:
        ProbabilisticCircuitRepresentation.validate_product(factors)
        result = cls()
        root = ProductUnit(probabilistic_circuit=result)
        for factor in factors:
            root.add_subcircuit(result._mounted_copy_of(factor))
        return result

    def _mounted_copy_of(self, other: Self) -> Unit:
        """
        :param other: A circuit to copy into this one.
        :return: The root of the copy, a unit of this circuit.
        """
        copied = copy.deepcopy(other)
        return self.mount(copied.root)[copied.root.index]

    def marginal(self, variables: Iterable[Variable]) -> Self | None:
        result = self.restrict_to_variables(variables)
        if result is None:
            return None
        result.simplify()
        return result

    def restrict_to_variables(self, variables: Iterable[Variable]) -> Self | None:
        """
        Restrict a copy of the circuit to variables, without ``simplify()``'s same-type
        merge. See :meth:`restrict_to_variables_in_place`.

        :param variables: The variables to keep.
        :return: The restricted copy, or ``None`` if none of ``variables`` are modeled.
        """
        result = copy.deepcopy(self)
        return result.restrict_to_variables_in_place(variables)

    def sample(self, amount: int) -> npt.NDArray:
        # initialize all results
        for node in self.graph.nodes():
            node.result_of_current_query = []

        variable_to_index_map = self.variable_to_index_map

        # the root is responsible for every row of the output array
        self.root.result_of_current_query.append(np.arange(amount))

        # initialize the samples
        samples = np.full((amount, len(variable_to_index_map)), np.nan)

        # forward through the circuit to sample
        [
            node.sample(samples, variable_to_index_map)
            for layer in self.layers
            for node in layer
        ]

        return samples

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        variable_to_index_map = self.variable_to_index_map
        [
            node.moment(order, center, variable_to_index_map)
            for layer in reversed(self.layers)
            for node in layer
        ]
        return MomentType(
            {
                variable: moment
                for variable, moment in zip(
                    variable_to_index_map.keys(), self.root.result_of_current_query
                )
            }
        )

    def simplify(self) -> Self:
        """
        Simplify the circuit inplace.
        """
        [node.simplify() for layer in reversed(self.layers) for node in layer]
        return self

    @property
    def support(self) -> Event:
        [node.support() for layer in reversed(self.layers) for node in layer]
        return self.root.result_of_current_query

    def is_decomposable(self) -> bool:
        """
        Check if the whole circuit is decomposed.

        A circuit is decomposed if all its product units are decomposed.

        :return: if the whole circuit is decomposed
        """
        return all(
            [
                subcircuit.is_decomposable()
                for subcircuit in self.leaves
                if isinstance(subcircuit, ProductUnit)
            ]
        )

    def __eq__(self, other: Self):
        raise NotImplementedError

    def empty_copy(self) -> Self:
        """
        Create a copy of this circuit without any nodes.

        Only the parameters should be copied. This is used whenever a new circuit has to
        be created during inference.

        :return: A copy of this circuit without any subcircuits that is not in this
            units graph.
        """
        return self.__class__()

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        result = self.empty_copy()

        # remap nodes to new copies
        remapped_indices = {
            node.index: node.copy_without_graph(memo) for node in self.nodes()
        }

        # add copied nodes
        result.add_nodes_from(remapped_indices.values())

        # copy edges and edge data
        [
            result.graph.add_edge(
                remapped_indices[parent].index,
                remapped_indices[child].index,
                self.graph.get_edge_data(parent, child),
            )
            for parent, child in self.graph.edge_list()
        ]

        return result

    def to_json(self) -> dict[str, Any]:
        # get super result
        result = super().to_json()

        index_to_node_map = {node.index: to_json(node) for node in self.nodes()}
        edges = [
            (parent.index, child.index, data) for parent, child, data in self.edges()
        ]

        result["index_to_node_map"] = index_to_node_map
        result["edges"] = edges

        return result

    @classmethod
    def parameters_from_json(cls, data: dict[str, Any]) -> Self:
        return cls()

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        result = cls.parameters_from_json(data)
        hash_remap: dict[int, Unit] = dict()

        for index, node_data in data["index_to_node_map"].items():
            node = from_json(node_data)
            hash_remap[int(index)] = node
            result.add_node(node)

        [
            result.graph.add_edge(
                hash_remap[parent_index].index, hash_remap[child_index].index, data
            )
            for parent_index, child_index, data in data["edges"]
        ]

        return result

    def update_variables(self, new_variables: VariableMap):
        """
        Update the variables of this unit and its descendants.

        :param new_variables: The new variables to set.
        """
        self.root.update_variables(new_variables)

    def rename_variables_with_prefix(
        self, prefix: str, excluded_variables: Iterable[Variable] = ()
    ) -> None:
        """
        Rename each variable in this circuit to include ``prefix`` as a namespace.

        Produces names of the form ``"{prefix}.{variable.name}"``. Variables in
        ``excluded_variables``, and variables already inside that namespace, are left
        unchanged.

        :param prefix: String prefix to prepend to every variable name.
        :param excluded_variables: Variables that should keep their current names.
        """
        variable_renames = {
            variable: type(variable)(
                f"{prefix}.{variable.name}", domain=variable.domain
            )
            for variable in self.variables
            if variable not in excluded_variables
            and not variable.name.startswith(f"{prefix}.")
        }
        self.update_variables(variable_renames)

    def is_deterministic(self) -> bool:
        """
        :return: Whether, this circuit is deterministic or not.
        """
        # calculate the support
        support = self.support

        # check for determinism of every node
        return all(
            node.is_deterministic()
            for node in self.graph.nodes()
            if isinstance(node, SumUnit)
        )

    def normalize(self):
        """
        Normalize every sum node of this circuit in-place.
        """
        [node.normalize() for node in self.graph.nodes() if isinstance(node, SumUnit)]

    def add_edges_and_nodes_from_circuit(self, other: Self):
        """
        Add all edges and nodes from another circuit to this circuit.

        :param other: The other circuit to add.
        """
        self.add_nodes_from(other.graph.nodes())
        self.graph.add_edges_from(other.unweighted_edges)
        self.add_weighted_edges_from(other.log_weighted_edges, weight="log_weight")

    def add_weighted_edges_from(self, ebunch_to_add, weight="log_weight", **attr):
        return self.graph.add_weighted_edges_from(ebunch_to_add, weight=weight, **attr)

    def subgraph_of(self, node: Unit) -> Self:
        """
        Create a subgraph with a node as root.

        :param node: The root of the subgraph.
        :return: The subgraph.
        """
        result = self.__class__()
        result.mount(node)
        return result

    def fill_node_colors(self, node_colors: dict[Unit, str]):
        """
        Fill the node colors for the structure plot.

        :param node_colors: The node colors to fill.
        """
        # fill the colors for the nodes
        if node_colors is None:
            node_colors = dict()
        for node in self.graph.nodes():
            if node not in node_colors:
                node_colors[node] = "black"
        return node_colors

    def breadth_first_search_layout(
        self, scale: float = 1.0, align: PlotAlignment = PlotAlignment.VERTICAL
    ) -> dict[int, npt.NDArray]:
        """
        Generate a bfs layout for this circuit.

        :return: A dict mapping the node indices to 2d coordinates.
        """
        layers = self.layers

        position = None
        nodes = []
        width = len(layers)
        for index, layer in enumerate(layers):
            height = len(layer)
            xs = np.repeat(index, height)
            ys = np.arange(0, height, dtype=float)
            offset = ((width - 1) / 2, (height - 1) / 2)
            layer_position = np.column_stack([xs, ys]) - offset
            if position is None:
                position = layer_position
            else:
                position = np.concatenate([position, layer_position])
            nodes.extend(layer)

        # Find max length over all dimensions
        position -= position.mean(axis=0)
        lim = np.abs(position).max()  # max coordinate for all axes
        # rescale to (-scale, scale) in all directions, preserves aspect
        if lim > 0:
            position *= scale / lim

        if align == PlotAlignment.HORIZONTAL:
            position = position[:, ::-1]  # swap x and y coords

        position = dict(zip([node.index for node in nodes], position))
        return position

    def plot_structure(
        self,
        node_colors: dict[Unit, str] | None = None,
        variable_name_offset=0.2,
        plot_inference=False,
        inference_representation: Callable = lambda node: str(
            node.result_of_current_query
        ),
        inference_result_offset: float = -0.25,
    ):
        """
        Plot the structure of the circuit using matplotlib.

        :param node_colors: Optionally specified colors of the node. If nodes are not
            specified in the dictionary, they will be black.
        :param node_size: The size of the nodes
        :param variable_name_offset: The offset to the right of the variable names.
        :param plot_inference: If the results of the inference should be plotted.
        :param inference_representation: The representation of the inference results as
            a function from node to string.
        :param inference_result_offset: The vertical offset of the inference results.
        """
        # fill the colors for the nodes
        node_colors = self.fill_node_colors(node_colors)
        scale = 1.0
        layers = self.layers

        # get the positions of the nodes
        positions = self.breadth_first_search_layout(
            scale=scale, align=PlotAlignment.VERTICAL
        )
        position_for_variable_name = {
            node: (x + variable_name_offset, y) for node, (x, y) in positions.items()
        }

        def node_labels(node: Unit) -> str:
            if isinstance(node, SumUnit):
                return "+"
            elif isinstance(node, ProductUnit):
                return "×"
            elif isinstance(node, LeafUnit):
                return str(node.distribution)
            else:
                raise NotImplementedError

        def edge_labels(data) -> str:
            if data is None:
                return ""
            else:
                return str(np.round(data, decimals=2))

        rustworkx.visualization.mpl_draw(
            self.graph,
            pos=positions,
            labels=node_labels,
            with_labels=True,
            edge_labels=edge_labels,
        )

    def nodes_weights(self) -> dict:
        """
        :return: dict with keys as nodes and values as list of all the log_weights for the node.
        """
        node_weights = {hash(self.root): [1]}
        seen_nodes = set()
        seen_nodes.add(hash(self.root))

        to_visit_nodes = queue.Queue()

        to_visit_nodes.put(self.root)
        while not to_visit_nodes.empty():
            node = to_visit_nodes.get()
            succ_iter = self.graph.successors(node)
            for succ in succ_iter:
                if self.graph.has_edge(node, succ):
                    weight = self.graph.get_edge_data(node, succ).get("weight", 1)
                    node_weights[hash(succ)] = [
                        old * weight for old in node_weights[hash(node)]
                    ] + node_weights.get(hash(succ), [])
                    if hash(succ) not in seen_nodes:
                        seen_nodes.add(hash(succ))
                        to_visit_nodes.put(succ)
        return node_weights

    def replace_discrete_distribution_with_deterministic_sum(self):
        """
        Splits the distribution into sum unit with all the discrete possibilities as
        leaf.
        """
        old_leafs = self.leaves
        for leaf in old_leafs:

            if isinstance(leaf, UnivariateDiscreteLeaf):
                leaf: UnivariateDiscreteLeaf
                sum_leaf = leaf.as_deterministic_sum()
                old_predecessors = list(self.graph.predecessors(leaf))
                for predecessor in old_predecessors:
                    weight = self.graph.get_edge_data(predecessor, leaf).get(
                        "log_weight", -1
                    )
                    if weight == -1:
                        predecessor.add_subcircuit(sum_leaf)
                    else:
                        predecessor.add_subcircuit(sum_leaf, log_weight=weight)
                    self.graph.remove_edge(predecessor, leaf)
                self.graph.remove_node(leaf)
        self._invalidate_topology_cache()

    def apply_translation(self, translation: dict[Variable, float]):
        for leaf in self.leaves:
            if any(v.is_numeric for v in leaf.variables):
                leaf.distribution.apply_translation(translation)

    def apply_scaling(self, scale: dict[Variable, float]):
        for leaf in self.leaves:
            if any(v.is_numeric for v in leaf.variables):
                leaf.distribution.apply_scaling(scale)

    def mount(self, other: Unit) -> dict[int, Unit]:
        """
        Mount another unit including its descendants.

        There will be no edge from `self` to `other`. This will also remove the nodes in
        other and their descendants from their circuit.

        :param other: The other unit to mount.
        :returns: A mapping from the indices of the nodes in `other` to the nodes in
            `self` that were added.
        """
        if other.probabilistic_circuit is not None:
            descendants = other.probabilistic_circuit.descendants(other)
            descendants = descendants.union([other])
            subgraph = other.probabilistic_circuit.graph.subgraph(
                [u.index for u in descendants]
            )
            result = self.add_from_subgraph(subgraph)
            return result
        else:
            raise ValueError(
                "Trying to mount a unit that doesn't belong to any probabilistic circuit."
            )

    def __repr__(self):
        return f"{self.__class__.__name__} with {len(self.nodes())} nodes and {len(self.edges())} edges"


class ShallowProbabilisticCircuit(ProbabilisticCircuit):
    """
    Class for PC in shallow form, sum unit as root followed by product units which only
    have leafs as children.
    """

    @classmethod
    def from_probabilistic_circuit(
        cls, probabilistic_circuit: ProbabilisticCircuit
    ) -> Self:
        """
        Create the shallow circuit with the same distribution as a circuit, leaving that
        circuit unchanged.

        :param probabilistic_circuit: The circuit to convert.
        :return: The shallow circuit.
        """
        result = cls()
        root = SumUnit(probabilistic_circuit=result)
        copies_of_leaves: dict[int, LeafUnit] = {}
        for weighted_product in probabilistic_circuit.root.weighted_leaf_products():
            product = ProductUnit(probabilistic_circuit=result)
            for leaf_unit in weighted_product.leaves:
                if leaf_unit.index not in copies_of_leaves:
                    copies_of_leaves[leaf_unit.index] = leaf_unit.copy_without_graph()
                product.add_subcircuit(copies_of_leaves[leaf_unit.index])
            root.add_subcircuit(product, weighted_product.log_weight)
        return result

    def cells_of_constant_density(self, other: Self) -> list[Event]:
        """
        Partition the union of the supports of this and another shallow circuit into
        disjoint cells that each lie either entirely inside or entirely outside the
        support of every product unit of both circuits.

        :param other: The other shallow circuit.
        :return: The cells of the partition.
        """
        cells: list[Event] = []
        for circuit in [self, other]:
            for product in circuit.root.subcircuits:
                support = circuit.subgraph_of(product).support
                uncovered = support
                refined_cells = []
                for cell in cells:
                    refined_cells.extend(
                        part
                        for part in [cell & support, cell - support]
                        if not part.is_empty()
                    )
                    uncovered = uncovered - cell
                if not uncovered.is_empty():
                    refined_cells.append(uncovered)
                cells = refined_cells
        return cells

    def event_of_higher_density(self, other: Self, tolerance: float = 10e-8) -> Event:
        """
        Construct the event where this shallow circuit has a higher density than another
        one.

        .. note::

            The densities are compared at one point per cell of
            :meth:`cells_of_constant_density`, which is exact when every leaf is
            uniform.

        :param other: The other shallow circuit.
        :param tolerance: The likelihood difference below which two likelihoods count
            as equal.
        :return: The event where this circuit is more likely.
        """
        result = Event.from_simple_sets()
        for cell in self.cells_of_constant_density(other):
            point = np.array(
                [
                    [
                        assignment.simple_sets[0].center()
                        for assignment in cell.simple_sets[0].values()
                    ]
                ]
            )
            if self.likelihood(point)[0] - other.likelihood(point)[0] > tolerance:
                result |= cell
        return result

    def total_variation_distance(self, other: Self, tolerance: float = 10e-8) -> float:
        """
        Calculate the total variation distance :math:`\\frac{1}{2} \\int |p(x) - q(x)| dx`
        between this circuit and another shallow circuit, which lies in :math:`[0, 1]`.

        .. note::

            Call :meth:`replace_discrete_distribution_with_deterministic_sum` on a
            circuit before making it shallow.

        :param other: The other shallow circuit.
        :param tolerance: The likelihood difference below which two likelihoods count
            as equal when constructing the event where this circuit is more likely.
        :return: The total variation distance between the two circuits.
        """
        event_of_higher_density = self.event_of_higher_density(other, tolerance)
        return self.probability(event_of_higher_density) - other.probability(
            event_of_higher_density
        )


@dataclass
class UnivariateLeaf(LeafUnit):

    @property
    def variable(self) -> Variable:
        return self.distribution.variables[0]


@dataclass
class UnivariateContinuousLeaf(UnivariateLeaf):
    distribution: ContinuousDistribution | None

    __hash__ = Unit.__hash__

    def log_truncated_of_simple_event_in_place(
        self, event: SimpleEvent, singleton_allowed: bool = False
    ):
        return self.univariate_log_truncated_of_simple_event_in_place(
            event[self.variable], singleton_allowed
        )

    def univariate_log_truncated_of_simple_event_in_place(
        self, event: Interval, singleton_allowed: bool = False
    ):
        """
        Condition this distribution on a simple event in-place but use sum units to
        create conditions on composite intervals.

        :param event: The simple event to condition on.
        """
        # if it is a simple truncation
        if len(event.simple_sets) == 1:
            self.adopt_truncation(
                self.distribution.log_conditional_from_simple_interval(
                    event.simple_sets[0], singleton_allowed
                )
            )
            return self

        return self.replace_by_mixture(
            self.distribution.log_conditional_from_simple_interval(
                simple_interval, singleton_allowed
            )
            for simple_interval in event.simple_sets
        )


@dataclass
class UnivariateDiscreteLeaf(UnivariateLeaf):

    distribution: DiscreteDistribution | None
    __hash__ = Unit.__hash__

    def as_deterministic_sum(self) -> SumUnit:
        """
        Convert this distribution to a deterministic sum unit that encodes the same
        distribution in-place. The result has as many children as the probability
        dictionary of this distribution. Each child encodes the value of the variable.

        :return: The deterministic sum unit that encodes the same distribution.
        """
        result = SumUnit(probabilistic_circuit=self.probabilistic_circuit)

        for element, probability in self.distribution.probabilities.items():
            result.add_subcircuit(
                leaf(
                    self.distribution.__class__(
                        variable=self.variable,
                        probabilities=MissingDict(float, {element: 1.0}),
                    ),
                    self.probabilistic_circuit,
                ),
                np.log(probability),
            )
        self.connect_incoming_edges_to(result)
        self.probabilistic_circuit.remove_node(self)
        return result

    @classmethod
    def from_mixture(cls, mixture: ProbabilisticCircuit):
        """
        Create a discrete distribution from a univariate mixture.

        :param mixture: The mixture to create the distribution from.
        :return: The discrete distribution.
        """
        assert (
            len(mixture.variables) == 1
        ), "Can only convert univariate sum units to discrete distributions."
        variable = mixture.variables[0]
        probabilities = MissingDict(float)

        for element in mixture.support.simple_sets[0][variable].simple_sets:
            probability = mixture.probability_of_simple_event(
                SimpleEvent.from_data({variable: element})
            )
            if isinstance(element, SimpleInterval):
                element = element.lower
            probabilities[hash(element)] = probability

        distribution_class = (
            IntegerDistribution
            if isinstance(variable, Integer)
            else SymbolicDistribution
        )
        distribution = distribution_class(
            variable=variable, probabilities=probabilities
        )
        return cls(distribution)


@dataclass(eq=False)
class MultivariateLeaf(LeafUnit):
    """
    A leaf whose distribution is over several variables at once.

    Whatever the distribution cannot represent itself, the circuit represents around
    it: conditioning on all of its variables leaves a product of Dirac leaves, and an
    event of several boxes becomes a mixture of one truncation per box.
    """

    def log_conditional_in_place(self, point: dict[Variable, Any]):
        own_point = self.filter_variable_map_by_self(point)
        if not own_point:
            self.result_of_current_query = 0.0
            return
        if len(own_point) == len(self.distribution.variables):
            self.replace_by_dirac_product(own_point)
            return
        self.adopt_conditioning(self.distribution.log_conditional(own_point))

    def moment(self, order, center, variable_to_index_map):
        result = np.zeros(len(variable_to_index_map))
        requested = [
            variable for variable in self.distribution.variables if variable in order
        ]
        if requested:
            moment = self.distribution.moment(
                {variable: order[variable] for variable in requested},
                {variable: center[variable] for variable in requested},
            )
            for variable in requested:
                result[variable_to_index_map[variable]] = moment[variable]
        self.result_of_current_query = result

    def replace_by_dirac_product(self, point: dict[Variable, Any]) -> ProductUnit:
        """
        Replace this leaf by the product of one Dirac leaf per variable.

        :param point: The value of every variable of this leaf.
        :return: The product unit that replaced this leaf, carrying the log-density of
            the point under the distribution.
        """
        result = ProductUnit(probabilistic_circuit=self.probabilistic_circuit)
        for variable in self.distribution.variables:
            result.add_subcircuit(
                leaf(make_dirac(variable, point[variable]), self.probabilistic_circuit)
            )
        result.result_of_current_query = self.distribution.log_likelihood(
            np.array([[point[variable] for variable in self.distribution.variables]])
        )[0]

        self.connect_incoming_edges_to(result)
        self.probabilistic_circuit.remove_node(self)
        return result

    def log_truncated_of_simple_event_in_place(
        self, event: SimpleEvent, singleton_allowed: bool = False
    ):
        """
        Truncate the distribution to every box the event makes of this leaf's variables,
        and mix the truncations when there is more than one.

        :param event: The simple event to truncate to.
        :param singleton_allowed: Whether singletons are allowed in the event.
        """
        variables = self.distribution.variables
        boxes = [
            SimpleEvent.from_data(
                {
                    variable: simple_interval.as_composite_set()
                    for variable, simple_interval in zip(variables, simple_intervals)
                }
            ).as_composite_set()
            for simple_intervals in itertools.product(
                *(event[variable].simple_sets for variable in variables)
            )
        ]
        if len(boxes) == 1:
            self.adopt_truncation(
                self.distribution.log_truncated(boxes[0], singleton_allowed)
            )
            return self

        return self.replace_by_mixture(
            self.distribution.log_truncated(box, singleton_allowed) for box in boxes
        )


def leaf(
    distribution: ProbabilisticModel,
    probabilistic_circuit: ProbabilisticCircuit | None = None,
) -> LeafUnit:
    """
    Factory that creates the correct leaf from a distribution.

    :return: The leaf.
    """
    if not isinstance(distribution, UnivariateDistribution):
        return MultivariateLeaf(
            distribution, probabilistic_circuit=probabilistic_circuit
        )
    if isinstance(distribution.variable, Continuous):
        return UnivariateContinuousLeaf(
            distribution, probabilistic_circuit=probabilistic_circuit
        )
    else:
        return UnivariateDiscreteLeaf(
            distribution, probabilistic_circuit=probabilistic_circuit
        )
