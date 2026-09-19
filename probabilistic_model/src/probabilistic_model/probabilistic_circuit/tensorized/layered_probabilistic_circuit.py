from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from krrood.adapters.json_serializer import DataclassJSONSerializer
from random_events.product_algebra import Event, SimpleEvent, VariableMap
from random_events.variable import Variable
from sortedcontainers import SortedSet
from typing_extensions import Any, Dict, Iterable, List, Optional, Self, Tuple

from probabilistic_model.distributions.helper import make_dirac
from probabilistic_model.exceptions import IntractableError, UnorderedVariablesError
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer import (
    ForwardSampleAssignment,
    Layer,
    ProductLayer,
    QueryCache,
    SumLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer import (
    layer_of_distributions,
)
from probabilistic_model.probabilistic_circuit.tensorized.rustworkx_conversion import (
    circuit_of_root_layer,
    root_layer_of_circuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.utils import SparseArray
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit as RustworkxProbabilisticCircuit,
)
from probabilistic_model.probabilistic_model import (
    CenterType,
    MomentType,
    OrderType,
    ProbabilisticModel,
)
from probabilistic_model.utils import logsumexp


@dataclass(eq=False)
class LayeredProbabilisticCircuit(ProbabilisticModel, DataclassJSONSerializer):
    """
    A probabilistic circuit whose units are grouped into layers of numpy arrays.

    The circuit is a rooted directed acyclic graph of :class:`Layer` objects. Every
    layer holds the parameters of all of its nodes in contiguous arrays, so that a query
    is evaluated for all nodes of a layer at once instead of node by node.

    The root layer has exactly one node, which is the output of the circuit.
    """

    variables: SortedSet
    """
    The variables of the circuit, ordered.

    The layers refer to them by their index here, so the order is part of the circuit
    and a container that does not fix it is rejected.
    """

    root: Layer
    """
    The root layer of the circuit.
    """

    def __post_init__(self):
        if not isinstance(self.variables, SortedSet):
            raise UnorderedVariablesError(type(self.variables))

    @property
    def variable_to_index_map(self) -> Dict[Variable, int]:
        """
        :return: A map from every variable of the circuit to its column index.
        """
        return {variable: index for index, variable in enumerate(self.variables)}

    @property
    def number_of_nodes(self) -> int:
        """
        :return: The number of nodes of the circuit.
        """
        return sum(layer.number_of_nodes for layer in self.root.all_layers())

    @property
    def number_of_parameters(self) -> int:
        """
        :return: The number of parameters of the circuit.
        """
        return self.root.number_of_parameters

    @property
    def layers(self) -> List[Layer]:
        """
        :return: Every layer of the circuit, parents before children.
        """
        return self.root.all_layers()

    def validate(self):
        """
        Check that the parameter arrays of every layer have consistent shapes.
        """
        self.root.validate()

    def __repr__(self):
        return (
            f"{self.__class__.__name__} over {list(self.variables)} "
            f"with {len(self.layers)} layers and {self.number_of_nodes} nodes"
        )

    # %% queries

    def log_likelihood(self, events: npt.NDArray) -> npt.NDArray:
        return self.root.log_likelihood_of_nodes(np.asarray(events))[:, 0]

    def cumulative_distribution_function(self, events: npt.NDArray) -> npt.NDArray:
        return self.root.cumulative_distribution_of_nodes(np.asarray(events))[:, 0]

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        return float(
            self.root.probability_of_simple_event_of_nodes(event, self.variables)[0]
        )

    @property
    def support(self) -> Event:
        return self.root.support_of_nodes(self.variables)[0]

    def log_mode(self, check_determinism: bool = True) -> Tuple[Event, float]:
        if check_determinism and not self.is_deterministic():
            raise IntractableError(self)
        modes, values = self.root.log_mode_of_nodes(self.variables)
        return modes[0], float(values[0])

    def sample(self, amount: int) -> npt.NDArray:
        order = self.root.topological_layer_order()
        assignment = ForwardSampleAssignment.for_layers(order)

        # the root is responsible for every row of the output array
        assignment.assign(self.root, 0, np.arange(amount))

        samples = np.full((amount, len(self.variables)), np.nan)
        for layer in order:
            layer.sample_forward(assignment, samples, self.variables)
        return samples

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        number_of_variables = len(self.variables)
        order_array = np.zeros(number_of_variables, dtype=np.int64)
        center_array = np.zeros(number_of_variables)
        requested = np.zeros(number_of_variables, dtype=bool)

        for variable, value in order.items():
            index = self.variables.index(variable)
            order_array[index] = value
            requested[index] = True

        for variable, value in center.items():
            center_array[self.variables.index(variable)] = value

        result = self.root.moment_of_nodes(
            order_array, center_array, requested, self.variables
        )
        return MomentType(
            {
                variable: result[0, index]
                for index, variable in enumerate(self.variables)
            }
        )

    def is_deterministic(self) -> bool:
        """
        :return: Whether every sum node of this circuit has children with pairwise
            disjoint supports.
        """
        return self.root.is_deterministic(self.variables)

    def is_decomposable(self) -> bool:
        """
        :return: Whether every product node of this circuit factorizes over disjoint
            scopes.
        """
        return self.root.is_decomposable()

    # %% structural

    def log_truncated(
        self, event: Event, singleton_allowed: bool = False
    ) -> Tuple[Optional[Self], float]:
        result = self.__deepcopy__()
        return result.log_truncated_in_place(event, singleton_allowed)

    def log_truncated_in_place(
        self, event: Event, singleton_allowed: bool = False
    ) -> Tuple[Optional[Self], float]:
        """
        Truncate this circuit to an event in place.

        A composite event is truncated to each of its disjoint simple sets, and the
        results become the children of a new root sum layer weighted by the probability
        of the set they were truncated to.

        :param event: The event to truncate to.
        :param singleton_allowed: Whether singletons are allowed in the event.
        :return: This circuit and the log-probability of the event, or ``(None, -inf)``.
        """
        if event.is_empty():
            return None, -np.inf

        event.fill_missing_variables(set(self.variables))

        if len(event.simple_sets) == 1:
            return self.log_truncated_of_simple_event_in_place(
                event.simple_sets[0], singleton_allowed
            )

        batched = self.truncated_root_of_simple_events(
            list(event.simple_sets), singleton_allowed
        )
        if batched is not None:
            root, total_log_probability = batched
            if root is None:
                return None, -np.inf
            self.root = root
            return self, total_log_probability

        # no copy per simple set: a structural pass never writes into the layers it was
        # given, it builds new ones, so all the truncations can read the same circuit
        truncated = []
        for simple_event in event.simple_sets:
            root, log_probability = self.truncated_root_of_simple_event(
                simple_event, singleton_allowed
            )
            if root is not None and log_probability > -np.inf:
                truncated.append((root, log_probability))

        if not truncated:
            return None, -np.inf

        # the simple sets of an event are disjoint, so P(E) = sum_k P(E_k)
        total_log_probability = float(
            logsumexp(np.array([log_probability for _, log_probability in truncated]))
        )

        log_weights = [
            SparseArray.from_coordinates(
                np.array([0]),
                np.array([0]),
                np.array([log_probability]),
                (1, root.number_of_nodes),
            )
            for root, log_probability in truncated
        ]
        self.root = SumLayer([root for root, _ in truncated], log_weights)
        self.root.normalize()
        return self, total_log_probability

    def truncated_root_of_simple_event(
        self, event: SimpleEvent, singleton_allowed: bool = False
    ) -> Tuple[Optional[Layer], float]:
        """
        Build the root of this circuit truncated to a simple event.

        This leaves the circuit itself untouched: the pass creates new layers rather than
        writing into the existing ones, which is what lets a truncation to a composite
        event reuse one circuit for all of its simple sets instead of copying it per set.

        :param event: The simple event to truncate to.
        :param singleton_allowed: Whether singletons are allowed in the event.
        :return: The new root layer and the log-probability of the event, or
            ``(None, -inf)`` if the event is impossible.
        """
        log_probabilities: Dict[int, npt.NDArray] = {}
        new_root, node_log_probabilities = self.root.log_truncated_of_simple_event(
            event,
            self.variables,
            singleton_allowed,
            cache=QueryCache(),
            log_probabilities=log_probabilities,
        )

        log_probability = float(node_log_probabilities[0])
        if log_probability == -np.inf:
            return None, -np.inf

        pruned = new_root.prune(log_probabilities)
        if pruned is None:
            return None, -np.inf

        root = pruned.simplify()
        root.normalize()
        return root, log_probability

    def truncated_root_of_simple_events(
        self, events: List[SimpleEvent], singleton_allowed: bool = False
    ) -> Optional[Tuple[Optional[Layer], float]]:
        """
        Build the root of this circuit truncated to several simple events in one pass.

        Every layer is replicated once per event, so the result has the same number of
        *layers* as this circuit and blocks that are as many times taller as there are
        events. Truncating once per event and mixing the results instead would produce
        one set of layers per event, which is what makes the following queries slow:
        with a hundred simple sets, the same circuit ends up spread over hundreds of
        layers of a few nodes each.

        :param events: The simple events to truncate to.
        :param singleton_allowed: Whether singletons are allowed in the events.
        :return: The new root and the log-probability of the union of the events, or
            ``None`` if a layer of this circuit cannot be truncated this way and the
            caller has to fall back to truncating once per event.
        """
        log_probabilities: Dict[int, npt.NDArray] = {}
        batched = self.root.log_truncated_of_simple_events(
            events,
            self.variables,
            singleton_allowed,
            cache=QueryCache(),
            log_probabilities=log_probabilities,
        )
        if batched is None:
            return None
        replicated, node_log_probabilities = batched

        # the simple sets of an event are disjoint, so P(E) = sum_k P(E_k)
        total_log_probability = float(logsumexp(node_log_probabilities))
        if total_log_probability == -np.inf:
            return None, -np.inf

        # mix the copy of the root that belongs to each event by the probability of that
        # event, which turns the replicated root into the single root of the result
        mixture = SumLayer(
            [replicated],
            [
                SparseArray.from_coordinates(
                    np.zeros(len(node_log_probabilities), dtype=np.int64),
                    np.arange(len(node_log_probabilities)),
                    node_log_probabilities,
                    (1, len(node_log_probabilities)),
                )
            ],
        )
        log_probabilities[id(mixture)] = np.array([total_log_probability])

        pruned = mixture.prune(log_probabilities)
        if pruned is None:
            return None, -np.inf

        root = pruned.simplify()
        root.normalize()
        return root, total_log_probability

    def log_truncated_of_simple_event_in_place(
        self, event: SimpleEvent, singleton_allowed: bool = False
    ) -> Tuple[Optional[Self], float]:
        """
        Truncate this circuit to a simple event in place.

        :param event: The simple event to truncate to.
        :param singleton_allowed: Whether singletons are allowed in the event.
        :return: This circuit and the log-probability of the event, or ``(None, -inf)``.
        """
        root, log_probability = self.truncated_root_of_simple_event(
            event, singleton_allowed
        )
        if root is None:
            return None, -np.inf

        self.root = root
        return self, log_probability

    def log_conditional(
        self, point: Dict[Variable, Any]
    ) -> Tuple[Optional[Self], float]:
        result = self.__deepcopy__()
        return result.log_conditional_in_place(point)

    def log_conditional_in_place(
        self, point: Dict[Variable, Any]
    ) -> Tuple[Optional[Self], float]:
        """
        Condition this circuit on a partial point in place.

        The variables of the point are marginalized out of the conditioned circuit and
        reattached as Dirac layers under a new product root.

        :param point: The partial point.
        :return: This circuit and the log-density at the point, or ``(None, -inf)``.
        """
        log_probabilities: Dict[int, npt.NDArray] = {}
        new_root, node_log_probabilities = self.root.log_conditional_of_point(
            point,
            self.variables,
            cache=QueryCache(),
            log_probabilities=log_probabilities,
        )

        log_probability = float(node_log_probabilities[0])
        if log_probability == -np.inf:
            return None, -np.inf

        pruned = new_root.prune(log_probabilities)
        if pruned is None:
            return None, -np.inf

        self.root = pruned

        original_variables = self.variables
        remaining = [variable for variable in self.variables if variable not in point]

        children: List[Layer] = []
        if remaining:
            if self.marginal_in_place(remaining) is None:
                return None, -np.inf
            self.restore_variables(original_variables)
            children.append(self.root)

        for variable, value in point.items():
            children.append(
                layer_of_distributions(
                    original_variables.index(variable), [make_dirac(variable, value)]
                )
            )

        edges = SparseArray.from_coordinates(
            np.arange(len(children)),
            np.zeros(len(children), dtype=np.int64),
            np.zeros(len(children), dtype=np.int64),
            (len(children), 1),
        )
        self.root = ProductLayer(children, edges).simplify()
        self.root.normalize()
        return self, log_probability

    def restore_variables(self, variables: SortedSet):
        """
        Re-embed the circuit into a larger set of variables.

        :param variables: The variables to embed into. Every variable of this circuit
            must be one of them.
        """
        remap = np.array(
            [variables.index(variable) for variable in self.variables], dtype=np.int64
        )
        self.root.remap_variables(remap, QueryCache())
        self.variables = variables

    def marginal(self, variables: Iterable[Variable]) -> Optional[Self]:
        result = self.__deepcopy__()
        if result.marginal_in_place(variables) is None:
            return None
        return result

    def marginal_in_place(self, variables: Iterable[Variable]) -> Optional[Self]:
        """
        Restrict this circuit to a subset of its variables in place.

        :param variables: The variables to keep.
        :return: This circuit, or ``None`` if it models none of the variables.
        """
        requested = set(variables)
        kept_variables = SortedSet(
            variable for variable in self.variables if variable in requested
        )
        if not kept_variables:
            return None

        kept = np.array(
            [variable in kept_variables for variable in self.variables], dtype=bool
        )
        new_root = self.root.marginal(kept, QueryCache())
        if new_root is None:
            return None

        remap = np.full(len(self.variables), -1, dtype=np.int64)
        for new_index, variable in enumerate(kept_variables):
            remap[self.variables.index(variable)] = new_index
        new_root.remap_variables(remap, QueryCache())

        self.root = new_root.simplify()
        self.variables = kept_variables
        return self

    def simplify(self) -> Self:
        """
        Remove the layers that have no effect on the distribution, in place.

        :return: This circuit.
        """
        self.root = self.root.simplify()
        return self

    def normalize(self) -> Self:
        """
        Normalize the weights of every sum layer in place.

        :return: This circuit.
        """
        self.root.normalize()
        return self

    def update_variables(self, new_variables: VariableMap):
        """
        Replace variables of this circuit by other ones.

        :param new_variables: A map from the variables to replace to their replacement.
        """
        replaced = SortedSet(
            new_variables.get(variable, variable) for variable in self.variables
        )
        remap = np.array(
            [
                replaced.index(new_variables.get(variable, variable))
                for variable in self.variables
            ],
            dtype=np.int64,
        )
        self.root.remap_variables(remap, QueryCache())
        self.variables = replaced

    def rename_variables_with_prefix(
        self, prefix: str, excluded_variables: Iterable[Variable] = ()
    ) -> None:
        """
        Prefix the name of every variable of this circuit with a namespace.

        :param prefix: The prefix to prepend.
        :param excluded_variables: The variables to leave unchanged.
        """
        self.update_variables(
            VariableMap(
                {
                    variable: type(variable)(
                        f"{prefix}.{variable.name}", domain=variable.domain
                    )
                    for variable in self.variables
                    if variable not in excluded_variables
                }
            )
        )

    def apply_translation(self, translation: Dict[Variable, float]):
        values = np.zeros(len(self.variables))
        for variable, value in translation.items():
            values[self.variables.index(variable)] = value
        self.root.apply_translation(values)

    def apply_scaling(self, scaling: Dict[Variable, float]):
        values = np.ones(len(self.variables))
        for variable, value in scaling.items():
            values[self.variables.index(variable)] = value
        self.root.apply_scaling(values)

    # %% conversion

    @classmethod
    def from_rustworkx(
        cls, circuit: RustworkxProbabilisticCircuit, progress_bar: bool = False
    ) -> Self:
        """
        Convert a circuit of the ``rx`` package into a layered circuit.

        The result describes the same distribution.

        :param circuit: The circuit to convert.
        :param progress_bar: Whether to show a progress bar.
        :return: The layered circuit.
        """
        return cls(
            SortedSet(circuit.variables), root_layer_of_circuit(circuit, progress_bar)
        )

    def to_rustworkx(self, progress_bar: bool = False) -> RustworkxProbabilisticCircuit:
        """
        Convert this circuit into a circuit of the ``rx`` package.

        :param progress_bar: Whether to show a progress bar.
        :return: The converted circuit.
        """
        return circuit_of_root_layer(self.root, self.variables, progress_bar)

    def __deepcopy__(self, memo=None) -> Self:
        return self.__class__(SortedSet(self.variables), self.root.__deepcopy__({}))

    def __copy__(self) -> Self:
        return self.__deepcopy__()
