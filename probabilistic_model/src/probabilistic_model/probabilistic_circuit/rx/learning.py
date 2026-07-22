import copy
import numpy as np

from dataclasses import dataclass
from typing import Dict

from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    SumUnit,
    Unit,
)


@dataclass(frozen=True)
class Edge:
    parent: Unit
    child: Unit


def average_log_likelihood(
    circuit: ProbabilisticCircuit,
    data: np.ndarray,
) -> float:
    """
    Evaluate the average log likelihood of a probabilistic circuit.

    :param circuit: Probabilistic circuit to evaluate.
    :type circuit: ProbabilisticCircuit
    :param data: Dataset used for evaluation.
    :type data: np.ndarray
    :return: Average log likelihood over the provided samples.
    :rtype: float
    """

    log_likelihood = circuit.log_likelihood(data)

    return float(np.mean(log_likelihood))


def calculate_edge_flows(
    circuit: ProbabilisticCircuit,
    data: np.ndarray,
) -> Dict[Edge, float]:
    """
    Calculate the average probability flow through every weighted edge.

    Edge flows measure the contribution of every SumUnit connection
    according to the current dataset.

    :param circuit: Probabilistic circuit where flows are computed.
    :type circuit: ProbabilisticCircuit
    :param data: Dataset used for estimating flows.
    :type data: np.ndarray
    :return: Mapping from edges to their average probability flow.
    :rtype: Dict[Edge, float]
    """

    circuit.log_likelihood(data)

    flows: Dict[Edge, float] = {}

    for parent, child, log_weight in circuit.edges():

        if not isinstance(parent, SumUnit):
            continue

        parent_likelihood = parent.result_of_current_query
        child_likelihood = child.result_of_current_query

        flow = np.exp(child_likelihood + log_weight - parent_likelihood)

        flows[Edge(parent, child)] = float(np.mean(flow))

    return flows


def prune_edges(
    circuit: ProbabilisticCircuit,
    data: np.ndarray,
    prune_fraction: float = 0.1,
) -> ProbabilisticCircuit:
    """
    Remove low-contribution edges from SumUnits.

    :param circuit: Circuit to prune.
    :type circuit: ProbabilisticCircuit
    :param data: Dataset used to compute edge importance.
    :type data: np.ndarray
    :param prune_fraction: Fraction of removable edges removed.
    :type prune_fraction: float
    :return: Pruned circuit.
    :rtype: ProbabilisticCircuit
    """

    flows = calculate_edge_flows(
        circuit,
        data,
    )

    for node in list(circuit.graph.nodes()):

        if not isinstance(node, SumUnit):
            continue

        outgoing = list(circuit.graph.out_edges(node.index))

        if len(outgoing) <= 1:
            continue

        ranked_edges = []

        for _, child_index, _ in outgoing:

            child = circuit.graph[child_index]

            ranked_edges.append(
                (
                    child,
                    flows.get(
                        Edge(node, child),
                        0.0,
                    ),
                )
            )

        ranked_edges.sort(key=lambda item: item[1])

        number_remove = int(len(ranked_edges) * prune_fraction)

        number_remove = min(
            number_remove,
            len(ranked_edges) - 1,
        )

        for child, _ in ranked_edges[:number_remove]:

            circuit.graph.remove_edge(
                node.index,
                child.index,
            )

        node.normalize()

    circuit._invalidate_topology_cache()

    return circuit


def select_sum_units_to_grow(
    circuit: ProbabilisticCircuit,
    data: np.ndarray | None,
    fraction: float,
) -> list[SumUnit]:
    """
    Select SumUnits that should be duplicated during growing.

    :param circuit: Circuit containing candidate SumUnits.
    :type circuit: ProbabilisticCircuit
    :param data: Dataset used for importance estimation.
    :type data: np.ndarray | None
    :param fraction: Fraction of nodes selected.
    :type fraction: float
    :return: Selected SumUnits.
    :rtype: list[SumUnit]
    """

    sum_nodes = [node for node in circuit.graph.nodes() if isinstance(node, SumUnit)]

    if not sum_nodes:
        return []

    number_to_duplicate = max(
        1,
        int(len(sum_nodes) * fraction),
    )

    if data is None:

        return list(
            np.random.choice(
                sum_nodes,
                size=number_to_duplicate,
                replace=False,
            )
        )

    flows = calculate_edge_flows(
        circuit,
        data,
    )

    node_scores = {}

    for node in sum_nodes:

        score = 0.0

        for child in node.subcircuits:

            score += flows.get(
                Edge(node, child),
                0.0,
            )

        node_scores[node] = score

    return sorted(
        sum_nodes,
        key=lambda node: node_scores[node],
        reverse=True,
    )[:number_to_duplicate]


def duplicate_sum_unit(
    circuit: ProbabilisticCircuit,
    node: SumUnit,
    noise_scale: float,
) -> None:
    """
    Duplicate a SumUnit and attach it to the circuit.

    :param circuit: Circuit where the duplicated node is added.
    :type circuit: ProbabilisticCircuit
    :param node: SumUnit to duplicate.
    :type node: SumUnit
    :param noise_scale: Standard deviation of weight perturbation.
    :type noise_scale: float
    """

    parent_indices = list(circuit.graph.predecessors(node.index))

    duplicate = node.copy_without_graph()

    circuit.add_node(duplicate)

    for weight, child in node.log_weighted_subcircuits:

        duplicate.add_subcircuit(
            child,
            weight
            + np.random.normal(
                0,
                noise_scale,
            ),
        )

    if len(parent_indices) == 0:

        new_root = SumUnit(probabilistic_circuit=circuit)

        circuit.add_node(new_root)

        new_root.add_subcircuit(
            node,
            np.log(0.5),
        )

        new_root.add_subcircuit(
            duplicate,
            np.log(0.5),
        )

        return

    for parent in parent_indices:

        edge_weight = circuit.graph.get_edge_data(
            parent.index,
            node.index,
        )

        if edge_weight is None:
            continue

        parent.add_subcircuit(
            duplicate,
            edge_weight
            + np.random.normal(
                0,
                noise_scale,
            ),
        )

    duplicate.normalize()


def grow_nodes(
    circuit: ProbabilisticCircuit,
    data: np.ndarray | None = None,
    fraction: float = 0.2,
    noise_scale: float = 1e-3,
) -> ProbabilisticCircuit:
    """
    Increase circuit structure by duplicating SumUnits.

    :param circuit: Circuit to expand.
    :type circuit: ProbabilisticCircuit
    :param data: Dataset used to estimate node importance.
    :type data: np.ndarray | None
    :param fraction: Fraction of SumUnits to duplicate.
    :type fraction: float
    :param noise_scale: Standard deviation of weight perturbation.
    :type noise_scale: float
    :return: Expanded probabilistic circuit.
    :rtype: ProbabilisticCircuit
    """

    selected_nodes = select_sum_units_to_grow(
        circuit,
        data,
        fraction,
    )

    for node in selected_nodes:

        duplicate_sum_unit(
            circuit,
            node,
            noise_scale,
        )

    circuit._invalidate_topology_cache()

    return circuit


def sparse_probabilistic_circuit_learning(
    circuit: ProbabilisticCircuit,
    data: np.ndarray,
    prune_fraction: float = 0.1,
    grow_fraction: float = 0.2,
    noise_scale: float = 1e-3,
    iterations: int = 1,
) -> ProbabilisticCircuit:
    """
    Perform prune-and-grow structural learning.

    :param circuit: Initial probabilistic circuit.
    :type circuit: ProbabilisticCircuit
    :param data: Dataset used for optimization.
    :type data: np.ndarray
    :param prune_fraction: Fraction of edges removed.
    :type prune_fraction: float
    :param grow_fraction: Fraction of SumUnits duplicated.
    :type grow_fraction: float
    :param noise_scale: Weight perturbation magnitude.
    :type noise_scale: float
    :param iterations: Number of structural learning iterations.
    :type iterations: int
    :return: Best probabilistic circuit found.
    :rtype: ProbabilisticCircuit
    """

    best_circuit = copy.deepcopy(circuit)

    best_score = average_log_likelihood(
        circuit,
        data,
    )

    for _ in range(iterations):

        circuit = prune_edges(
            circuit,
            data,
            prune_fraction,
        )

        circuit = grow_nodes(
            circuit,
            data,
            grow_fraction,
            noise_scale,
        )

        current_score = average_log_likelihood(
            circuit,
            data,
        )

        if current_score > best_score:

            best_score = current_score

            best_circuit = copy.deepcopy(circuit)

    return best_circuit
