import copy
import numpy as np

from typing import Dict, Tuple

from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    SumUnit,
    Unit,
)


def average_log_likelihood(
    circuit: ProbabilisticCircuit,
    data: np.ndarray,
) -> float:
    """
    Evaluate the average log likelihood of a probabilistic circuit.

    :param circuit: Probabilistic circuit to evaluate.
    :param data: Dataset used for evaluation.
    :return: Average log likelihood over the provided samples.
    """

    log_likelihood = circuit.log_likelihood(data)

    return float(np.mean(log_likelihood))


def calculate_edge_flows(
    circuit: ProbabilisticCircuit,
    data: np.ndarray,
) -> Dict[Tuple[Unit, Unit], float]:
    """
    Calculate the average probability flow through every weighted edge.

    Edge flows measure the contribution of every SumUnit connection
    according to the current dataset.

    :param circuit: Probabilistic circuit where flows are computed.
    :param data: Dataset used for estimating flows.
    :return: Mapping from edges to their average probability flow.
    """

    circuit.log_likelihood(data)

    flows: Dict[Tuple[Unit, Unit], float] = {}

    for parent, child, log_weight in circuit.edges():

        if not isinstance(parent, SumUnit):
            continue

        parent_likelihood = parent.result_of_current_query
        child_likelihood = child.result_of_current_query

        flow = np.exp(child_likelihood + log_weight - parent_likelihood)

        flows[(parent, child)] = float(np.mean(flow))

    return flows


def prune_edges(
    circuit: ProbabilisticCircuit,
    data: np.ndarray,
    prune_fraction: float = 0.1,
) -> ProbabilisticCircuit:
    """
    Remove low-contribution edges from SumUnits.

    :param circuit: Circuit to prune.
    :param data: Dataset used to compute edge importance.
    :param prune_fraction: Fraction of removable edges removed.
    :return: Pruned circuit.
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
                        (node, child),
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
    :param data: Dataset used for importance estimation.
    :param fraction: Fraction of nodes selected.
    :return: Selected SumUnits.
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
                (node, child),
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
    :param node: SumUnit to duplicate.
    :param noise_scale: Standard deviation of weight perturbation.
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
    :param data: Dataset used to estimate node importance.
    :param fraction: Fraction of SumUnits to duplicate.
    :param noise_scale: Standard deviation of weight perturbation.
    :return: Expanded probabilistic circuit.
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


def update_sum_weights(
    circuit: ProbabilisticCircuit,
    data: np.ndarray,
) -> None:
    """
    Update the weights of SumUnits based on the estimated probability flows.

    The function re-estimates the outgoing weights of every SumUnit using
    the average probability flow through each edge. The new weights are
    normalized so that they form a valid probability distribution.

    :param circuit: Probabilistic circuit whose SumUnit weights are updated.
    :param data: Dataset used to estimate edge probability flows.
    """

    flows = calculate_edge_flows(circuit, data)

    for node in circuit.graph.nodes():

        if not isinstance(node, SumUnit):
            continue

        children = list(node.subcircuits)

        if len(children) == 0:
            continue

        values = np.array(
            [flows.get((node, child), 0.0) for child in children],
            dtype=float,
        )

        if values.sum() == 0:
            continue

        values /= values.sum()

        for child in children:
            circuit.graph.remove_edge(node.index, child.index)

        for child, weight in zip(children, values):
            node.add_subcircuit(child, np.log(weight))

    circuit._invalidate_topology_cache()


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
    :param data: Dataset used for optimization.
    :param prune_fraction: Fraction of edges removed.
    :param grow_fraction: Fraction of SumUnits duplicated.
    :param noise_scale: Weight perturbation magnitude.
    :param iterations: Number of structural learning iterations.
    :return: Best probabilistic circuit found.
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

        update_sum_weights(
            circuit,
            data,
        )

        current_score = average_log_likelihood(
            circuit,
            data,
        )

        if current_score > best_score:

            best_score = current_score

            best_circuit = copy.deepcopy(circuit)

    return best_circuit
