import numpy as np

from typing import Dict, Tuple

from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    SumUnit,
)


def evaluate_likelihood(
    circuit: ProbabilisticCircuit,
    data: np.ndarray,
) -> float:
    """
    Evaluate the average log likelihood of a probabilistic circuit.

    Args:
        circuit:
            Probabilistic circuit to evaluate.

        data:
            Dataset used for evaluation.

    Returns:
        Average log likelihood over the provided samples.
    """

    log_likelihood = circuit.log_likelihood(data)

    return float(np.mean(log_likelihood))


def calculate_edge_flows(
    circuit: ProbabilisticCircuit,
    data: np.ndarray,
) -> Dict[Tuple[object, object], float]:
    """
    Calculate the average probability flow through every weighted edge.

    Edge flows measure the contribution of every SumUnit connection
    according to the current dataset.

    Args:
        circuit:
            Probabilistic circuit where flows are computed.

        data:
            Dataset used for estimating flows.

    Returns:
        Mapping from edges to their average probability flow.
    """

    circuit.log_likelihood(data)

    flows: Dict[Tuple[object, object], float] = {}

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

    Args:
        circuit:
            Circuit to prune.

        data:
            Dataset used to compute edge importance.

        prune_fraction:
            Fraction of removable edges removed.

    Returns:
        Pruned circuit.
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


def grow_nodes(
    circuit: ProbabilisticCircuit,
    data: np.ndarray | None = None,
    fraction: float = 0.2,
    noise_scale: float = 1e-3,
) -> ProbabilisticCircuit:
    """
    Increase circuit structure by duplicating SumUnits.

    Selected SumUnits are copied and their outgoing weights are slightly
    perturbed. If a dataset is provided, nodes can be selected according
    to their estimated probability flow. Otherwise, SumUnits are selected
    uniformly.

    Args:
        circuit:
            Circuit to expand.

        data:
            Dataset used to estimate node importance. Optional.

        fraction:
            Fraction of SumUnits to duplicate.

        noise_scale:
            Standard deviation of the weight perturbation.

    Returns:
        Expanded probabilistic circuit.
    """

    sum_nodes = [node for node in circuit.graph.nodes() if isinstance(node, SumUnit)]

    if not sum_nodes:
        return circuit

    # ------------------------------------------------------------
    # Node selection
    # ------------------------------------------------------------
    # If data is available, use edge flows as importance measure.
    # Otherwise keep the previous uniform random selection.
    # ------------------------------------------------------------

    if data is not None:

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

        sorted_nodes = sorted(
            sum_nodes,
            key=lambda n: node_scores[n],
            reverse=True,
        )

        number_to_duplicate = max(
            1,
            int(len(sum_nodes) * fraction),
        )

        selected_nodes = sorted_nodes[:number_to_duplicate]

    else:

        number_to_duplicate = max(
            1,
            int(len(sum_nodes) * fraction),
        )

        selected_nodes = np.random.choice(
            sum_nodes,
            size=number_to_duplicate,
            replace=False,
        )

    for node in selected_nodes:

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

            continue

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

    circuit._invalidate_topology_cache()

    return circuit


def sparse_pc_learning(
    circuit: ProbabilisticCircuit,
    data: np.ndarray,
    prune_fraction: float = 0.1,
    grow_fraction: float = 0.2,
    noise_scale: float = 1e-3,
    iterations: int = 1,
) -> ProbabilisticCircuit:
    """
    Perform iterative structural optimization.

    Each iteration removes unnecessary connections and increases
    capacity by duplicating relevant components.

    Args:
        circuit:
            Initial probabilistic circuit.

        data:
            Dataset used for optimization.

        prune_fraction:
            Fraction of edges removed.

        grow_fraction:
            Fraction of SumUnits duplicated.

        noise_scale:
            Weight perturbation magnitude.

        iterations:
            Number of optimization iterations.

    Returns:
        Optimized probabilistic circuit.
    """

    best_score = evaluate_likelihood(
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

        new_score = evaluate_likelihood(
            circuit,
            data,
        )

        # Keep track of the current best structure.
        best_score = max(
            best_score,
            new_score,
        )

    return circuit
