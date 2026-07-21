from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.jax.inner_layer import (
    SparseSumLayer,
    InnerLayer,
)
from probabilistic_model.probabilistic_circuit.jax.utils import copy_bcoo


def _compute_node_log_likelihoods(
    circuit: ProbabilisticCircuit,
    data: jax.Array,
) -> dict[int, jax.Array]:
    """
    Compute node log-likelihoods for every layer in a circuit.

    :param circuit: Probabilistic circuit to evaluate.
    :param data: Samples used for likelihood computation.
    :return: Mapping between layers and their node log-likelihood
        values.
    """
    node_log_likelihoods: dict[int, jax.Array] = {}

    for layer in reversed(circuit.root.all_layers()):
        node_log_likelihoods[id(layer)] = layer.log_likelihood_of_nodes(data)

    return node_log_likelihoods


def calculate_edge_flows(
    circuit: ProbabilisticCircuit, data: jax.Array
) -> list[jax.Array]:
    """
    Calculate average probability flow through circuit edges.

    :param circuit: Probabilistic circuit to analyse.
    :param data: Samples used to estimate edge flows.
    :return: Mean flow values for edges in sparse sum layers.
    """
    node_log_likelihoods = _compute_node_log_likelihoods(
        circuit,
        data,
    )

    flows_per_sum_layer = []

    for layer in circuit.root.all_layers():

        if isinstance(layer, SparseSumLayer):

            parent_log_likelihoods = node_log_likelihoods[id(layer)]

            for log_weights, child_layer in layer.log_weighted_child_layers:

                child_log_likelihoods = node_log_likelihoods[id(child_layer)]

                parent_indices = log_weights.indices[:, 0]
                child_indices = log_weights.indices[:, 1]

                edge_parent_log_likelihoods = parent_log_likelihoods[
                    :,
                    parent_indices,
                ]

                edge_child_log_likelihoods = child_log_likelihoods[
                    :,
                    child_indices,
                ]

                edge_log_flows = (
                    edge_child_log_likelihoods
                    + log_weights.data
                    - edge_parent_log_likelihoods
                )

                edge_flows = jnp.exp(edge_log_flows)

                mean_edge_flows = jnp.mean(
                    edge_flows,
                    axis=0,
                )

                flows_per_sum_layer.append(mean_edge_flows)

    return flows_per_sum_layer


def _duplicate_parent_rows(
    log_weights: BCOO,
    rows_to_duplicate: jax.Array,
    key: jax.Array,
    noise_scale: float,
) -> BCOO:
    """
    Duplicate selected parent nodes in sparse layer weights.

    :param log_weights: Sparse weight matrix.
    :param rows_to_duplicate: Parent nodes to duplicate.
    :param key: Random key used to generate perturbations.
    :param noise_scale: Scale of the noise added to duplicated weights.
    :return: Expanded sparse weight matrix.
    """
    number_of_parent_nodes = log_weights.shape[0]

    duplicated_indices = log_weights.indices[
        jnp.isin(
            log_weights.indices[:, 0],
            rows_to_duplicate,
        )
    ]

    duplicated_indices = duplicated_indices.at[:, 0].add(number_of_parent_nodes)

    duplicated_data = log_weights.data[
        jnp.isin(
            log_weights.indices[:, 0],
            rows_to_duplicate,
        )
    ]

    noise = (
        jax.random.normal(
            key,
            shape=duplicated_data.shape,
        )
        * noise_scale
    )

    duplicated_data = duplicated_data + noise

    new_indices = jnp.concatenate(
        [
            log_weights.indices,
            duplicated_indices,
        ],
        axis=0,
    )

    new_data = jnp.concatenate(
        [
            log_weights.data,
            duplicated_data,
        ],
        axis=0,
    )

    new_shape = (
        number_of_parent_nodes + len(rows_to_duplicate),
        log_weights.shape[1],
    )

    return BCOO(
        (
            new_data,
            new_indices,
        ),
        shape=new_shape,
        indices_sorted=False,
        unique_indices=True,
    ).sort_indices()


def prune_circuit_eflow(
    circuit: ProbabilisticCircuit,
    data: jax.Array,
    prune_fraction: float,
) -> ProbabilisticCircuit:
    """
    Remove edges with the lowest probability flow.

    :param circuit: Probabilistic circuit to prune.
    :param data: Samples used to estimate edge flows.
    :param prune_fraction: Fraction of edges removed from each sparse
        sum layer.
    :return: Structurally pruned probabilistic circuit.
    """
    mean_flows = calculate_edge_flows(
        circuit,
        data,
    )

    flow_index = 0

    def prune_layer(layer):
        nonlocal flow_index

        if isinstance(layer, InnerLayer):
            child_layers = [
                prune_layer(child_layer) for child_layer in layer.child_layers
            ]

            layer = eqx.tree_at(
                lambda current_layer: current_layer.child_layers,
                layer,
                child_layers,
            )

        if isinstance(layer, SparseSumLayer):
            new_log_weights = []

            for log_weights in layer.log_weights:
                edge_flows = mean_flows[flow_index]
                flow_index += 1

                number_to_remove = int(prune_fraction * edge_flows.shape[0])

                if number_to_remove > 0:
                    kept_edges = jnp.argsort(edge_flows)[number_to_remove:]

                    keep_mask = jnp.zeros_like(
                        edge_flows,
                        dtype=bool,
                    )

                    keep_mask = keep_mask.at[kept_edges].set(True)
                else:
                    keep_mask = jnp.ones_like(
                        edge_flows,
                        dtype=bool,
                    )

                copied_weights = copy_bcoo(log_weights)

                new_log_weights.append(
                    BCOO(
                        (
                            copied_weights.data[keep_mask],
                            copied_weights.indices[keep_mask],
                        ),
                        shape=copied_weights.shape,
                        indices_sorted=False,
                        unique_indices=True,
                    ).sort_indices()
                )

            layer = eqx.tree_at(
                lambda current_layer: current_layer.log_weights,
                layer,
                new_log_weights,
            )

        return layer

    return ProbabilisticCircuit(
        circuit.variables,
        prune_layer(circuit.root),
    )


def grow_circuit(
    circuit: ProbabilisticCircuit,
    key: jax.random.PRNGKey,
    grow_fraction: float = 0.2,
    noise_scale: float = 1e-3,
) -> ProbabilisticCircuit:
    """
    Increase circuit capacity by duplicating selected nodes.

    :param circuit: Probabilistic circuit to expand.
    :param key: Random key used during node duplication.
    :param grow_fraction: Fraction of nodes duplicated.
    :param noise_scale: Noise applied to duplicated parameters.
    :return: Expanded probabilistic circuit.
    """

    def grow_layer(layer, current_key):
        if isinstance(layer, InnerLayer):
            keys = jax.random.split(
                current_key,
                len(layer.child_layers) + 1,
            )

            new_child_layers = [
                grow_layer(
                    child_layer,
                    keys[i],
                )
                for i, child_layer in enumerate(layer.child_layers)
            ]

            layer = eqx.tree_at(
                lambda l: l.child_layers,
                layer,
                new_child_layers,
            )

            current_key = keys[-1]

        if isinstance(layer, SparseSumLayer):
            new_log_weights_list = []

            for log_weights in layer.log_weights:

                noise_key, current_key = jax.random.split(current_key)

                number_of_nodes = log_weights.shape[0]

                number_to_grow = max(
                    1,
                    int(number_of_nodes * grow_fraction),
                )

                selected_nodes = jax.random.choice(
                    noise_key,
                    number_of_nodes,
                    shape=(number_to_grow,),
                    replace=False,
                )

                expanded_bcoo = _duplicate_parent_rows(
                    log_weights,
                    selected_nodes,
                    noise_key,
                    noise_scale,
                )

                new_log_weights_list.append(expanded_bcoo)
            layer = eqx.tree_at(
                lambda l: l.log_weights,
                layer,
                new_log_weights_list,
            )

        return layer

    init_key, process_key = jax.random.split(key)
    new_root = grow_layer(circuit.root, process_key)

    return ProbabilisticCircuit(circuit.variables, new_root)


def prune_and_grow(
    circuit: ProbabilisticCircuit,
    data: jax.Array,
    key: jax.random.PRNGKey,
    prune_fraction: float = 0.1,
    grow_fraction: float = 0.2,
    noise_scale: float = 1e-3,
) -> ProbabilisticCircuit:
    """
    Apply pruning and growth structural learning.

    :param circuit: Probabilistic circuit to modify.
    :param data: Training samples used for pruning.
    :param key: Random key used during growth.
    :param prune_fraction: Fraction of edges removed.
    :param grow_fraction: Fraction of nodes duplicated.
    :param noise_scale: Noise applied during growth.
    :return: Structurally modified probabilistic circuit.
    """
    circuit = prune_circuit_eflow(
        circuit,
        data,
        prune_fraction,
    )

    circuit = grow_circuit(
        circuit,
        key,
        grow_fraction,
        noise_scale,
    )

    return circuit
