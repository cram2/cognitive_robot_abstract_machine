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
    """Compute the log-likelihoods for every layer in the circuit."""

    node_log_likelihoods: dict[int, jax.Array] = {}

    for layer in reversed(circuit.root.all_layers()):
        node_log_likelihoods[id(layer)] = layer.log_likelihood_of_nodes(data)

    return node_log_likelihoods


def calculate_edge_flows(
    circuit: ProbabilisticCircuit, data: jax.Array
) -> list[jax.Array]:
    """
    Calculate the mean probability flow for every edge in each sparse sum layer.

    :param circuit:
        Probabilistic circuit to analyse.
    :param data:
        Batch of samples used to estimate the edge flows.
    :returns:
        Mean flow for every edge of every sparse sum layer.
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
    Duplicate selected parent nodes in a sparse weight matrix.
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
    circuit: ProbabilisticCircuit, data: jax.Array, prune_fraction: float
) -> ProbabilisticCircuit:
    """
    EFLOW Pruning Algorithm: Evaluates edge flows, identifies redundant edges
    with a mean flow below tau, and reconstructs the circuit structure.

    :param circuit: The original ProbabilisticCircuit instance.
    :param data: Input data array used to calculate current edge flows.
    :param tau: Threshold tolerance for edge flow (edges with flow < tau are pruned).
    :return: A new structurally modified ProbabilisticCircuit instance.
    """
    # 1. Compute the mean flow passing through every active edge
    mean_flows_list = calculate_edge_flows(circuit, data)

    log_weight_matrix_index = 0

    # 2. Inner function to recursively traverse and prune the Pytree layers bottom-up
    def prune_layer(layer):
        if isinstance(layer, InnerLayer):
            new_child_layers = [prune_layer(cl) for cl in layer.child_layers]
            # Handle immutable update using Equinox tree tools
            layer = eqx.tree_at(lambda l: l.child_layers, layer, new_child_layers)

        if isinstance(layer, SparseSumLayer):
            nonlocal log_weight_matrix_index

            new_log_weights_list = []

            for log_weights in layer.log_weights:
                mean_edge_flows = mean_flows_list[log_weight_matrix_index]
                log_weight_matrix_index += 1
                # Boolean mask: True to keep the edge, False to prune it
                number_of_edges = mean_edge_flows.shape[0]

                number_to_prune = int(prune_fraction * number_of_edges)

                if number_to_prune > 0:
                    sorted_indices = jnp.argsort(mean_edge_flows)

                    edges_to_keep = sorted_indices[number_to_prune:]

                    keep_mask = jnp.zeros_like(
                        mean_edge_flows,
                        dtype=bool,
                    )

                    keep_mask = keep_mask.at[edges_to_keep].set(True)
                else:
                    keep_mask = jnp.ones_like(
                        mean_edge_flows,
                        dtype=bool,
                    )

                cloned_weights = copy_bcoo(log_weights)

                # Keep only edges with sufficient flow
                new_indices = cloned_weights.indices[keep_mask]
                new_data = cloned_weights.data[keep_mask]

                updated_bcoo = BCOO(
                    (
                        new_data,
                        new_indices,
                    ),
                    shape=cloned_weights.shape,
                    indices_sorted=False,
                    unique_indices=True,
                )

                updated_bcoo = updated_bcoo.sort_indices()

                new_log_weights_list.append(updated_bcoo)

            # Safely replace the log weights list in the current Equinox module layer
            layer = eqx.tree_at(
                lambda l: l.log_weights,
                layer,
                new_log_weights_list,
            )

        return layer

    # 3. Process the complete architecture starting from the root
    new_root = prune_layer(circuit.root)

    # 4. Return the newly built pruned circuit wrapper
    return ProbabilisticCircuit(circuit.variables, new_root)


def grow_circuit(
    circuit: ProbabilisticCircuit,
    key: jax.random.PRNGKey,
    grow_fraction: float = 0.2,
    noise_scale: float = 1e-3,
) -> ProbabilisticCircuit:
    """
    GROW Algorithm: Identifies candidate nodes to split, duplicates their parameters
    to increase structural capacity, and injects a small amount of random noise
    to break gradient symmetry during subsequent training.

    :param circuit: The current ProbabilisticCircuit instance.
    :param key: JAX random key for noise generation.
    :param noise_scale: Standard deviation of the Gaussian noise applied to new weights.
    :return: A new structurally expanded ProbabilisticCircuit instance.
    """

    def grow_layer(layer, current_key):
        # 1. Recurse through child layers first
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

        # 2. Expand SparseSumLayer nodes
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

    # 3. Process the complete architecture starting from the root node
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
    Apply the complete structural learning procedure:
    EFLOW pruning followed by GROW expansion.

    :param circuit:
        Probabilistic circuit to modify.
    :param data:
        Training data used to estimate edge flows.
    :param key:
        JAX random key used during growth.
    :param prune_fraction:
        Fraction of low-flow edges to remove.
    :param grow_fraction:
        Fraction of nodes to duplicate.
    :param noise_scale:
        Noise added to duplicated parameters.
    :return:
        New structurally modified probabilistic circuit.
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
