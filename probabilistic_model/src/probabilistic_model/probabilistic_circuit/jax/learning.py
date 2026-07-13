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

    all_layers = list(circuit.root.all_layers())

    node_log_likelihoods = _compute_node_log_likelihoods(
        circuit,
        data,
    )

    flows_per_sum_layer = []

    for layer in circuit.root.all_layers():
        if isinstance(layer, SparseSumLayer):
            for log_weights, child_layer in layer.log_weighted_child_layers:
                child_log_likelihoods = node_log_likelihoods[id(child_layer)]

                child_node_indices = log_weights.indices[:, 1]

                edge_child_log_likelihoods = child_log_likelihoods[
                    :, child_node_indices
                ]

                edge_log_flows = edge_child_log_likelihoods + log_weights.data

                edge_flows = jnp.exp(edge_log_flows)

                mean_edge_flows = jnp.mean(edge_flows, axis=0)

                flows_per_sum_layer.append(mean_edge_flows)

    return flows_per_sum_layer


def prune_circuit_eflow(
    circuit: ProbabilisticCircuit, data: jax.Array, tau: float
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
                keep_mask = mean_edge_flows >= tau

                # Clone BCOO to avoid unexpected side effects in JAX memory management
                cloned_weights = copy_bcoo(log_weights)

                # Update sparse data: send low flow edges to -inf (linear weight of 0)
                new_data = jnp.where(
                    keep_mask,
                    cloned_weights.data,
                    -jnp.inf,
                )

                # Reconstruct the updated BCOO tensor with identical tracking indices
                updated_bcoo = BCOO(
                    (new_data, cloned_weights.indices),
                    shape=cloned_weights.shape,
                    indices_sorted=True,
                    unique_indices=True,
                )

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
    circuit: ProbabilisticCircuit, key: jax.random.PRNGKey, noise_scale: float = 1e-3
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
                grow_layer(child_layer, keys[i])
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
                cloned_weights = copy_bcoo(log_weights)

                number_of_original_edges = cloned_weights.data.shape[0]

                duplicated_indices = cloned_weights.indices.copy()

                number_of_nodes = cloned_weights.shape[0]

                duplicated_indices = duplicated_indices.at[:, 0].add(number_of_nodes)

                new_indices = jnp.concatenate(
                    [
                        cloned_weights.indices,
                        duplicated_indices,
                    ],
                    axis=0,
                )

                noise_key, current_key = jax.random.split(current_key)

                noise = (
                    jax.random.normal(
                        noise_key,
                        shape=(number_of_original_edges,),
                    )
                    * noise_scale
                )

                new_log_weight_values = jnp.concatenate(
                    [
                        cloned_weights.data,
                        cloned_weights.data + noise,
                    ],
                    axis=0,
                )

                new_shape = (
                    cloned_weights.shape[0] * 2,
                    cloned_weights.shape[1],
                )

                expanded_bcoo = BCOO(
                    (
                        new_log_weight_values,
                        new_indices,
                    ),
                    shape=new_shape,
                    indices_sorted=False,
                    unique_indices=True,
                )

                expanded_bcoo = expanded_bcoo.sort_indices()

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
