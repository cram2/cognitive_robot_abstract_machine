import jax
import jax.numpy as jnp
import equinox as eqx
from jax.experimental.sparse import BCOO

from probabilistic_model.probabilistic_circuit.jax.learning import (
    prune_circuit_eflow,
    grow_circuit,
)
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.jax.inner_layer import SparseSumLayer


def create_tiny_dummy_circuit():
    """
    Generates a minimal probabilistic circuit for testing structural learning.
    """

    indices = jnp.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )

    data_weights = jnp.array(
        [
            -0.693,
            -0.693,
            -0.693,
            -0.693,
        ]
    )

    bcoo_weights = BCOO(
        (
            data_weights,
            indices,
        ),
        shape=(2, 2),
    )

    class DummyInputLayer(eqx.Module):
        def log_likelihood_of_nodes_single(self, x):
            return jnp.zeros((2,))

        def log_likelihood_of_nodes(self, data):
            return jnp.zeros(
                (
                    data.shape[0],
                    2,
                )
            )

        def all_layers(self):
            return [self]

        @property
        def variables(self):
            return (0, 1)

        @property
        def number_of_nodes(self):
            return 2

    child_layer = DummyInputLayer()

    sum_layer = SparseSumLayer(
        child_layers=[
            child_layer,
        ],
        log_weights=[
            bcoo_weights,
        ],
    )

    return ProbabilisticCircuit(
        variables=(0, 1),
        root=sum_layer,
    )


def test_jax_structural_learning():
    """
    Tests EFLOW pruning and GROW structural expansion.
    """

    key = jax.random.PRNGKey(42)

    dummy_data = jnp.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    circuit = create_tiny_dummy_circuit()

    # Initial structure
    assert circuit.root.log_weights[0].shape == (
        2,
        2,
    )

    assert circuit.root.log_weights[0].data.shape[0] == 4

    # -------------------------
    # EFLOW PRUNING
    # -------------------------

    pruned_circuit = prune_circuit_eflow(
        circuit,
        dummy_data,
        tau=0.6,
    )

    assert pruned_circuit is not None

    # -------------------------
    # GROW EXPANSION
    # -------------------------

    expanded_circuit = grow_circuit(
        pruned_circuit,
        key,
        noise_scale=1e-3,
    )

    expected_rows = pruned_circuit.root.log_weights[0].shape[0] * 2

    assert (
        expanded_circuit.root.log_weights[0].shape[0] == expected_rows
    ), "GROW failed to double the parent node capacity."

    # Check columns are still valid
    assert (
        expanded_circuit.root.log_weights[0].shape[1]
        == pruned_circuit.root.log_weights[0].shape[1]
    ), "Child dimension changed unexpectedly."
