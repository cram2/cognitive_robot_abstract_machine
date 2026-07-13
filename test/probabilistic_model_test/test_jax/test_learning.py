import jax
import jax.numpy as jnp
import equinox as eqx
from jax.experimental.sparse import BCOO

# Absolute imports required by the monorepo architecture
from probabilistic_model.probabilistic_circuit.jax.learning import (
    prune_circuit_eflow,
    grow_circuit,
)
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.jax.inner_layer import SparseSumLayer


def create_tiny_dummy_circuit():
    """Generates a toy probabilistic circuit from scratch for isolation testing."""
    # Define a basic BCOO sparse structure (2 sum nodes connected to 2 children)
    # Coordinate layout: [row, column]
    indices = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Initial weights in log scale (e.g., log(0.5) = -0.693)
    data_weights = jnp.array([-0.693, -0.693, -0.693, -0.693])

    # Construct the JAX sparse BCOO matrix
    bcoo_weights = BCOO((data_weights, indices), shape=(2, 2))

    # Mock an input layer to serve as the child reference
    class DummyInputLayer(eqx.Module):
        def log_likelihood_of_nodes_single(self, x):
            # Simulates flat leaf distribution for a single sample vector
            return jnp.zeros((2,))

        def log_likelihood_of_nodes(self, data):
            # Computes flat distribution for the entire batch matrix
            return jnp.zeros((data.shape[0], 2))

        def all_layers(self):
            return [self]

        @property
        def variables(self):
            return (0, 1)

    child_layer = DummyInputLayer()

    # Instantiate the sum layer using the project's native object schema
    sum_layer = SparseSumLayer(child_layers=[child_layer], log_weights=[bcoo_weights])

    # Return the wrapped complete ProbabilisticCircuit instance
    return ProbabilisticCircuit(variables=(0, 1), root=sum_layer)


def test_jax_structural_learning():
    """Automated unit test validating EFLOW Pruning and GROW Expansion pipelines."""
    key = jax.random.PRNGKey(42)
    dummy_data = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # Sample evaluation dataset

    # 1. Verify initial state of the constructed toy circuit
    circuit = create_tiny_dummy_circuit()
    assert circuit.root.log_weights[0].shape == (2, 2), "Initial tensor shape mismatch."
    assert (
        circuit.root.log_weights[0].data.shape[0] == 4
    ), "Initial active edge count must be 4."

    # 2. Validate EFLOW Pruning
    # A high tau value (0.6) forces low-flow components to be effectively masked
    pruned_circuit = prune_circuit_eflow(circuit, dummy_data, tau=0.6)
    assert pruned_circuit is not None, "Pruned circuit instance should not be None."

    # 3. Validate GROW Expansion
    expanded_circuit = grow_circuit(pruned_circuit, key, noise_scale=1e-3)

    # structural integrity assertion: doubling the nodes implies doubling the vertical shape of the BCOO matrix
    expected_rows = circuit.root.log_weights[0].shape[0] * 2
    assert (
        expanded_circuit.root.log_weights[0].shape[0] == expected_rows
    ), "GROW algorithm failed to expand node structural capacity."
