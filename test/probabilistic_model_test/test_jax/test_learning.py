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

from probabilistic_model.probabilistic_circuit.jax.learning import (
    calculate_edge_flows,
)
from probabilistic_model.probabilistic_circuit.jax.learning import (
    prune_and_grow,
)


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
    Test pruning followed by growth modifies the circuit structure.
    """
    key = jax.random.PRNGKey(42)

    dummy_data = jnp.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    circuit = create_tiny_dummy_circuit()

    assert circuit.root.log_weights[0].shape == (
        2,
        2,
    )

    assert circuit.root.log_weights[0].data.shape[0] == 4

    pruned_circuit = prune_circuit_eflow(
        circuit,
        dummy_data,
        prune_fraction=0.5,
    )

    assert pruned_circuit is not None

    original_edges = circuit.root.log_weights[0].nse

    pruned_edges = pruned_circuit.root.log_weights[0].nse

    assert pruned_edges < original_edges, "Pruning did not remove any edges."

    assert jnp.all(jnp.isfinite(pruned_circuit.root.log_weights[0].data))

    expanded_circuit = grow_circuit(
        pruned_circuit,
        key,
        grow_fraction=0.5,
        noise_scale=1e-3,
    )

    initial_rows = pruned_circuit.root.log_weights[0].shape[0]

    expanded_rows = expanded_circuit.root.log_weights[0].shape[0]

    assert expanded_rows > initial_rows, "GROW failed to increase the number of nodes."

    assert (
        expanded_circuit.root.log_weights[0].shape[1]
        == pruned_circuit.root.log_weights[0].shape[1]
    ), "Child dimension changed unexpectedly."


def test_calculate_edge_flows():
    """
    Test that edge flows are computed for sparse sum layers.
    """
    circuit = create_tiny_dummy_circuit()

    data = jnp.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    flows = calculate_edge_flows(
        circuit,
        data,
    )

    assert len(flows) == 1

    assert flows[0].shape[0] == 4

    assert jnp.all(flows[0] >= 0)


def test_pruning_removes_low_flow_edges():
    """
    Test that pruning removes edges with low estimated flow.
    """
    circuit = create_tiny_dummy_circuit()

    data = jnp.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    pruned = prune_circuit_eflow(
        circuit,
        data,
        prune_fraction=0.5,
    )

    assert pruned.root.log_weights[0].nse < circuit.root.log_weights[0].nse


def test_growing_increases_nodes():
    """
    Test that growth increases the number of parent nodes.
    """
    circuit = create_tiny_dummy_circuit()

    key = jax.random.PRNGKey(0)

    grown = grow_circuit(
        circuit,
        key,
        grow_fraction=0.5,
    )

    assert grown.root.log_weights[0].shape[0] > circuit.root.log_weights[0].shape[0]


def test_prune_and_grow_pipeline():
    """
    Test the complete structural learning pipeline.
    """
    circuit = create_tiny_dummy_circuit()

    data = jnp.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    key = jax.random.PRNGKey(0)

    result = prune_and_grow(
        circuit,
        data,
        key,
        prune_fraction=0.5,
        grow_fraction=0.5,
    )

    assert result is not None
    assert isinstance(
        result,
        ProbabilisticCircuit,
    )
