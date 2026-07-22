import numpy as np

from probabilistic_model.probabilistic_circuit.rx.learning import (
    calculate_edge_flows,
    prune_edges,
    grow_nodes,
    sparse_pc_learning,
)

from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    SumUnit,
    ProductUnit,
    Continuous,
    leaf,
)

from probabilistic_model.distributions.gaussian import GaussianDistribution


def create_tiny_dummy_circuit() -> ProbabilisticCircuit:
    """
    Create a small probabilistic circuit used for testing.

    Structure:

              Sum
             /   \
          Prod   Prod
           |      |
         Leaf    Leaf

    Returns:
        A small Gaussian mixture probabilistic circuit.
    """

    x = Continuous("x")

    pc = ProbabilisticCircuit()

    root = SumUnit(probabilistic_circuit=pc)

    product1 = ProductUnit(probabilistic_circuit=pc)

    product2 = ProductUnit(probabilistic_circuit=pc)

    leaf1 = leaf(
        GaussianDistribution(
            variable=x,
            location=0.0,
            scale=1.0,
        ),
        pc,
    )

    leaf2 = leaf(
        GaussianDistribution(
            variable=x,
            location=5.0,
            scale=1.0,
        ),
        pc,
    )

    product1.add_subcircuit(leaf1)
    product2.add_subcircuit(leaf2)

    root.add_subcircuit(
        product1,
        np.log(0.5),
    )

    root.add_subcircuit(
        product2,
        np.log(0.5),
    )

    return pc


def test_calculate_edge_flows():
    """
    Verify that edge flows are computed correctly.
    """

    circuit = create_tiny_dummy_circuit()

    data = np.array(
        [
            [0.0],
            [5.0],
        ]
    )

    flows = calculate_edge_flows(
        circuit,
        data,
    )

    assert len(flows) > 0

    for _, flow in flows.items():

        assert flow >= 0


def test_pruning_removes_edges():
    """
    Verify that pruning decreases the number of edges.
    """

    circuit = create_tiny_dummy_circuit()

    data = np.array(
        [
            [0.0],
            [5.0],
        ]
    )

    before = len(list(circuit.edges()))

    prune_edges(
        circuit,
        data,
        prune_fraction=0.5,
    )

    after = len(list(circuit.edges()))

    assert after < before


def test_growing_increases_structure():
    """
    Verify that growing increases the number of nodes.
    """

    circuit = create_tiny_dummy_circuit()

    before = len(list(circuit.graph.nodes()))

    grow_nodes(
        circuit,
        fraction=1.0,
        noise_scale=1e-3,
    )

    after = len(list(circuit.graph.nodes()))

    assert after > before


def test_sparse_pc_learning():
    """
    Verify that the complete learning pipeline returns
    a valid probabilistic circuit.
    """

    circuit = create_tiny_dummy_circuit()

    data = np.array(
        [
            [0.0],
            [5.0],
        ]
    )

    result = sparse_pc_learning(
        circuit,
        data,
        prune_fraction=0.2,
        grow_fraction=0.5,
        iterations=2,
    )

    assert result is not None

    assert isinstance(
        result,
        ProbabilisticCircuit,
    )
