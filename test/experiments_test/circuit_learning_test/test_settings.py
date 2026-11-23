"""
Tests of how a setting places a learned circuit: folded, with its edges stored as the
setting says.
"""

from __future__ import annotations

# %% imports
import numpy as np
import pytest

from experiments.circuit_learning.datasets import BenchmarkDataset
from experiments.circuit_learning.methods import (
    JointProbabilityTreeBaseline,
    TrainingContext,
)
from experiments.circuit_learning.settings import (
    ComputationSetting,
    EdgeRepresentation,
)
from probabilistic_model.learning.parameters.training_schedule import Convergence
from probabilistic_model.probabilistic_circuit.tensorized.folding import LayerFolding
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import (
    InnerLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.edge_matrix import (
    BlockEdgeMatrix,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
    SumLayer,
)

# %% fixtures


@pytest.fixture
def tree_circuit(dataset: BenchmarkDataset) -> LayeredProbabilisticCircuit:
    """
    :return: The layered circuit of a joint probability tree learned on the dataset.
    """
    return (
        JointProbabilityTreeBaseline(minimum_samples_per_leaf=0.2)
        .learn(
            dataset=dataset,
            setting=ComputationSetting.host(),
            context=TrainingContext(
                convergence=Convergence(maximum_epochs=1), memory_budget_bytes=2**30
            ),
        )
        .circuit
    )


# %% placing


def test_the_joint_probability_tree_is_folded(tree_circuit):
    """
    The circuit of a joint probability tree has one layer per type of layer and height.
    """
    heights = LayerFolding(root=tree_circuit.root).heights
    kinds = [(type(layer), heights[layer]) for layer in tree_circuit.layers]
    assert len(kinds) == len(set(kinds))


def test_placing_keeps_the_likelihood(tree_circuit, dataset):
    """
    A placed circuit gives every test row the likelihood of the circuit it was placed
    from.
    """
    rows = dataset.encoded(frame=dataset.test)
    np.testing.assert_allclose(
        ComputationSetting.host()
        .placed(circuit=tree_circuit)
        .log_likelihood(events=rows),
        tree_circuit.log_likelihood(events=rows),
    )


def test_block_edges_are_stored_as_block_edge_matrices(tree_circuit):
    """
    A setting with block edges stores the edges of every inner layer as a block edge
    matrix.
    """
    setting = ComputationSetting(
        array_library=ComputationSetting.host().array_library,
        device=ComputationSetting.host().device,
        floating_point_type=ComputationSetting.host().floating_point_type,
        edge_representation=EdgeRepresentation.BLOCK,
    )
    placed = setting.placed(circuit=tree_circuit)
    inner_layers = [layer for layer in placed.layers if isinstance(layer, InnerLayer)]
    assert any(isinstance(layer, SumLayer) for layer in inner_layers)
    assert all(isinstance(layer.edges, BlockEdgeMatrix) for layer in inner_layers)
