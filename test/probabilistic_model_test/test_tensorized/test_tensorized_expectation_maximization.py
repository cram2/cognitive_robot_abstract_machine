"""
Tests for expectation maximization of layered circuits with jax and torch: it learns the
parameters the host expectation maximization learns, on every type of edge matrix, and
blends its updates on mini-batches with a step size.
"""

from __future__ import annotations

# %% imports
import copy
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from array_api_compat import array_namespace
from random_events.set import Set
from random_events.variable import Symbolic
from sortedcontainers import SortedSet

from probabilistic_model.adapters.circuit_representations import (
    CircuitRepresentations,
)
from probabilistic_model.learning.parameters.expectation_maximization.exceptions import (
    UnsupportedLayerError,
)
from probabilistic_model.learning.parameters.expectation_maximization.expectation_maximization import (
    ExpectationMaximization,
    LearnableUnits,
)
from probabilistic_model.learning.parameters.expectation_maximization.jax_expectation_maximization import (
    JaxExpectationMaximization,
)
from probabilistic_model.learning.structure.region_graph.region_graph import RegionGraph
from probabilistic_model.learning.parameters.expectation_maximization.tensorized_expectation_maximization import (
    TensorizedExpectationMaximization,
)
from probabilistic_model.learning.parameters.training_schedule import (
    Convergence,
    MiniBatches,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.has_parameter_arrays import (
    HasParameterArrays,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import (
    Layer,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.edge_matrix import (
    BlockEdgeMatrix,
    EdgeMatrix,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
    SumLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.discrete_layer import (
    DiscreteLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.gaussian_layer import (
    GaussianLayer,
    TruncatedGaussianLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)

from ..test_region_graphs.test_region_graph import continuous_variables
from .test_inner_layers_on_array_backends import MixtureOfProductsCase

# %% circuits and rows


@dataclass(kw_only=True)
class RandomSumProductNetworkCase:
    """
    A small RAT-SPN on the host and its training rows.
    """

    host: ProbabilisticCircuit
    """
    The circuit as rustworkx units, which the host expectation maximization learns.
    """

    rows: np.ndarray
    """
    The training rows, columns ordered like the variables of the circuit.
    """

    def layered(
        self,
        *,
        host: ProbabilisticCircuit,
        namespace: ModuleType,
        edge_matrix_type: type[EdgeMatrix],
    ) -> LayeredProbabilisticCircuit:
        """
        :param host: The circuit as rustworkx units, :attr:`host` or a learned copy.
        :param namespace: The array namespace of the parameters.
        :param edge_matrix_type: The type of edge matrix of every inner layer.
        :return: The layered circuit of ``host``.
        """
        return (
            CircuitRepresentations()
            .convert(host, LayeredProbabilisticCircuit)
            .store_edges_as(edge_matrix_type)
            .to_namespace(namespace)
        )

    def host_after(self, *, iterations: int) -> ProbabilisticCircuit:
        """
        :param iterations: The number of iterations.
        :return: A copy of :attr:`host` after iterations of the host expectation
            maximization on all rows.
        """
        learned = copy.deepcopy(self.host)
        ExpectationMaximization().learn(
            learned, self.rows, LearnableUnits.from_units(learned.nodes()), iterations
        )
        return learned


class VariableKinds(StrEnum):
    """
    The kinds of variables of the RAT-SPN of a test.
    """

    CONTINUOUS = "continuous"
    """
    Four continuous variables, learned by Gaussian leaves.
    """

    MIXED = "mixed"
    """
    Two continuous and two symbolic variables, the symbolic ones learned by dense
    probability tables.
    """


@pytest.fixture(params=list(VariableKinds))
def random_sum_product_network(request) -> RandomSumProductNetworkCase:
    """
    A RAT-SPN over four variables with three input distributions per leaf region and
    two sums per region, and 40 rows, the continuous values drawn from two clusters.
    """
    generator = np.random.default_rng(5)
    continuous_rows = np.concatenate(
        [
            generator.normal(loc=-1.0, scale=0.5, size=(25, 4)),
            generator.normal(loc=2.0, scale=1.0, size=(15, 4)),
        ]
    )
    variables = continuous_variables(4)
    rows = continuous_rows
    if request.param is VariableKinds.MIXED:
        variables = SortedSet(
            [
                Symbolic(name="s", domain=Set.from_iterable(iterable=[0, 1, 2])),
                Symbolic(name="t", domain=Set.from_iterable(iterable=[0, 1])),
                *continuous_variables(2, prefix="x"),
            ]
        )
        # the columns follow the variables sorted by name: s, t, x0, x1
        rows = np.column_stack(
            [
                generator.integers(0, 3, len(continuous_rows)),
                continuous_rows[:, 0] > 0,
                continuous_rows[:, :2],
            ]
        ).astype(float)
    layered = (
        RegionGraph(variables=variables, partitions=2, depth=2, repetitions=2, seed=3)
        .create_random_region_graph()
        .to_probabilistic_circuit(input_units=3, sum_units=2)
    )
    return RandomSumProductNetworkCase(
        host=CircuitRepresentations().convert(layered, ProbabilisticCircuit),
        rows=rows,
    )


def comparable_parameters(root: Layer) -> list[np.ndarray]:
    """
    :param root: The root of a circuit.
    :return: The normalized log-weights of every sum layer and the parameters of every
        other layer, on the host.
    """
    return [
        HasParameterArrays.host_copy(parameter)
        for layer in root.all_layers()
        for parameter in (
            [layer.normalized_log_weights]
            if isinstance(layer, SumLayer)
            else layer.parameters
        )
    ]


def learner_of(
    *, layer: Layer, namespace: ModuleType, **settings: object
) -> TensorizedExpectationMaximization:
    """
    :param layer: The root of the circuit to learn.
    :param namespace: The array namespace of its parameters.
    :param settings: Further fields of the learner.
    :return: The expectation maximization of that backend.
    """
    if namespace is jnp:
        return JaxExpectationMaximization(layer=layer, **settings)
    torch_expectation_maximization = pytest.importorskip(
        "probabilistic_model.learning.parameters.expectation_maximization.torch_expectation_maximization"
    )
    return torch_expectation_maximization.TorchExpectationMaximization(
        layer=layer, **settings
    )


def epochs_of(epochs: int) -> Convergence:
    """
    :param epochs: The number of epochs.
    :return: A criterion that stops after exactly that many epochs.
    """
    return Convergence(maximum_epochs=epochs, patience=epochs)


# %% agreement with the host expectation maximization


@pytest.mark.parametrize("iterations", [1, 3])
@pytest.mark.parametrize("batch_size", [None, 16], ids=["one-batch", "batches-of-16"])
def test_full_batch_iterations_learn_the_parameters_of_host_expectation_maximization(
    random_sum_product_network,
    differentiable_backend,
    edge_matrix_type,
    iterations,
    batch_size,
):
    """
    Every epoch that updates once after all batches is one iteration of the host
    expectation maximization on all rows.
    """
    case = random_sum_product_network
    circuit = case.layered(
        host=case.host,
        namespace=differentiable_backend,
        edge_matrix_type=edge_matrix_type,
    )
    run = learner_of(
        layer=circuit.root, namespace=differentiable_backend
    ).minimize_until_converged(
        differentiable_backend.asarray(circuit.encoded(case.rows)),
        convergence=epochs_of(iterations),
        mini_batches=MiniBatches(batch_size=batch_size),
    )
    expected = case.layered(
        host=case.host_after(iterations=iterations),
        namespace=np,
        edge_matrix_type=edge_matrix_type,
    )

    for learned, reference in zip(
        comparable_parameters(run.trained_root),
        comparable_parameters(expected.root),
        strict=True,
    ):
        np.testing.assert_allclose(learned, reference, rtol=1e-9, atol=1e-12)


def test_the_loss_of_an_epoch_is_the_negative_average_log_likelihood_before_it(
    random_sum_product_network, differentiable_backend
):
    """
    Like the host expectation maximization, an epoch reports the negative average log-
    likelihood of the rows before its update.
    """
    case = random_sum_product_network
    iterations = 3
    circuit = case.layered(
        host=case.host,
        namespace=differentiable_backend,
        edge_matrix_type=BlockEdgeMatrix,
    )
    learned = copy.deepcopy(case.host)
    expected = ExpectationMaximization().learn(
        learned, case.rows, LearnableUnits.from_units(learned.nodes()), iterations
    )
    run = learner_of(
        layer=circuit.root, namespace=differentiable_backend
    ).minimize_until_converged(
        differentiable_backend.asarray(circuit.encoded(case.rows)),
        convergence=epochs_of(iterations),
    )

    assert run.history.epoch_losses == pytest.approx(
        [-log_likelihood for log_likelihood in expected], rel=1e-10
    )


@pytest.mark.parametrize(
    "random_sum_product_network", [VariableKinds.MIXED], indirect=True
)
def test_rows_impossible_under_the_circuit_are_left_out_as_by_the_host(
    random_sum_product_network, differentiable_backend
):
    """
    A row with a symbolic value outside the domain of its variable has probability zero;
    like the host expectation maximization, the update leaves it out of every count.
    """
    case = random_sum_product_network
    impossible = case.rows[:3].copy()
    impossible[:, 0] = 7
    rows = np.concatenate([case.rows, impossible])
    circuit = case.layered(
        host=case.host,
        namespace=differentiable_backend,
        edge_matrix_type=BlockEdgeMatrix,
    )
    run = learner_of(
        layer=circuit.root, namespace=differentiable_backend
    ).minimize_until_converged(
        differentiable_backend.asarray(circuit.encoded(rows)),
        convergence=epochs_of(1),
    )
    learned = copy.deepcopy(case.host)
    ExpectationMaximization().learn(
        learned, rows, LearnableUnits.from_units(learned.nodes()), 1
    )
    expected = case.layered(
        host=learned, namespace=np, edge_matrix_type=BlockEdgeMatrix
    )

    for learned_parameter, reference in zip(
        comparable_parameters(run.trained_root),
        comparable_parameters(expected.root),
        strict=True,
    ):
        np.testing.assert_allclose(learned_parameter, reference, rtol=1e-9, atol=1e-12)


# %% online updates


def test_an_update_after_as_many_batches_as_an_epoch_has_is_a_full_batch_update(
    random_sum_product_network, differentiable_backend
):
    """
    Updating after every three batches of an epoch of three batches learns what one
    update at the end of every epoch learns.
    """
    case = random_sum_product_network
    batch_size = 16
    batches_per_epoch = math.ceil(len(case.rows) / batch_size)

    def parameters_after_two_epochs(
        batches_per_update: int | None,
    ) -> list[np.ndarray]:
        """
        :param batches_per_update: The batches after which the parameters are updated.
        :return: The parameters after two epochs.
        """
        circuit = case.layered(
            host=case.host,
            namespace=differentiable_backend,
            edge_matrix_type=BlockEdgeMatrix,
        )
        run = learner_of(
            layer=circuit.root,
            namespace=differentiable_backend,
            batches_per_update=batches_per_update,
        ).minimize_until_converged(
            differentiable_backend.asarray(circuit.encoded(case.rows)),
            convergence=epochs_of(2),
            mini_batches=MiniBatches(batch_size=batch_size),
        )
        return comparable_parameters(run.trained_root)

    for every_epoch, full_batch in zip(
        parameters_after_two_epochs(batches_per_epoch),
        parameters_after_two_epochs(None),
        strict=True,
    ):
        np.testing.assert_allclose(every_epoch, full_batch, rtol=1e-10, atol=1e-13)


def test_a_step_size_blends_the_weights_and_moments_toward_the_full_update(
    random_sum_product_network, differentiable_backend
):
    """
    With a step size, an update moves the weights of every sum, the probabilities of
    every table, and the mean and second moment of every Gaussian that share of the way
    from their values before it to those of an update with step size one.
    """
    case = random_sum_product_network
    step_size = 0.3

    def root_after_one_update(step: float) -> Layer:
        """
        :param step: The step size.
        :return: The root after one update on all rows.
        """
        circuit = case.layered(
            host=case.host,
            namespace=differentiable_backend,
            edge_matrix_type=BlockEdgeMatrix,
        )
        return (
            learner_of(
                layer=circuit.root,
                namespace=differentiable_backend,
                batches_per_update=1,
                step_size=step,
            )
            .minimize_until_converged(
                differentiable_backend.asarray(circuit.encoded(case.rows)),
                convergence=epochs_of(1),
            )
            .trained_root
        )

    before = case.layered(
        host=case.host, namespace=np, edge_matrix_type=BlockEdgeMatrix
    ).root
    blended, full = root_after_one_update(step_size), root_after_one_update(1.0)

    for layer_before, layer_blended, layer_full in zip(
        before.all_layers(), blended.all_layers(), full.all_layers(), strict=True
    ):
        if isinstance(layer_before, SumLayer):
            np.testing.assert_allclose(
                HasParameterArrays.host_copy(layer_blended.normalized_weights),
                (1 - step_size) * layer_before.normalized_weights
                + step_size
                * HasParameterArrays.host_copy(layer_full.normalized_weights),
                rtol=1e-10,
                atol=1e-13,
            )
        if isinstance(layer_before, DiscreteLayer):
            np.testing.assert_allclose(
                HasParameterArrays.host_copy(
                    layer_blended.table.normalized_probabilities
                ),
                (1 - step_size) * layer_before.table.normalized_probabilities
                + step_size
                * HasParameterArrays.host_copy(
                    layer_full.table.normalized_probabilities
                ),
                rtol=1e-10,
                atol=1e-13,
            )
        if isinstance(layer_before, GaussianLayer):
            mean_full = HasParameterArrays.host_copy(layer_full.location)
            mean = (1 - step_size) * layer_before.location + step_size * mean_full
            second_moment = (1 - step_size) * (
                layer_before.location**2 + layer_before.scale**2
            ) + step_size * (
                mean_full**2 + HasParameterArrays.host_copy(layer_full.scale) ** 2
            )
            np.testing.assert_allclose(
                HasParameterArrays.host_copy(layer_blended.location), mean, rtol=1e-10
            )
            np.testing.assert_allclose(
                HasParameterArrays.host_copy(layer_blended.scale),
                np.sqrt(second_moment - mean**2),
                rtol=1e-10,
            )


def test_steps_on_rows_learn_what_steps_on_the_batches_of_those_rows_learn(
    random_sum_product_network, differentiable_backend
):
    """
    Stepping on rows of the training array, updating after every second step, learns
    the parameters of stepping on the batches of those rows.
    """
    case = random_sum_product_network
    circuit = case.layered(
        host=case.host,
        namespace=differentiable_backend,
        edge_matrix_type=BlockEdgeMatrix,
    )
    data = differentiable_backend.asarray(circuit.encoded(case.rows))
    row_indices = list(
        MiniBatches(batch_size=16, seed=0).row_indices_of_epoch(
            number_of_rows=data.shape[0]
        )
    )

    def trained_parameters(steps_on_rows: bool) -> list[np.ndarray]:
        """
        :param steps_on_rows: Whether every step is given rows instead of their batch.
        :return: The parameters after one step on each of the row indices.
        """
        training = learner_of(
            layer=circuit.root, namespace=differentiable_backend, batches_per_update=2
        ).start_training(arguments=(data,), mini_batches=MiniBatches())
        for rows in row_indices:
            if steps_on_rows:
                training.step_on_rows(rows=rows)
            else:
                training.step_on_batch(
                    batch=(data[differentiable_backend.asarray(rows)],)
                )
        return comparable_parameters(training.trained_root())

    for on_rows, on_batches in zip(
        trained_parameters(True), trained_parameters(False), strict=True
    ):
        np.testing.assert_allclose(on_rows, on_batches, rtol=1e-10, atol=1e-13)


# %% steps recorded as CUDA graphs


def torch_parameters_after_learning(
    *,
    case: MixtureOfProductsCase,
    edge_matrix_type: type[EdgeMatrix],
    records_cuda_graphs: bool,
) -> list[np.ndarray]:
    """
    :param case: The circuit to learn and to sample the data from.
    :param edge_matrix_type: The type of matrix every inner layer stores its edges in.
    :param records_cuda_graphs: Whether the steps may be recorded as CUDA graphs.
    :return: The parameters after four epochs of online updates with step size one half
        on mini-batches of 100 of 450 rows the circuit samples, on the GPU.
    """
    torch = pytest.importorskip("torch")
    torch_expectation_maximization = pytest.importorskip(
        "probabilistic_model.learning.parameters.expectation_maximization.torch_expectation_maximization"
    )
    np.random.seed(0)
    host = case.circuit(np, edge_matrix_type)
    namespace = array_namespace(torch.zeros(0))
    circuit = host.to_namespace(namespace, device="cuda")
    run = torch_expectation_maximization.TorchExpectationMaximization(
        layer=circuit.root,
        batches_per_update=1,
        step_size=0.5,
        records_cuda_graphs=records_cuda_graphs,
    ).minimize_until_converged(
        namespace.asarray(host.sample(450), device="cuda"),
        convergence=epochs_of(4),
        mini_batches=MiniBatches(batch_size=100, seed=0),
    )
    return [
        HasParameterArrays.host_copy(parameter)
        for parameter in run.trained_root.circuit_parameters
    ]


def test_steps_recorded_as_cuda_graphs_learn_as_steps_taken_one_by_one(
    mixture_of_products_case, edge_matrix_type
):
    """
    Recording the steps as CUDA graphs and replaying them learns the parameters that
    taking every step operation by operation learns.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    recorded, taken = (
        torch_parameters_after_learning(
            case=mixture_of_products_case,
            edge_matrix_type=edge_matrix_type,
            records_cuda_graphs=records_cuda_graphs,
        )
        for records_cuda_graphs in (True, False)
    )
    for replayed, one_by_one in zip(recorded, taken, strict=True):
        np.testing.assert_allclose(replayed, one_by_one, rtol=1e-12, atol=1e-14)


# %% batches split across devices


def split_batch_expectation_maximization(*, devices: int) -> dict:
    """
    :param devices: The number of CPU devices jax starts with.
    :return: What :mod:`split_batch_expectation_maximization` prints, run in a process
        of its own.
    """
    finished = subprocess.run(
        [
            sys.executable,
            "-m",
            "test.probabilistic_model_test.test_tensorized."
            "split_batch_expectation_maximization",
        ],
        cwd=Path(__file__).resolve().parents[3],
        env=dict(
            os.environ,
            XLA_FLAGS=f"--xla_force_host_platform_device_count={devices}",
            JAX_PLATFORMS="cpu",
            JAX_ENABLE_X64="1",
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(finished.stdout.splitlines()[-1])


def test_batches_split_across_devices_learn_as_on_one_device():
    """
    With four CPU devices, every batch whose rows four divides is split across them and
    the others run whole on every device; learning gives the parameters and losses of
    learning on one device, and the learned parameters are back on one device.
    """
    one, four = (
        split_batch_expectation_maximization(devices=devices) for devices in (1, 4)
    )

    assert (four["devices"], four["devices_of_state"]) == (4, 4)
    assert four["devices_of_trained_parameters"] == 1
    np.testing.assert_allclose(four["epoch_losses"], one["epoch_losses"], rtol=1e-12)
    for split, whole in zip(four["parameters"], one["parameters"], strict=True):
        np.testing.assert_allclose(split, whole, rtol=1e-10, atol=1e-12)


# %% unsupported layers


def test_a_circuit_with_a_layer_that_cannot_be_learned_raises():
    """
    A truncated Gaussian is not learned; a circuit with one is refused before training
    starts.
    """
    with jax.enable_x64(True):
        layer = TruncatedGaussianLayer(
            np.zeros(1, dtype=np.int64),
            jnp.array([[0.0, 1.0]]),
            np.array([[1, 1]], dtype=np.int64),
            jnp.zeros(1),
            jnp.zeros(1),
        )
        with pytest.raises(UnsupportedLayerError):
            JaxExpectationMaximization(layer=layer)
