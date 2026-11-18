"""
Tests for training a layered circuit on mini-batches until its loss converges: when the
convergence criterion stops, how the rows are split into mini-batches, and that both
trainers raise the log-likelihood of their data that way.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from array_api_compat import array_namespace

from probabilistic_model.learning.gradient_descent import (
    AverageNegativeLogLikelihood,
    JaxGradientDescent,
)
from probabilistic_model.learning.training_schedule import (
    Convergence,
    MiniBatches,
    MiniBatchTraining,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.gaussian_layer import (
    GaussianLayer,
)

from .test_circuit_parameters_and_training import (
    minimum_improvement,
    parameters_shifted,
)
from .test_inner_layers_on_array_backends import mixture_of_products_case

# %% convergence criterion


@dataclass(kw_only=True)
class ScriptedLossTraining(MiniBatchTraining):
    """
    Training whose every step reports the next loss of a fixed script, one batch per
    epoch.
    """

    losses: list[float]
    """
    The loss each step reports, in order.
    """

    def step_on_batch(self, *, batch: tuple[np.ndarray, ...]) -> float:
        return self.losses.pop(0)


def scripted_training(losses: list[float]) -> ScriptedLossTraining:
    """
    :param losses: The loss of every epoch, in order.
    :return: A training whose epochs report these losses.
    """
    return ScriptedLossTraining(
        arguments=(np.zeros(1),),
        mini_batches=MiniBatches(),
        losses=losses,
    )


@pytest.fixture
def convergence() -> Convergence:
    """
    A criterion that stops after two epochs that improve by less than 0.1, at most ten
    epochs.
    """
    return Convergence(tolerance=0.1, patience=2, maximum_epochs=10)


def test_small_improvements_for_the_patience_converge(convergence):
    """
    Losses whose last improvements are all below the tolerance, as many as the patience,
    have converged.
    """
    assert convergence.has_converged(epoch_losses=[5.0, 2.0, 1.95, 1.9])


def test_one_large_improvement_within_the_patience_has_not_converged(convergence):
    """
    A single improvement of at least the tolerance among the last ones keeps training.
    """
    assert not convergence.has_converged(epoch_losses=[5.0, 2.0, 1.5, 1.45])


def test_fewer_epochs_than_the_patience_have_not_converged(convergence):
    """
    Training never stops before it has seen as many improvements as the patience.
    """
    assert not convergence.has_converged(epoch_losses=[1.0, 1.0])


def test_a_loss_that_stops_improving_ends_training_once_converged(convergence):
    """
    A constant loss converges after the first epoch plus the patience.
    """
    history = convergence.history_until_converged(
        training=scripted_training([1.0] * convergence.maximum_epochs)
    )

    assert history.converged
    assert history.epochs == convergence.patience + 1


def test_a_loss_that_keeps_improving_ends_training_at_the_epoch_limit(convergence):
    """
    A loss that improves by more than the tolerance every epoch stops at the epoch limit
    without converging.
    """
    history = convergence.history_until_converged(
        training=scripted_training([float(loss) for loss in range(100, 0, -1)])
    )

    assert not history.converged
    assert history.epochs == convergence.maximum_epochs


# %% mini-batches


def test_the_batches_of_an_epoch_hold_every_row_once():
    """
    The mini-batches of one epoch together hold every row exactly once, in batches of
    the batch size and one smaller rest.
    """
    rows = np.arange(10.0)

    batches = list(
        MiniBatches(batch_size=4, seed=0).batches_of_epoch(arguments=(rows,))
    )

    assert [len(batch) for (batch,) in batches] == [4, 4, 2]
    assert sorted(np.concatenate([batch for (batch,) in batches])) == list(rows)


def test_the_batches_keep_the_rows_of_all_arguments_together():
    """
    Every argument is split by the same rows, so a row of data stays with its label.
    """
    data = np.arange(10.0)
    labels = np.arange(10.0) * 2

    for batch_data, batch_labels in MiniBatches(batch_size=3, seed=0).batches_of_epoch(
        arguments=(data, labels)
    ):
        np.testing.assert_array_equal(batch_labels, batch_data * 2)


def test_without_a_batch_size_an_epoch_is_one_batch_of_every_row():
    """
    No batch size means full-batch training.
    """
    rows = np.arange(10.0)

    batches = list(
        MiniBatches(batch_size=None, seed=0).batches_of_epoch(arguments=(rows,))
    )

    assert len(batches) == 1
    assert sorted(batches[0][0]) == list(rows)


# %% training until convergence


@pytest.fixture
def training_convergence() -> Convergence:
    """
    The criterion the trainers are run with: tight enough to train every one of its few
    epochs, so that the tests stay fast.
    """
    return Convergence(tolerance=1e-4, patience=3, maximum_epochs=20)


def test_training_a_jax_circuit_on_mini_batches_raises_the_log_likelihood(
    mixture_of_products_case, minimum_improvement, training_convergence
):
    """
    Maximizing the likelihood on mini-batches until convergence, starting from a circuit
    with moved locations, raises the average log-likelihood of the data.
    """
    case = mixture_of_products_case
    np.random.seed(0)
    data = case.circuit(np).sample(500)
    start = case.circuit(np)
    with jax.enable_x64(True):
        circuit = start.copy_with_parameters(
            parameters_shifted(start, GaussianLayer, 1.5)
        ).to_namespace(jnp)
        events = jnp.asarray(data)
        before = float(jnp.mean(circuit.log_likelihood(events)))
        run = JaxGradientDescent(
            circuit.root, AverageNegativeLogLikelihood(), optax.adam(5e-2)
        ).minimize_until_converged(
            events,
            convergence=training_convergence,
            mini_batches=MiniBatches(batch_size=100),
        )
        after = float(jnp.mean(run.trained_root.log_likelihood_of_nodes(events)[:, 0]))
    assert after > before + minimum_improvement


def test_training_a_torch_circuit_on_mini_batches_raises_the_log_likelihood(
    mixture_of_products_case, minimum_improvement, training_convergence
):
    """
    Maximizing the likelihood on mini-batches until convergence with torch raises the
    average log-likelihood of the data.
    """
    torch = pytest.importorskip("torch")
    torch_gradient_descent = pytest.importorskip(
        "probabilistic_model.learning.torch_gradient_descent"
    )

    case = mixture_of_products_case
    np.random.seed(0)
    data = case.circuit(np).sample(500)
    start = case.circuit(np)
    namespace = array_namespace(torch.zeros(0))
    circuit = start.copy_with_parameters(
        parameters_shifted(start, GaussianLayer, 1.5)
    ).to_namespace(namespace)
    events = namespace.asarray(data)
    before = float(circuit.log_likelihood(events).mean())
    run = torch_gradient_descent.TorchGradientDescent(
        circuit.root, AverageNegativeLogLikelihood(), learning_rate=5e-2
    ).minimize_until_converged(
        events,
        convergence=training_convergence,
        mini_batches=MiniBatches(batch_size=100),
    )
    after = float(run.trained_root.log_likelihood_of_nodes(events)[:, 0].mean())
    assert after > before + minimum_improvement
