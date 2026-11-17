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

from probabilistic_model.learning.exceptions import UnscheduledOptimizerError
from probabilistic_model.learning.gradient_descent import (
    AverageNegativeLogLikelihood,
    JaxGradientDescent,
)
from probabilistic_model.learning.training_schedule import (
    Convergence,
    LearningRatePlateau,
    MiniBatches,
    MiniBatchTraining,
    PlateauScheduledTraining,
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


def test_an_elapsed_time_limit_starts_no_further_epoch(convergence):
    """
    Once the time limit has passed, no epoch starts and training ends without having
    converged.
    """
    convergence.maximum_seconds = 0.0
    history = convergence.history_until_converged(
        training=scripted_training([1.0] * convergence.maximum_epochs)
    )

    assert not history.converged
    assert history.epochs == 0


# %% learning rate plateau


def test_a_plateau_reduces_the_learning_rate_after_the_patience():
    """
    The learning rate is reduced once more epochs than the patience brought no relative
    improvement, and not before.
    """
    plateau = LearningRatePlateau(factor=0.5, patience=2, relative_tolerance=0.01)
    for loss in [10.0, 9.0, 8.95, 8.95]:
        plateau.record(average_loss=loss)
    assert plateau.scale == 1.0
    plateau.record(average_loss=8.95)
    assert plateau.scale == plateau.factor


def test_a_plateau_settles_after_its_reductions():
    """
    A constant loss reduces the learning rate every patience plus one epochs until the
    plateau has settled.
    """
    plateau = LearningRatePlateau(patience=1, reductions_until_settled=2)
    for _ in range(1 + 2 * (plateau.patience + 1)):
        plateau.record(average_loss=1.0)
    assert plateau.is_settled
    assert plateau.reductions == plateau.reductions_until_settled


@dataclass(kw_only=True)
class ScriptedPlateauTraining(PlateauScheduledTraining, ScriptedLossTraining):
    """
    Scripted training under a learning rate schedule that records every scale it is
    given.
    """

    scales: list[float]
    """
    The scale of the learning rate after every epoch, in order.
    """

    def scale_learning_rate(self, *, scale: float) -> None:
        self.scales.append(scale)


def test_training_under_a_plateau_converges_only_once_the_plateau_settled(
    convergence,
):
    """
    A constant loss converges by the criterion after the patience, but training goes on
    until the plateau has reduced the learning rate often enough.
    """
    plateau = LearningRatePlateau(patience=1, reductions_until_settled=2)
    training = ScriptedPlateauTraining(
        arguments=(np.zeros(1),),
        mini_batches=MiniBatches(),
        losses=[1.0] * convergence.maximum_epochs,
        scales=[],
        learning_rate_plateau=plateau,
    )
    history = convergence.history_until_converged(training=training)

    assert history.converged
    assert history.epochs == 1 + plateau.reductions_until_settled * (
        plateau.patience + 1
    )
    assert training.scales[-1] == plateau.factor**plateau.reductions_until_settled


@pytest.fixture
def jax_circuit(mixture_of_products_case):
    """
    The mixture of products of :func:`mixture_of_products_case` with jax parameters.
    """
    return mixture_of_products_case.circuit(np).to_namespace(jnp)


def test_a_plateau_needs_an_optimizer_whose_learning_rate_can_change(jax_circuit):
    """
    A schedule for an optax optimizer with a fixed learning rate is rejected.
    """
    trainer = JaxGradientDescent(
        jax_circuit.root, AverageNegativeLogLikelihood(), optax.adam(1e-2)
    )
    with pytest.raises(UnscheduledOptimizerError):
        trainer.start_training(
            arguments=(jnp.zeros((1, len(jax_circuit.variables))),),
            mini_batches=MiniBatches(),
            learning_rate_plateau=LearningRatePlateau(),
        )


def test_jax_training_scales_the_learning_rate_of_its_optimizer(jax_circuit):
    """
    Scaling the learning rate multiplies the initial learning rate of the optimizer.
    """
    learning_rate = 0.1
    training = JaxGradientDescent(
        jax_circuit.root,
        AverageNegativeLogLikelihood(),
        optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
    ).start_training(
        arguments=(jnp.zeros((1, len(jax_circuit.variables))),),
        mini_batches=MiniBatches(),
        learning_rate_plateau=LearningRatePlateau(),
    )
    training.scale_learning_rate(scale=0.5)

    assert float(
        training.state.optimizer_state.hyperparams["learning_rate"]
    ) == pytest.approx(0.5 * learning_rate)


def test_torch_training_scales_the_learning_rate_of_its_optimizer(
    mixture_of_products_case,
):
    """
    Scaling the learning rate multiplies the learning rate of every parameter group.
    """
    torch = pytest.importorskip("torch")
    torch_gradient_descent = pytest.importorskip(
        "probabilistic_model.learning.torch_gradient_descent"
    )
    circuit = mixture_of_products_case.circuit(np).to_namespace(
        array_namespace(torch.zeros(0))
    )
    trainer = torch_gradient_descent.TorchGradientDescent(
        circuit.root, AverageNegativeLogLikelihood(), learning_rate=0.1
    )
    training = trainer.start_training(
        arguments=(torch.zeros((1, len(circuit.variables))),),
        mini_batches=MiniBatches(),
        learning_rate_plateau=LearningRatePlateau(),
    )
    training.scale_learning_rate(scale=0.5)

    assert [group["lr"] for group in training.optimizer.param_groups] == [
        pytest.approx(0.5 * trainer.learning_rate)
    ] * len(training.optimizer.param_groups)


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
