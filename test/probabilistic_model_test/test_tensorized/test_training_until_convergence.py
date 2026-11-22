"""
Tests for training a layered circuit on mini-batches until its loss converges: when the
convergence criterion stops, how the rows are split into mini-batches, and that both
trainers raise the log-likelihood of their data that way.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from array_api_compat import array_namespace

from probabilistic_model.learning.exceptions import UnscheduledOptimizerError
from probabilistic_model.learning.gradient_descent import AverageNegativeLogLikelihood
from probabilistic_model.learning.jax_gradient_descent import JaxGradientDescent
from probabilistic_model.learning.training_schedule import (
    Convergence,
    ConvergenceHistory,
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
# %% convergence criterion


@dataclass(kw_only=True)
class ScriptedLossTraining(MiniBatchTraining):
    """
    Training whose every step reports the next loss of a fixed script, one batch per
    epoch.
    """

    losses: list[float | np.ndarray]
    """
    The loss each step reports, in order, a float or a scalar array.
    """

    def step_on_batch(self, *, batch: tuple[np.ndarray, ...]) -> float | np.ndarray:
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
    convergence.maximum_duration = timedelta()
    history = convergence.history_until_converged(
        training=scripted_training([1.0] * convergence.maximum_epochs)
    )

    assert not history.converged
    assert history.epochs == 0


def test_every_epoch_records_its_duration(convergence):
    """
    Training records one duration per epoch, which together take no longer than the
    whole training.
    """
    history = convergence.history_until_converged(
        training=scripted_training([1.0] * convergence.maximum_epochs)
    )

    assert len(history.epoch_durations) == history.epochs
    assert sum(history.epoch_durations, timedelta()) <= history.duration


def test_the_average_epoch_duration_leaves_out_the_first_epoch():
    """
    The average duration of an epoch is that of the epochs after the first, which also
    compiles the training step.
    """
    history = ConvergenceHistory(
        epoch_losses=[3.0, 2.0, 1.0],
        converged=True,
        epoch_durations=[
            timedelta(seconds=10),
            timedelta(seconds=1),
            timedelta(seconds=3),
        ],
    )

    assert history.average_epoch_duration == timedelta(seconds=2)


def test_a_single_epoch_is_its_own_average_duration():
    """
    With a single epoch, the average duration of an epoch is its duration.
    """
    history = ConvergenceHistory(
        epoch_losses=[1.0], converged=False, epoch_durations=[timedelta(seconds=4)]
    )

    assert history.average_epoch_duration == timedelta(seconds=4)


def test_no_epoch_has_no_average_duration():
    """
    A training that ran no epoch has no average epoch duration.
    """
    assert (
        ConvergenceHistory(epoch_losses=[], converged=False).average_epoch_duration
        is None
    )


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
        layer=jax_circuit.root,
        loss=AverageNegativeLogLikelihood(),
        optimizer=optax.adam(1e-2),
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
        layer=jax_circuit.root,
        loss=AverageNegativeLogLikelihood(),
        optimizer=optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
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
        layer=circuit.root, loss=AverageNegativeLogLikelihood(), learning_rate=0.1
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


# %% losses of an epoch


def test_an_epoch_averages_the_losses_of_its_batches_over_their_rows():
    """
    The loss of an epoch is the average of the scalar array losses of its batches, each
    weighted by its number of rows, as a float.
    """
    batch_losses = [np.asarray(1.0), np.asarray(4.0), np.asarray(10.0)]
    training = ScriptedLossTraining(
        arguments=(np.zeros(5),),
        mini_batches=MiniBatches(batch_size=2),
        losses=list(batch_losses),
    )

    average = training.train_one_epoch()

    assert type(average) is float
    assert average == pytest.approx(
        float(2 * batch_losses[0] + 2 * batch_losses[1] + batch_losses[2]) / 5
    )


def test_a_jax_step_reports_its_loss_as_a_jax_array(jax_circuit):
    """
    A step of jax training reports the loss of its batch as a jax array.
    """
    training = JaxGradientDescent(
        layer=jax_circuit.root,
        loss=AverageNegativeLogLikelihood(),
        optimizer=optax.adam(1e-2),
    ).start_training(
        arguments=(jnp.zeros((1, len(jax_circuit.variables))),),
        mini_batches=MiniBatches(),
    )

    assert isinstance(training.step_on_batch(batch=training.arguments), jax.Array)


def test_a_torch_step_reports_its_loss_as_a_tensor(mixture_of_products_case):
    """
    A step of torch training reports the loss of its batch as a tensor without a
    gradient.
    """
    torch = pytest.importorskip("torch")
    torch_gradient_descent = pytest.importorskip(
        "probabilistic_model.learning.torch_gradient_descent"
    )
    circuit = mixture_of_products_case.circuit(np).to_namespace(
        array_namespace(torch.zeros(0))
    )
    training = torch_gradient_descent.TorchGradientDescent(
        layer=circuit.root, loss=AverageNegativeLogLikelihood(), learning_rate=0.1
    ).start_training(
        arguments=(torch.zeros((1, len(circuit.variables)), dtype=torch.float64),),
        mini_batches=MiniBatches(),
    )

    loss = training.step_on_batch(batch=training.arguments)

    assert isinstance(loss, torch.Tensor)
    assert not loss.requires_grad


# %% steps recorded as CUDA graphs


def torch_parameters_after_training(
    *,
    case,
    edge_matrix_type: type,
    device: str,
    records_cuda_graphs: bool,
    takes_steps_one_by_one: bool,
) -> list[np.ndarray]:
    """
    :param case: The circuit to train and to sample the data from.
    :param edge_matrix_type: The type of matrix every inner layer stores its edges in.
    :param device: The torch device to train on.
    :param records_cuda_graphs: Whether the steps may be recorded as CUDA graphs.
    :param takes_steps_one_by_one: Whether every step is taken by
        :meth:`TorchGradientDescent.step` with the optimizer of the training, instead
        of by the training itself.
    :return: The parameters after four epochs of mini-batches of 100 rows of data the
        circuit samples, the learning rate scaled by a tenth before the last epoch.
    """
    torch = pytest.importorskip("torch")
    torch_gradient_descent = pytest.importorskip(
        "probabilistic_model.learning.torch_gradient_descent"
    )
    np.random.seed(0)
    host = case.circuit(np, edge_matrix_type)
    namespace = array_namespace(torch.zeros(0))
    circuit = host.to_namespace(namespace, device=device)
    data = namespace.asarray(host.sample(450), device=device)
    trainer = torch_gradient_descent.TorchGradientDescent(
        layer=circuit.root,
        loss=AverageNegativeLogLikelihood(),
        learning_rate=0.05,
        records_cuda_graphs=records_cuda_graphs,
    )
    training = trainer.start_training(
        arguments=(data,),
        mini_batches=MiniBatches(batch_size=100, seed=0),
        learning_rate_plateau=LearningRatePlateau(),
    )
    for epoch in range(4):
        if epoch == 3:
            training.scale_learning_rate(scale=0.1)
        for batch in training.mini_batches.batches_of_epoch(arguments=(data,)):
            if takes_steps_one_by_one:
                trainer.step(optimizer=training.optimizer, arguments=batch)
            else:
                training.step_on_batch(batch=batch)
    return [parameter.detach().cpu().numpy() for parameter in trainer.parameters]


def test_batches_of_the_same_array_shapes_share_their_recorded_step():
    """
    Batches whose arrays have the same shapes have one batch shape, which tells them
    apart from batches of other shapes.
    """
    torch = pytest.importorskip("torch")
    torch_circuit_parameter_learning = pytest.importorskip(
        "probabilistic_model.learning.torch_circuit_parameter_learning"
    )
    batch_shape = torch_circuit_parameter_learning.BatchShape.from_batch

    first = batch_shape(batch=(torch.zeros(100, 3),))

    assert first == batch_shape(batch=(torch.ones(100, 3),))
    assert hash(first) == hash(batch_shape(batch=(torch.ones(100, 3),)))
    assert first != batch_shape(batch=(torch.zeros(50, 3),))


def test_steps_recorded_as_cuda_graphs_train_as_steps_taken_one_by_one(
    mixture_of_products_case, edge_matrix_type
):
    """
    Recording the steps as CUDA graphs and replaying them gives the parameters that
    taking every step on its own with the same optimizer gives, also after the learning
    rate is scaled.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    recorded, taken = (
        torch_parameters_after_training(
            case=mixture_of_products_case,
            edge_matrix_type=edge_matrix_type,
            device="cuda",
            records_cuda_graphs=True,
            takes_steps_one_by_one=one_by_one,
        )
        for one_by_one in (False, True)
    )
    for recorded_parameter, taken_parameter in zip(recorded, taken):
        np.testing.assert_array_equal(recorded_parameter, taken_parameter)


def test_steps_on_the_cpu_are_taken_one_by_one_even_if_cuda_graphs_are_asked_for(
    mixture_of_products_case, edge_matrix_type
):
    """
    Parameters on the CPU are trained step by step with the same optimizer, whether CUDA
    graphs are asked for or not.
    """
    asked, not_asked = (
        torch_parameters_after_training(
            case=mixture_of_products_case,
            edge_matrix_type=edge_matrix_type,
            device="cpu",
            records_cuda_graphs=records,
            takes_steps_one_by_one=False,
        )
        for records in (True, False)
    )
    for asked_parameter, not_asked_parameter in zip(asked, not_asked):
        np.testing.assert_array_equal(asked_parameter, not_asked_parameter)


# %% batches split across devices


def split_batch_training(*, devices: int) -> dict:
    """
    :param devices: The number of CPU devices jax starts with.
    :return: What :mod:`split_batch_training` prints, run in a process of its own.
    """
    repository = Path(__file__).resolve().parents[3]
    environment = dict(
        os.environ,
        XLA_FLAGS=f"--xla_force_host_platform_device_count={devices}",
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="1",
    )
    finished = subprocess.run(
        [
            sys.executable,
            "-m",
            "test.probabilistic_model_test.test_tensorized.split_batch_training",
        ],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(finished.stdout.splitlines()[-1])


def test_batches_split_across_devices_train_as_on_one_device():
    """
    With four CPU devices, every batch whose rows four divides is split across them and
    the others run whole on every device; training gives the parameters and losses of
    training on one device, and the trained parameters are back on one device.
    """
    one, four = (split_batch_training(devices=devices) for devices in (1, 4))

    assert (four["devices"], four["devices_of_state"]) == (4, 4)
    assert four["devices_of_trained_parameters"] == 1
    np.testing.assert_allclose(four["epoch_losses"], one["epoch_losses"], rtol=1e-12)
    for split, whole in zip(four["parameters"], one["parameters"]):
        np.testing.assert_allclose(split, whole, rtol=1e-10, atol=1e-12)


def test_the_largest_even_batch_size_is_a_multiple_of_the_devices():
    """
    The largest batch size of at most 450 rows that four devices split evenly is 448,
    and one device splits every batch size evenly.
    """
    assert split_batch_training(devices=4)["even_batch_size"] == 448
    assert split_batch_training(devices=1)["even_batch_size"] == 450


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
            parameters=parameters_shifted(start, GaussianLayer, 1.5)
        ).to_namespace(jnp)
        events = jnp.asarray(data)
        before = float(jnp.mean(circuit.log_likelihood(events=events)))
        run = JaxGradientDescent(
            layer=circuit.root,
            loss=AverageNegativeLogLikelihood(),
            optimizer=optax.adam(5e-2),
        ).minimize_until_converged(
            events,
            convergence=training_convergence,
            mini_batches=MiniBatches(batch_size=100),
        )
        after = float(
            jnp.mean(run.trained_root.log_likelihood_of_nodes(events=events)[:, 0])
        )
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
        parameters=parameters_shifted(start, GaussianLayer, 1.5)
    ).to_namespace(namespace)
    events = namespace.asarray(data)
    before = float(circuit.log_likelihood(events=events).mean())
    run = torch_gradient_descent.TorchGradientDescent(
        layer=circuit.root, loss=AverageNegativeLogLikelihood(), learning_rate=5e-2
    ).minimize_until_converged(
        events,
        convergence=training_convergence,
        mini_batches=MiniBatches(batch_size=100),
    )
    after = float(run.trained_root.log_likelihood_of_nodes(events=events)[:, 0].mean())
    assert after > before + minimum_improvement
