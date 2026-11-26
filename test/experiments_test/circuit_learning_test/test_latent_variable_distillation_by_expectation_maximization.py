"""
Tests of the learning method of the circuit learning benchmark that distills the latent
states of a hidden Chow-Liu tree circuit and fine-tunes it by expectation maximization.
"""

from __future__ import annotations

# %% imports
import jax
import numpy as np
import pytest
import torch
from krrood_compute_resources.devices import Device

from experiments.circuit_learning.methods import (
    BenchmarkMethods,
    HiddenChowLiuTreeByExpectationMaximization,
    LatentVariableDistillationByExpectationMaximization,
    LearningMethod,
    TrainingContext,
)
from experiments.circuit_learning.settings import (
    ArrayLibrary,
    ComputationSetting,
    EdgeRepresentation,
    FloatingPointType,
)
from probabilistic_model.learning.parameters.training_schedule import Convergence
from probabilistic_model.learning.structure.latent_variable_distillation.latent_variable_distillation import (
    MaskedAutoencoderTraining,
)


@pytest.fixture
def short_training() -> TrainingContext:
    """
    Two epochs of every phase.
    """
    return TrainingContext(
        convergence=Convergence(maximum_epochs=2, patience=2),
        memory_budget_bytes=2**30,
    )


@pytest.fixture
def small_method() -> LatentVariableDistillationByExpectationMaximization:
    """
    A distillation into four hidden states by a small autoencoder trained briefly.
    """
    return LatentVariableDistillationByExpectationMaximization(
        hidden_states=4,
        autoencoder=MaskedAutoencoderTraining(
            width=16, layers=1, heads=2, feedforward_width=32, epochs=2
        ),
    )


def central_processor_setting(*, array_library: ArrayLibrary) -> ComputationSetting:
    """
    :param array_library: The array library that trains.
    :return: The setting of that library on the central processor in double precision.
    """
    return ComputationSetting(
        array_library=array_library,
        device=Device.CENTRAL_PROCESSOR,
        floating_point_type=FloatingPointType.FLOAT64,
        edge_representation=EdgeRepresentation.BLOCK,
    )


# %% registration


def test_the_benchmark_lists_latent_variable_distillation():
    """
    The benchmark finds the method by its name.
    """
    method = BenchmarkMethods().find(name=LearningMethod.LATENT_VARIABLE_DISTILLATION)

    assert isinstance(method, LatentVariableDistillationByExpectationMaximization)


def test_distilled_circuits_fine_tune_in_the_settings_of_hidden_chow_liu_trees():
    """
    The fine-tuning trains in every setting the hidden Chow-Liu tree method trains in.
    """
    assert (
        LatentVariableDistillationByExpectationMaximization().settings()
        == HiddenChowLiuTreeByExpectationMaximization().settings()
    )


# %% learning


@pytest.mark.parametrize(
    "array_library", [ArrayLibrary.JAX, ArrayLibrary.TORCH], ids=lambda library: library
)
def test_fine_tuning_raises_the_training_log_likelihood_of_the_distilled_circuit(
    array_library: ArrayLibrary, dataset, short_training, small_method
):
    """
    Each phase fine-tunes as long as the context says, and the fine-tuned circuit gives
    the training rows a higher log-likelihood than the distilled circuit it started
    from.
    """
    setting = central_processor_setting(array_library=array_library)
    rows = dataset.encoded(frame=dataset.train)
    distilled = small_method.initial_circuit(dataset=dataset, setting=setting)

    with jax.enable_x64(True):
        learned = small_method.learn(
            dataset=dataset, setting=setting, context=short_training
        )
        log_likelihoods = np.asarray(
            learned.circuit.log_likelihood(events=setting.asarray(rows=rows))
        )

    assert learned.history.epochs == (
        short_training.convergence.maximum_epochs * len(small_method.phases)
    )
    assert log_likelihoods.mean() > distilled.log_likelihood(events=rows).mean()


def test_the_autoencoder_trains_on_the_device_of_the_setting(dataset, small_method):
    """
    The distillation runs on the torch device of the setting's device.
    """
    setting = central_processor_setting(array_library=ArrayLibrary.JAX)

    distillation = small_method.distillation(dataset=dataset, setting=setting)

    assert distillation.device == setting.torch_device
