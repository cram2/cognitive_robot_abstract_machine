"""
Tests of the learning method of the circuit learning benchmark that trains hidden Chow-
Liu tree circuits by expectation maximization.
"""

from __future__ import annotations

# %% imports
import jax
import numpy as np
import pytest
from krrood_compute_resources.devices import Device

from experiments.circuit_learning.methods import (
    BenchmarkMethods,
    HiddenChowLiuTreeByExpectationMaximization,
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
from probabilistic_model.learning.structure.exceptions import (
    UnsupportedMutualInformationVariableError,
)
from probabilistic_model.learning.structure.mutual_information import (
    GaussianMutualInformation,
)
from probabilistic_model.learning.structure.hidden_chow_liu_tree.hidden_chow_liu_tree import (
    HiddenChowLiuTree,
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


def test_the_benchmark_lists_the_hidden_chow_liu_tree():
    """
    The benchmark finds the method by its name.
    """
    method = BenchmarkMethods().find(name=LearningMethod.HIDDEN_CHOW_LIU_TREE)

    assert isinstance(method, HiddenChowLiuTreeByExpectationMaximization)


def test_hidden_chow_liu_tree_learns_in_every_jax_and_torch_setting():
    """
    The tensorized expectation maximization trains with jax and torch.
    """
    assert HiddenChowLiuTreeByExpectationMaximization().settings() == (
        ComputationSetting.grid(array_libraries=(ArrayLibrary.JAX, ArrayLibrary.TORCH))
    )


# %% learning


@pytest.mark.parametrize(
    "array_library", [ArrayLibrary.JAX, ArrayLibrary.TORCH], ids=lambda library: library
)
def test_every_phase_trains_and_raises_the_training_log_likelihood(
    array_library: ArrayLibrary, dataset, short_training
):
    """
    Each phase trains as long as the context says, and the trained circuit gives the
    training rows a higher log-likelihood than the circuit it started from.
    """
    method = HiddenChowLiuTreeByExpectationMaximization(hidden_states=4)
    setting = central_processor_setting(array_library=array_library)
    rows = dataset.encoded(frame=dataset.train)
    untrained = HiddenChowLiuTree(
        variables=dataset.variables,
        hidden_states=method.hidden_states,
        seed=method.seed,
    ).build_circuit(data=rows)

    with jax.enable_x64(True):
        learned = method.learn(dataset=dataset, setting=setting, context=short_training)
        log_likelihoods = np.asarray(
            learned.circuit.log_likelihood(events=setting.asarray(rows=rows))
        )

    assert learned.history.epochs == (
        short_training.convergence.maximum_epochs * len(method.phases)
    )
    assert log_likelihoods.mean() > untrained.log_likelihood(events=rows).mean()


def test_the_tree_is_spanned_over_the_mutual_information_of_the_method(
    dataset, short_training
):
    """
    The method estimates the mutual information with its own estimator, here one that
    refuses the symbolic column of the dataset.
    """
    method = HiddenChowLiuTreeByExpectationMaximization(
        mutual_information=GaussianMutualInformation()
    )

    with pytest.raises(UnsupportedMutualInformationVariableError):
        method.learn(
            dataset=dataset,
            setting=central_processor_setting(array_library=ArrayLibrary.JAX),
            context=short_training,
        )
