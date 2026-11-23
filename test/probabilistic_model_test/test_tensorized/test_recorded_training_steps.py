"""
Tests that steps recorded as CUDA graphs leave the recording of other steps intact: a
finished training is freed without the garbage collector, and a collection that frees a
recorded graph while another step is recorded does not invalidate that recording.
"""

from __future__ import annotations

# %% imports
import gc
import weakref
from collections.abc import Iterator

import numpy as np
import pytest
from array_api_compat import array_namespace

from probabilistic_model.learning.parameters.training_schedule import MiniBatches
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.edge_matrix import (
    BlockEdgeMatrix,
)

torch = pytest.importorskip("torch")
torch_circuit_parameter_learning = pytest.importorskip(
    "probabilistic_model.learning.parameters.torch_circuit_parameter_learning"
)
torch_expectation_maximization = pytest.importorskip(
    "probabilistic_model.learning.parameters.expectation_maximization.torch_expectation_maximization"
)

# %% fixtures


@pytest.fixture
def cuda_device() -> str:
    """
    :return: The CUDA device the test records its steps on; the test is skipped without
        one.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    return "cuda"


@pytest.fixture
def garbage_collection_after_every_allocation() -> Iterator[None]:
    """
    Run the garbage collector over every generation after every allocation of a tracked
    object during the test, and restore its thresholds and whether it runs afterwards.
    """
    thresholds, was_enabled = gc.get_threshold(), gc.isenabled()
    gc.set_threshold(1, 1, 1)
    gc.enable()
    yield
    gc.set_threshold(*thresholds)
    if not was_enabled:
        gc.disable()


@pytest.fixture
def garbage_collection_disabled() -> Iterator[None]:
    """
    Keep the garbage collector from running during the test, and let it run afterwards
    if it did before.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    yield
    if was_enabled:
        gc.enable()


# %% freeing a finished training


def test_a_training_that_records_its_steps_is_freed_without_the_garbage_collector(
    mixture_of_products_case, cuda_device, garbage_collection_disabled
):
    """
    A training whose recorded steps are replayed is freed as soon as it is no longer
    referenced, so that its recorded graphs never wait for a collection.
    """
    np.random.seed(0)
    host = mixture_of_products_case.circuit(np, BlockEdgeMatrix)
    namespace = array_namespace(torch.zeros(0))
    circuit = host.to_namespace(namespace, device=cuda_device)
    training = torch_expectation_maximization.TorchExpectationMaximization(
        layer=circuit.root, batches_per_update=1, records_cuda_graphs=True
    ).start_training(
        arguments=(namespace.asarray(host.sample(400), device=cuda_device),),
        mini_batches=MiniBatches(batch_size=100, seed=0),
    )
    for _ in range(2):
        training.train_one_epoch()
    assert len(training.recorded_steps) == 1
    reference = weakref.ref(training)
    del training
    assert reference() is None


# %% collecting while recording


def test_recording_a_step_survives_a_collection_that_frees_another_recorded_step(
    cuda_device, garbage_collection_after_every_allocation
):
    """
    A collection during the recording of a step that frees another recorded step, kept
    alive only by a reference cycle, leaves the recording intact.
    """
    batch = (torch.ones(1024, device=cuda_device),)

    def doubled_sum(*, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        :return: The sum of the doubled first array of the batch.
        """
        return (batch[0] * 2).sum()

    recorded_before, recording = (
        torch_circuit_parameter_learning.RecordedTrainingStep() for _ in range(2)
    )
    for recorded_step in (recorded_before, recording):
        recorded_step.take_on_side_stream(step=doubled_sum, batch=batch)
    recorded_before.record(step=doubled_sum, batch=batch)
    referenced_until_recording = [recorded_before]
    del recorded_before

    def drop_into_cycle_and_allocate(
        *, batch: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """
        Move the last reference to the step recorded before into a new reference cycle,
        then allocate enough tracked objects for a collection of the youngest objects.
        """
        garbage_in_cycle = [referenced_until_recording.pop()]
        garbage_in_cycle.append(garbage_in_cycle)
        del garbage_in_cycle
        [[] for _ in range(100)]
        return doubled_sum(batch=batch)

    recording.record(step=drop_into_cycle_and_allocate, batch=batch)
    recording.graph.replay()
    assert recording.loss.item() == 2 * batch[0].shape[0]
