"""
Tests for sizing mini-batches to the free memory of a device: the memory one training
step allocates, and the batch size that fits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from array_api_compat import array_namespace

from krrood_compute_resources.memory import HostMemory

from probabilistic_model.learning.parameters.step_memory import (
    ProbeMeasurement,
    StepMemory,
)
from probabilistic_model.learning.parameters.exceptions import (
    NotEnoughMemoryError,
    UnmeasurableHostStepMemoryError,
    UnmeasurableStepMemoryError,
)
from probabilistic_model.learning.parameters.gradient_descent.gradient_descent import (
    AverageNegativeLogLikelihood,
)
from probabilistic_model.learning.parameters.gradient_descent.jax_gradient_descent import (
    JaxGradientDescent,
)
from probabilistic_model.learning.parameters.training_schedule import MiniBatches

from ...krrood_compute_resources_test.dataset.memory import FixedFreeMemory
# %% memory of a step


def test_two_probes_give_the_line_through_both():
    """
    The fixed part and the part per row are those of the line through the two
    measurements.
    """
    step_memory = StepMemory.from_two_probes(
        smaller=ProbeMeasurement(rows=10, step_bytes=300),
        larger=ProbeMeasurement(rows=20, step_bytes=500),
    )

    assert step_memory == StepMemory(fixed_bytes=100, bytes_per_row=20)


def test_probes_without_more_memory_for_more_rows_measure_nothing():
    """
    A larger batch that needed no more memory than the smaller one gives no memory per
    row.
    """
    with pytest.raises(UnmeasurableStepMemoryError):
        StepMemory.from_two_probes(
            smaller=ProbeMeasurement(rows=10, step_bytes=300),
            larger=ProbeMeasurement(rows=20, step_bytes=300),
        )


# %% batch size from free memory


def test_the_batches_fitting_in_memory_have_the_most_rows_that_fit():
    """
    The batch size is the number of rows whose step, fixed part included, fits in the
    free memory.
    """
    mini_batches = MiniBatches.fitting_in_memory(
        memory=FixedFreeMemory(free=1000),
        step_memory=StepMemory(fixed_bytes=200, bytes_per_row=10),
    )

    assert mini_batches.batch_size == 80


def test_no_batch_fits_when_the_fixed_part_exceeds_the_free_memory():
    """
    A step whose fixed part alone exceeds the free memory fits no batch.
    """
    with pytest.raises(NotEnoughMemoryError):
        MiniBatches.fitting_in_memory(
            memory=FixedFreeMemory(free=100),
            step_memory=StepMemory(fixed_bytes=200, bytes_per_row=10),
        )


# %% memory of the trainers


def test_a_jax_circuit_on_the_cpu_is_trained_in_host_memory(mixture_of_products_case):
    """
    The memory of a trainer of a jax circuit on the CPU is the memory of the host.
    """
    with jax.enable_x64(True):
        circuit = mixture_of_products_case.circuit(np).to_namespace(
            jnp, device=jax.devices("cpu")[0]
        )
        trainer = JaxGradientDescent(
            layer=circuit.root, loss=AverageNegativeLogLikelihood()
        )

    assert isinstance(trainer.device_memory(), HostMemory)


def test_a_compiled_jax_step_needs_memory_for_every_row(mixture_of_products_case):
    """
    The step jax compiles for a batch allocates more memory for more rows.
    """
    case = mixture_of_products_case
    np.random.seed(0)
    data = case.circuit(np).sample(500)
    with jax.enable_x64(True):
        circuit = case.circuit(np).to_namespace(jnp)
        step_memory = JaxGradientDescent(
            layer=circuit.root, loss=AverageNegativeLogLikelihood()
        ).step_memory(jnp.asarray(data))

    assert step_memory.bytes_per_row > 0


@dataclass(kw_only=True)
class StepBytesGivenPerBatchSize(JaxGradientDescent):
    """
    A jax trainer whose steps report given bytes per batch size instead of compiling,
    recording the batch sizes it was probed on.
    """

    step_bytes_by_rows: dict[int, int]
    """
    The bytes a step on each number of rows reports.
    """

    probed_rows: list[int] = field(default_factory=list)
    """
    The number of rows of every batch probed, in order.
    """

    def probe_step_bytes(self, *, batch: tuple[jax.Array, ...]) -> int:
        rows = batch[0].shape[0]
        self.probed_rows.append(rows)
        return self.step_bytes_by_rows[rows]


def test_memory_that_shrinks_from_the_smallest_batches_is_probed_on_larger_ones(
    mixture_of_products_case,
):
    """
    When the step on twice the rows reports no more memory, the next probe doubles the
    rows again, and the memory is the line through the first pair that grows.
    """
    step_bytes_by_rows = {64: 512728, 128: 462296, 256: 859608}
    with jax.enable_x64(True):
        circuit = mixture_of_products_case.circuit(np).to_namespace(jnp)
        trainer = StepBytesGivenPerBatchSize(
            layer=circuit.root,
            loss=AverageNegativeLogLikelihood(),
            step_bytes_by_rows=step_bytes_by_rows,
        )
        step_memory = trainer.step_memory(jnp.zeros((1000, 2)))

    assert trainer.probed_rows == [64, 128, 256]
    assert step_memory == StepMemory.from_two_probes(
        smaller=ProbeMeasurement(rows=128, step_bytes=step_bytes_by_rows[128]),
        larger=ProbeMeasurement(rows=256, step_bytes=step_bytes_by_rows[256]),
    )


def test_memory_that_never_grows_up_to_all_rows_is_unmeasurable(
    mixture_of_products_case,
):
    """
    A step that reports no more memory for more rows, however many rows there are, gives
    no memory per row.
    """
    with jax.enable_x64(True):
        circuit = mixture_of_products_case.circuit(np).to_namespace(jnp)
        trainer = StepBytesGivenPerBatchSize(
            layer=circuit.root,
            loss=AverageNegativeLogLikelihood(),
            step_bytes_by_rows={64: 500, 128: 500, 256: 400},
        )

        with pytest.raises(UnmeasurableStepMemoryError):
            trainer.step_memory(jnp.zeros((300, 2)))

    assert trainer.probed_rows == [64, 128, 256]


@pytest.fixture
def torch_trainer(mixture_of_products_case, request):
    """
    A trainer of a torch circuit on the device named by the parameter of the test, and
    data sampled from the circuit on that device.
    """
    torch = pytest.importorskip("torch")
    torch_gradient_descent = pytest.importorskip(
        "probabilistic_model.learning.parameters.gradient_descent.torch_gradient_descent"
    )
    torch_circuit_parameter_learning = pytest.importorskip(
        "probabilistic_model.learning.parameters.torch_circuit_parameter_learning"
    )
    device = torch.device(request.param)
    if (
        device.type
        == torch_circuit_parameter_learning.TorchDeviceType.COMPUTE_UNIFIED_DEVICE_ARCHITECTURE
        and not torch.cuda.is_available()
    ):
        pytest.skip("no CUDA device")
    case = mixture_of_products_case
    np.random.seed(0)
    namespace = array_namespace(torch.zeros(0))
    circuit = case.circuit(np).to_namespace(namespace, device=device)
    trainer = torch_gradient_descent.TorchGradientDescent(
        layer=circuit.root, loss=AverageNegativeLogLikelihood()
    )
    return trainer, namespace.asarray(case.circuit(np).sample(20000), device=device)


@pytest.mark.parametrize("torch_trainer", ["cpu"], indirect=True)
def test_the_memory_of_a_torch_step_on_the_cpu_cannot_be_measured(torch_trainer):
    """
    On the CPU, the memory of a torch step is not measured.
    """
    trainer, data = torch_trainer

    with pytest.raises(UnmeasurableHostStepMemoryError):
        trainer.step_memory(data)


@pytest.mark.parametrize("torch_trainer", ["cuda"], indirect=True)
def test_measuring_a_torch_step_leaves_the_parameters_unchanged(torch_trainer):
    """
    Measuring the memory of a step neither updates the parameters nor keeps gradients.
    """
    trainer, data = torch_trainer
    before = [parameter.detach().clone() for parameter in trainer.parameters]

    trainer.step_memory(data, probe_rows=5000)

    for parameter, original in zip(trainer.parameters, before):
        assert parameter.grad is None
        np.testing.assert_array_equal(
            parameter.detach().cpu().numpy(), original.cpu().numpy()
        )


@pytest.mark.parametrize("torch_trainer", ["cuda"], indirect=True)
def test_a_torch_step_on_a_cuda_device_needs_memory_for_every_row(torch_trainer):
    """
    On a CUDA device, the loss and gradient of a batch allocate more memory for more
    rows.
    """
    trainer, data = torch_trainer

    step_memory = trainer.step_memory(data, probe_rows=5000)

    assert step_memory.bytes_per_row > 0
