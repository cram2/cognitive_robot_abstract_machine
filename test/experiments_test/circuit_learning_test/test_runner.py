"""
Tests of how the benchmark runner assigns runs to the shares of the machine.
"""

from __future__ import annotations

# %% imports
from pathlib import Path

import pandas as pd
import pytest
from krrood_compute_resources.devices import Device
from krrood_compute_resources.machine import MachineDivision, MachineShare

from experiments.circuit_learning.datasets import (
    DatasetName,
    DatasetSplit,
    PreparedDatasets,
)
from experiments.circuit_learning.exceptions import NoShareForRunError
from experiments.circuit_learning.methods import LearningMethod
from experiments.circuit_learning.records import BenchmarkRun, RunKind
from experiments.circuit_learning.runner import BenchmarkRunner
from experiments.circuit_learning.settings import ComputationSetting
from probabilistic_model.learning.jax_gradient_descent import (
    CompilerOption,
    CompilerVariable,
)

# %% fixtures


@pytest.fixture
def prepared_cache(tmp_path: Path) -> Path:
    """
    :return: A cache holding a prepared binary dataset, so that nothing is downloaded.
    """
    datasets = PreparedDatasets(cache_directory=tmp_path)
    for split in DatasetSplit:
        file = datasets.split_file(name=DatasetName.NLTCS, split=split)
        file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"x0000": ["0", "1"], "x0001": ["1", "0"]}).to_pickle(file)
    return tmp_path


def runner_without_graphics_processor(*, cache: Path) -> BenchmarkRunner:
    """
    :return: A runner on every method whose machine has no share of the GPU.
    """
    return BenchmarkRunner(
        datasets=[DatasetName.NLTCS],
        methods=list(LearningMethod),
        output=cache,
        cache_directory=cache,
        division=MachineDivision(graphics_processor_shares=0),
    )


# %% shares without a GPU


def test_runs_without_graphics_processor_stay_on_processor_cores(
    prepared_cache: Path,
):
    """
    Without a share of the GPU, every method trains in a setting on the processor cores.
    """
    runs = runner_without_graphics_processor(cache=prepared_cache).runs()
    assert {run.method for run in runs if run.kind is RunKind.TRAINING} == set(
        LearningMethod
    )
    assert not any(run.setting.uses_graphics_processor for run in runs)


def test_run_without_a_share_fails(prepared_cache: Path):
    """
    A run on the GPU fails at once when the machine has no share of the GPU.
    """
    run = BenchmarkRun(
        kind=RunKind.TRAINING,
        dataset=DatasetName.NLTCS,
        method=LearningMethod.RANDOM_TENSORIZED_SUM_PRODUCT_NETWORK_GRADIENT_DESCENT,
        setting=ComputationSetting.reference(),
    )
    with pytest.raises(NoShareForRunError):
        runner_without_graphics_processor(cache=prepared_cache).run_pending(runs=[run])


# %% compiler options of a run


def compiler_options_of_share(*, cache: Path, device: Device) -> list[str]:
    """
    :param cache: The cache of the runner.
    :param device: The device of a share of 25 processor cores.
    :return: The options the runner gives the XLA compiler of a run on that share.
    """
    share = MachineShare(
        device=device,
        processor_cores=set(range(25)),
        memory_budget_bytes=2**30,
        host_memory_limit_bytes=2**30,
    )
    environment = runner_without_graphics_processor(cache=cache).run_environment(
        share=share
    )
    return environment[CompilerVariable.ACCELERATED_LINEAR_ALGEBRA_OPTIONS].split()


def test_a_run_on_processor_cores_gets_a_jax_device_per_power_of_two_cores(
    prepared_cache: Path,
):
    """
    A run on a share of 25 processor cores computes with jax on 16 CPU devices, the
    largest power of two of at most 25.
    """
    assert CompilerOption.HOST_PLATFORM_DEVICE_COUNT.to_flag(
        value=16
    ) in compiler_options_of_share(
        cache=prepared_cache, device=Device.CENTRAL_PROCESSOR
    )


def test_a_run_on_the_graphics_processor_keeps_one_jax_host_device(
    prepared_cache: Path,
):
    """
    A run on a share of the GPU leaves the number of CPU devices of jax alone.
    """
    options = compiler_options_of_share(
        cache=prepared_cache, device=Device.GRAPHICS_PROCESSOR
    )
    assert not any(
        option.startswith(CompilerOption.HOST_PLATFORM_DEVICE_COUNT)
        for option in options
    )
