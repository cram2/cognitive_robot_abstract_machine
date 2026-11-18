"""
Tests of how the benchmark runner assigns runs to the shares of the machine.
"""

from __future__ import annotations

# %% imports
from pathlib import Path

import pandas as pd
import pytest

from experiments.circuit_learning.datasets import (
    DatasetName,
    DatasetSplit,
    PreparedDatasets,
)
from experiments.circuit_learning.exceptions import NoSlotForRunError
from experiments.circuit_learning.methods import LearningMethod
from experiments.circuit_learning.records import BenchmarkRun, RunKind
from experiments.circuit_learning.runner import BenchmarkRunner, MachineShares
from experiments.circuit_learning.settings import ComputationSetting

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
        shares=MachineShares(graphics_processor_slots=0),
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
    with pytest.raises(NoSlotForRunError):
        runner_without_graphics_processor(cache=prepared_cache).run_pending(runs=[run])
