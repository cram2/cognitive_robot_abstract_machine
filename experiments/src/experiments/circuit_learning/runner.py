"""
Runs the circuit learning benchmark: every method on every dataset in every setting it
supports, and the queries of every circuit learned in a method's first setting, each run
in a process of its own on a share of the GPU or the processor cores.

A restarted benchmark resumes after the runs its database records.
"""

from __future__ import annotations

# %% imports
import argparse
import dataclasses
import functools
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from pathlib import Path

import jax
import numpy as np
import psutil
import torch
from krrood_compute_resources.machine import Machine, MachineDivision, MachineShare
from krrood_compute_resources.torch_memory import TorchDeviceMemory
from krrood.adapters.json_serializer import from_json, to_json

from experiments.circuit_learning.database import BenchmarkDatabase
from experiments.circuit_learning.datasets import DatasetName, PreparedDatasets
from experiments.circuit_learning.exceptions import NoShareForRunError
from experiments.circuit_learning.measurements import (
    QueryMeasurement,
    TrainingMeasurement,
)
from experiments.circuit_learning.methods import (
    BenchmarkMethods,
    LearningMethod,
    TrainingContext,
)
from experiments.circuit_learning.records import (
    BenchmarkRun,
    RunKind,
    RunRecord,
    RunStatus,
    TrainedCircuit,
)
from experiments.circuit_learning.rows import BenchmarkConfiguration
from experiments.circuit_learning.settings import (
    ArrayLibrary,
    ComputationSetting,
    Device,
)
from probabilistic_model.learning.torch_gradient_descent import TorchDeviceType
from probabilistic_model.learning.training_schedule import Convergence

# %% names


class CompilerVariable(StrEnum):
    """
    The environment variables that configure the compilers of a run.
    """

    ACCELERATED_LINEAR_ALGEBRA_OPTIONS = "XLA_FLAGS"
    """
    The options of the XLA compiler.
    """


class CompilerOption(StrEnum):
    """
    The options of the XLA compiler a run is given.
    """

    GRAPHICS_PROCESSOR_AUTOTUNE_LEVEL = "--xla_gpu_autotune_level"
    """
    How much XLA tunes the kernels it compiles for the GPU.
    """

    HOST_PLATFORM_DEVICE_COUNT = "--xla_force_host_platform_device_count"
    """
    The number of devices jax divides the processor cores into.
    """


class OutputFile(StrEnum):
    """
    The files and directories of a benchmark's output directory.
    """

    DATABASE = "results.sqlite"
    LOGS = "logs"


# %% limits


@dataclass(kw_only=True)
class RunLimits:
    """
    The limits every run of a benchmark keeps to.
    """

    training_duration: timedelta = timedelta(hours=1)
    """
    The time after which iterative training starts no further epoch.
    """

    query_duration: timedelta = timedelta(hours=1)
    """
    The time after which a run of queries is stopped.
    """

    evaluation_allowance: timedelta = timedelta(minutes=30)
    """
    The time a training run has beyond its training, for building the structure and
    evaluating the circuit.
    """

    training_timeout_factor: float = 2.0
    """
    The multiple of :attr:`training_duration` a training run may train before it is
    stopped, since the epoch that is running when :attr:`training_duration` passes still
    finishes.
    """

    memory_budget_bytes: int = 0
    """
    The most memory of its device a training step may allocate.
    """

    @property
    def training_timeout(self) -> timedelta:
        """
        :return: The time after which a training run is stopped.
        """
        return (
            self.training_timeout_factor * self.training_duration
            + self.evaluation_allowance
        )


# %% one run


@dataclass(kw_only=True)
class RunExecution:
    """
    Carries out one run in the current process and stores what it measured.
    """

    run: BenchmarkRun
    """
    The run.
    """

    database: BenchmarkDatabase
    """
    The database of the benchmark.
    """

    cache_directory: Path
    """
    The directory of the downloaded and prepared datasets.
    """

    limits: RunLimits
    """
    The limits of the run.
    """

    def execute(self) -> None:
        """
        Measure, and store the rows, and the learned circuit of a training run, in one
        transaction.
        """
        jax.config.update("jax_enable_x64", True)
        dataset = PreparedDatasets(cache_directory=self.cache_directory).load(
            name=self.run.dataset
        )
        method = BenchmarkMethods().find(name=self.run.method)
        if self.run.kind is RunKind.QUERIES:
            self.database.store(
                records=QueryMeasurement(
                    dataset=dataset,
                    configuration=BenchmarkConfiguration.from_setting(
                        learning_method=self.run.method,
                        dataset=self.run.dataset,
                        setting=self.run.setting,
                        seed=method.seed,
                    ),
                    setting=self.run.setting,
                    circuit=self.database.trained_circuit(run=self.run.training_run()),
                ).run()
            )
            return
        measured = TrainingMeasurement(
            dataset=dataset,
            method=method,
            setting=self.run.setting,
            context=TrainingContext(
                convergence=Convergence(maximum_duration=self.limits.training_duration),
                memory_budget_bytes=self.limits.memory_budget_bytes,
            ),
        ).run()
        self.database.store(
            records=[
                measured.result,
                TrainedCircuit(
                    training_run=self.run, circuit=measured.circuit.to_namespace(np)
                ),
            ]
        )


# %% scheduling


@dataclass(kw_only=True)
class RunInProgress:
    """
    A run and the process carrying it out.
    """

    run: BenchmarkRun
    """
    The run.
    """

    process: subprocess.Popen
    """
    Its process.
    """

    share: MachineShare
    """
    The share of the machine it uses.
    """

    start: float
    """
    When it started, as :func:`time.perf_counter` reads it.
    """

    timeout: timedelta
    """
    The time after which it is stopped.
    """

    @property
    def elapsed(self) -> timedelta:
        """
        :return: The time since the run started.
        """
        return timedelta(seconds=time.perf_counter() - self.start)


@dataclass(kw_only=True)
class BenchmarkRunner:
    """
    Runs every run on the shares of the machine, the most informative first.
    """

    datasets: list[DatasetName]
    """
    The datasets, smallest first.
    """

    methods: list[LearningMethod]
    """
    The methods.
    """

    output: Path
    """
    The output directory.
    """

    cache_directory: Path
    """
    The directory of the downloaded and prepared datasets.
    """

    limits: RunLimits = field(default_factory=RunLimits)
    """
    The limits of every run, its memory budget replaced by that of the share it runs on.
    """

    division: MachineDivision = field(default_factory=MachineDivision)
    """
    How the machine is divided among the runs.
    """

    poll_interval: timedelta = timedelta(seconds=5)
    """
    How often running runs are checked.
    """

    log_tail_characters: int = 3000
    """
    How much of the end of a failed run's output its record keeps.
    """

    first_settings_only: bool = False
    """
    Whether only the trainings in each method's first setting and their queries run.
    """

    finished: dict[str, RunStatus] = field(default_factory=dict)
    """
    The status of every run that ended, by its key.
    """

    @functools.cached_property
    def database(self) -> BenchmarkDatabase:
        """
        :return: The database of the output directory.
        """
        return BenchmarkDatabase(path=self.output / OutputFile.DATABASE)

    def runs(self) -> list[BenchmarkRun]:
        """
        :return: Every run in a setting the shares of the machine can compute in, in the
            order it is started: the training in each method's first such setting, the
            queries of those circuits, then every other setting.
        """
        methods = BenchmarkMethods()
        datasets = PreparedDatasets(cache_directory=self.cache_directory)
        first, queries, rest = [], [], []
        query_settings = [
            *ComputationSetting.grid(
                array_libraries=(ArrayLibrary.NUMPY,),
                devices=(Device.CENTRAL_PROCESSOR,),
            ),
            *ComputationSetting.grid(
                array_libraries=(ArrayLibrary.JAX, ArrayLibrary.TORCH)
            ),
        ]
        query_settings = [
            setting
            for setting in query_settings
            if self.division.shares_device(device=setting.device)
        ]
        query_settings.sort(key=lambda setting: setting != ComputationSetting.host())
        for name in self.datasets:
            dataset = datasets.load(name=name)
            for method_name in self.methods:
                method = methods.find(name=method_name)
                settings = [
                    setting
                    for setting in method.settings()
                    if self.division.shares_device(device=setting.device)
                ]
                if not settings or not method.supports(dataset=dataset):
                    continue
                first.append(
                    BenchmarkRun(
                        kind=RunKind.TRAINING,
                        dataset=name,
                        method=method_name,
                        setting=settings[0],
                    )
                )
                queries.extend(
                    BenchmarkRun(
                        kind=RunKind.QUERIES,
                        dataset=name,
                        method=method_name,
                        setting=setting,
                    )
                    for setting in query_settings
                )
                rest.extend(
                    BenchmarkRun(
                        kind=RunKind.TRAINING,
                        dataset=name,
                        method=method_name,
                        setting=setting,
                    )
                    for setting in settings[1:]
                )
        if self.first_settings_only:
            return first + queries
        return first + queries + rest

    def readiness(self, *, run: BenchmarkRun) -> RunStatus | None:
        """
        :param run: A run that has not started.
        :return: ``None`` if it can start, :attr:`RunStatus.SKIPPED` if it never can,
            :attr:`RunStatus.FINISHED` if it has to wait.
        """
        if run.kind is RunKind.TRAINING:
            return None
        trained = self.finished.get(run.training_run().key)
        if trained is None:
            return RunStatus.FINISHED
        if trained is RunStatus.FINISHED:
            return None
        return RunStatus.SKIPPED

    def machine(self) -> Machine:
        """
        :return: The machine the benchmark runs on, with the memory of the first CUDA
            device if the division has shares of the GPU.
        """
        return Machine.from_current_process(
            graphics_processor_memory=(
                TorchDeviceMemory(
                    device=torch.device(
                        TorchDeviceType.COMPUTE_UNIFIED_DEVICE_ARCHITECTURE
                    )
                )
                if self.division.graphics_processor_shares
                else None
            )
        )

    def log_file(self, *, run: BenchmarkRun) -> Path:
        """
        :param run: A run.
        :return: The file its output is written to.
        """
        return self.output / OutputFile.LOGS / f"{run.key}.log"

    def record(
        self, *, run: BenchmarkRun, status: RunStatus, duration: timedelta
    ) -> None:
        """
        Store how a run ended.
        """
        self.finished[run.key] = status
        log = self.log_file(run=run)
        log_tail = None
        if status is not RunStatus.FINISHED and log.exists():
            log_tail = log.read_text(errors="replace")[-self.log_tail_characters :]
        self.database.store(
            records=[
                RunRecord(run=run, status=status, duration=duration, log_tail=log_tail)
            ]
        )
        print(f"{status} {run.key} {duration}", flush=True)

    def run_environment(self, *, share: MachineShare) -> dict[str, str]:
        """
        :param share: The share of the machine a run computes on.
        :return: The environment of the process of the run, whose jax divides the
            processor cores of a share without the GPU into as many devices as the
            largest power of two of at most its number of cores.
        """
        environment = share.process_environment(environment=dict(os.environ))
        # autotuning makes compiling a step take minutes on large batches
        options = [f"{CompilerOption.GRAPHICS_PROCESSOR_AUTOTUNE_LEVEL}=0"]
        if share.device is Device.CENTRAL_PROCESSOR:
            # a power of two splits the batch sizes of the methods evenly
            devices = 2 ** int(math.log2(len(share.processor_cores)))
            options.append(f"{CompilerOption.HOST_PLATFORM_DEVICE_COUNT}={devices}")
        environment[CompilerVariable.ACCELERATED_LINEAR_ALGEBRA_OPTIONS] = " ".join(
            options
        )
        return environment

    def start(self, *, run: BenchmarkRun, share: MachineShare) -> RunInProgress:
        """
        :return: The run, started in a process of its own on the share.
        """
        environment = self.run_environment(share=share)
        limits = dataclasses.replace(
            self.limits, memory_budget_bytes=share.memory_budget_bytes
        )
        log = self.log_file(run=run)
        log.parent.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "experiments.circuit_learning.runner",
                Command.EXECUTE,
                "--run",
                json.dumps(to_json(run)),
                "--output",
                str(self.output),
                "--cache-directory",
                str(self.cache_directory),
                "--training-seconds",
                str(limits.training_duration.total_seconds()),
                "--memory-budget-bytes",
                str(limits.memory_budget_bytes),
            ],
            env=environment,
            stdout=log.open("w"),
            stderr=subprocess.STDOUT,
            preexec_fn=share.confine_current_process,
        )
        return RunInProgress(
            run=run,
            process=process,
            share=share,
            start=time.perf_counter(),
            timeout=(
                limits.training_timeout
                if run.kind is RunKind.TRAINING
                else limits.query_duration
            ),
        )

    def ended_status(self, *, running: RunInProgress) -> RunStatus | None:
        """
        :return: How the run ended, stopping it if it ran out of time or memory,
            ``None`` while it keeps running.
        """
        exit_code = running.process.poll()
        if exit_code is not None:
            return RunStatus.FINISHED if exit_code == 0 else RunStatus.FAILED
        if running.elapsed > running.timeout:
            running.process.kill()
            running.process.wait()
            return RunStatus.TIMED_OUT
        if (
            psutil.Process(running.process.pid).memory_info().rss
            > running.share.host_memory_limit_bytes
        ):
            running.process.kill()
            running.process.wait()
            return RunStatus.OUT_OF_MEMORY
        return None

    def run(self) -> None:
        """
        Run every run that has not ended in an earlier start.
        """
        self.run_pending(runs=self.runs())

    def run_pending(self, *, runs: list[BenchmarkRun]) -> None:
        """
        :param runs: The runs, in the order they are started; those that ended in an
            earlier start are left out.
        :raises NoShareForRunError: If a run is in a setting no share of the machine can
            compute in.
        """
        impossible = [
            run
            for run in runs
            if not self.division.shares_device(device=run.setting.device)
        ]
        if impossible:
            raise NoShareForRunError(runs=impossible)
        self.finished.update(
            {record.run.key: record.status for record in self.database.run_records()}
        )
        pending = [run for run in runs if run.key not in self.finished]
        free = self.division.shares(machine=self.machine())
        running: list[RunInProgress] = []
        while pending or running:
            for current in list(running):
                status = self.ended_status(running=current)
                if status is None:
                    continue
                running.remove(current)
                free.append(current.share)
                self.record(run=current.run, status=status, duration=current.elapsed)
            for run in list(pending):
                readiness = self.readiness(run=run)
                if readiness is RunStatus.SKIPPED:
                    pending.remove(run)
                    self.record(run=run, status=RunStatus.SKIPPED, duration=timedelta())
                    continue
                if readiness is not None:
                    continue
                share = next(
                    (each for each in free if each.device is run.setting.device), None
                )
                if share is None:
                    continue
                free.remove(share)
                pending.remove(run)
                running.append(self.start(run=run, share=share))
            time.sleep(self.poll_interval.total_seconds())


# %% command line


class Command(StrEnum):
    """
    The commands of the command line.
    """

    RUN = "run"
    EXECUTE = "execute"


def main() -> None:
    """
    Parse the command line and run the benchmark or one of its runs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=list(Command))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-directory",
        type=Path,
        default=Path.home() / ".cache" / "probabilistic_model" / "benchmark",
    )
    default_limits = RunLimits()
    default_division = MachineDivision()
    parser.add_argument(
        "--training-seconds",
        type=float,
        default=default_limits.training_duration.total_seconds(),
    )
    parser.add_argument(
        "--datasets", nargs="+", default=[name.name for name in DatasetName]
    )
    parser.add_argument(
        "--methods", nargs="+", default=[name.name for name in LearningMethod]
    )
    parser.add_argument(
        "--graphics-processor-shares",
        type=int,
        default=default_division.graphics_processor_shares,
    )
    parser.add_argument(
        "--processor-core-shares",
        type=int,
        default=default_division.processor_core_shares,
    )
    parser.add_argument("--first-settings-only", action="store_true")
    parser.add_argument("--run")
    parser.add_argument(
        "--memory-budget-bytes", type=int, default=default_limits.memory_budget_bytes
    )
    arguments = parser.parse_args()
    limits = RunLimits(
        training_duration=timedelta(seconds=arguments.training_seconds),
        memory_budget_bytes=arguments.memory_budget_bytes,
    )
    if Command(arguments.command) is Command.EXECUTE:
        RunExecution(
            run=from_json(json.loads(arguments.run)),
            database=BenchmarkDatabase(path=arguments.output / OutputFile.DATABASE),
            cache_directory=arguments.cache_directory,
            limits=limits,
        ).execute()
        return
    BenchmarkRunner(
        datasets=[DatasetName[name] for name in arguments.datasets],
        methods=[LearningMethod[name] for name in arguments.methods],
        output=arguments.output,
        cache_directory=arguments.cache_directory,
        limits=limits,
        division=MachineDivision(
            graphics_processor_shares=arguments.graphics_processor_shares,
            processor_core_shares=arguments.processor_core_shares,
        ),
        first_settings_only=arguments.first_settings_only,
    ).run()


if __name__ == "__main__":
    main()
