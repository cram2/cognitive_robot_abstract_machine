"""
Runs the circuit learning benchmark: every method on every dataset in every setting it
supports, and the queries of every circuit learned in a method's first setting, each run
in a process of its own on a share of the GPU or the processor cores.

A restarted benchmark resumes after the runs its database records.
"""

from __future__ import annotations

# %% imports
import argparse
import functools
import json
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
from krrood.adapters.json_serializer import from_json, to_json

from experiments.circuit_learning.database import BenchmarkDatabase
from experiments.circuit_learning.datasets import DatasetName, PreparedDatasets
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
    Job,
    JobKind,
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
from probabilistic_model.learning.training_schedule import Convergence

# %% names


class ThreadVariable(StrEnum):
    """
    The environment variables that cap the threads of the numeric libraries.
    """

    OPENMP = "OMP_NUM_THREADS"
    OPENBLAS = "OPENBLAS_NUM_THREADS"
    MKL = "MKL_NUM_THREADS"


class OutputFile(StrEnum):
    """
    The files and directories of a benchmark's output directory.
    """

    DATABASE = "results.sqlite"
    LOGS = "logs"


# %% limits


@dataclass(kw_only=True)
class JobLimits:
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

    memory_budget_bytes: int = 0
    """
    The most memory of its device a training step may allocate.
    """

    @property
    def training_timeout(self) -> timedelta:
        """
        :return: The time after which a training run is stopped.
        """
        return 2 * self.training_duration + self.evaluation_allowance


# %% one run


@dataclass(kw_only=True)
class JobExecution:
    """
    Carries out one run in the current process and stores what it measured.
    """

    job: Job
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

    limits: JobLimits
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
            name=self.job.dataset
        )
        method = BenchmarkMethods().find(name=self.job.method)
        if self.job.kind is JobKind.QUERIES:
            self.database.store(
                records=QueryMeasurement(
                    dataset=dataset,
                    configuration=BenchmarkConfiguration.from_setting(
                        learning_method=self.job.method,
                        dataset=self.job.dataset,
                        setting=self.job.setting,
                        seed=method.seed,
                    ),
                    setting=self.job.setting,
                    circuit=self.database.trained_circuit(job=self.job.training_job()),
                ).run()
            )
            return
        measured = TrainingMeasurement(
            dataset=dataset,
            method=method,
            setting=self.job.setting,
            context=TrainingContext(
                convergence=Convergence(maximum_duration=self.limits.training_duration),
                memory_budget_bytes=self.limits.memory_budget_bytes,
            ),
        ).run()
        self.database.store(
            records=[
                measured.result,
                TrainedCircuit(job=self.job, circuit=measured.circuit.to_namespace(np)),
            ]
        )


# %% scheduling


@dataclass(kw_only=True)
class Slot:
    """
    A share of the machine one run uses at a time.
    """

    uses_gpu: bool
    """
    Whether the share holds part of the GPU.
    """

    cores: set[int]
    """
    The processor cores of the share.
    """

    memory_budget_bytes: int
    """
    The memory of its device a training step of a run may allocate.
    """

    host_memory_limit_bytes: int
    """
    The resident memory above which a run is stopped.
    """


@dataclass(kw_only=True)
class RunningJob:
    """
    A run and the process carrying it out.
    """

    job: Job
    """
    The run.
    """

    process: subprocess.Popen
    """
    Its process.
    """

    slot: Slot
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
    Runs every job on the shares of the machine, the most informative first.
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

    training_duration: timedelta = timedelta(hours=1)
    """
    The time after which iterative training starts no further epoch.
    """

    gpu_slots: int = 3
    """
    The number of runs that share the GPU.
    """

    cpu_slots: int = 7
    """
    The number of runs that use processor cores only.
    """

    poll_interval: timedelta = timedelta(seconds=5)
    """
    How often running jobs are checked.
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

    def jobs(self) -> list[Job]:
        """
        :return: Every run in the order it is started: the training in each method's
            first setting, the queries of those circuits, then every other setting.
        """
        methods = BenchmarkMethods()
        datasets = PreparedDatasets(cache_directory=self.cache_directory)
        first, queries, rest = [], [], []
        query_settings = [
            *ComputationSetting.grid(
                array_libraries=(ArrayLibrary.NUMPY,), devices=(Device.CPU,)
            ),
            *ComputationSetting.grid(
                array_libraries=(ArrayLibrary.JAX, ArrayLibrary.TORCH)
            ),
        ]
        query_settings.sort(key=lambda setting: setting != ComputationSetting.host())
        for name in self.datasets:
            dataset = datasets.load(name=name)
            for method_name in self.methods:
                method = methods.find(name=method_name)
                if not method.supports(dataset=dataset):
                    continue
                settings = method.settings()
                first.append(
                    Job(
                        kind=JobKind.TRAINING,
                        dataset=name,
                        method=method_name,
                        setting=settings[0],
                    )
                )
                queries.extend(
                    Job(
                        kind=JobKind.QUERIES,
                        dataset=name,
                        method=method_name,
                        setting=setting,
                    )
                    for setting in query_settings
                )
                rest.extend(
                    Job(
                        kind=JobKind.TRAINING,
                        dataset=name,
                        method=method_name,
                        setting=setting,
                    )
                    for setting in settings[1:]
                )
        if self.first_settings_only:
            return first + queries
        return first + queries + rest

    def readiness(self, *, job: Job) -> RunStatus | None:
        """
        :param job: A run that has not started.
        :return: ``None`` if it can start, :attr:`RunStatus.SKIPPED` if it never can,
            :attr:`RunStatus.FINISHED` if it has to wait.
        """
        if job.kind is JobKind.TRAINING:
            return None
        trained = self.finished.get(job.training_job().key)
        if trained is None:
            return RunStatus.FINISHED
        if trained is RunStatus.FINISHED:
            return None
        return RunStatus.SKIPPED

    def slots(self) -> list[Slot]:
        """
        :return: The shares of the machine.
        """
        cores = sorted(os.sched_getaffinity(0))
        count = self.gpu_slots + self.cpu_slots
        per_slot = len(cores) // count
        available = psutil.virtual_memory().available
        host_share = int(0.8 * available / count)
        gpu_total = (
            torch.cuda.get_device_properties(0).total_memory if self.gpu_slots else 0
        )
        return [
            Slot(
                uses_gpu=index < self.gpu_slots,
                cores=set(cores[index * per_slot : (index + 1) * per_slot]),
                memory_budget_bytes=(
                    int(0.9 * gpu_total / max(self.gpu_slots, 1))
                    if index < self.gpu_slots
                    else host_share
                ),
                host_memory_limit_bytes=2 * host_share,
            )
            for index in range(count)
        ]

    def log_file(self, *, job: Job) -> Path:
        """
        :param job: A run.
        :return: The file its output is written to.
        """
        return self.output / OutputFile.LOGS / f"{job.key}.log"

    def record(self, *, job: Job, status: RunStatus, duration: timedelta) -> None:
        """
        Store how a run ended.
        """
        self.finished[job.key] = status
        log = self.log_file(job=job)
        log_tail = None
        if status is not RunStatus.FINISHED and log.exists():
            log_tail = log.read_text(errors="replace")[-self.log_tail_characters :]
        self.database.store(
            records=[
                RunRecord(job=job, status=status, duration=duration, log_tail=log_tail)
            ]
        )
        print(f"{status} {job.key} {duration}", flush=True)

    def start(self, *, job: Job, slot: Slot) -> RunningJob:
        """
        :return: The run, started in a process of its own on the share.
        """
        environment = dict(os.environ)
        for variable in ThreadVariable:
            environment[variable] = str(len(slot.cores))
        environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        # autotuning makes compiling a step take minutes on large batches
        environment["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
        if not slot.uses_gpu:
            environment["CUDA_VISIBLE_DEVICES"] = ""
        limits = JobLimits(
            training_duration=self.training_duration,
            memory_budget_bytes=slot.memory_budget_bytes,
        )
        log = self.log_file(job=job)
        log.parent.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "experiments.circuit_learning.runner",
                Command.EXECUTE,
                "--job",
                json.dumps(to_json(job)),
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
            preexec_fn=functools.partial(os.sched_setaffinity, 0, slot.cores),
        )
        return RunningJob(
            job=job,
            process=process,
            slot=slot,
            start=time.perf_counter(),
            timeout=(
                limits.training_timeout
                if job.kind is JobKind.TRAINING
                else limits.query_duration
            ),
        )

    def ended_status(self, *, running: RunningJob) -> RunStatus | None:
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
            > running.slot.host_memory_limit_bytes
        ):
            running.process.kill()
            running.process.wait()
            return RunStatus.OUT_OF_MEMORY
        return None

    def run(self) -> None:
        """
        Run every job that has not ended in an earlier start.
        """
        self.run_jobs(jobs=self.jobs())

    def run_jobs(self, *, jobs: list[Job]) -> None:
        """
        :param jobs: The runs, in the order they are started; those that ended in an
            earlier start are left out.
        """
        self.finished.update(
            {record.job.key: record.status for record in self.database.run_records()}
        )
        pending = [job for job in jobs if job.key not in self.finished]
        free = self.slots()
        running: list[RunningJob] = []
        while pending or running:
            for current in list(running):
                status = self.ended_status(running=current)
                if status is None:
                    continue
                running.remove(current)
                free.append(current.slot)
                self.record(job=current.job, status=status, duration=current.elapsed)
            for job in list(pending):
                readiness = self.readiness(job=job)
                if readiness is RunStatus.SKIPPED:
                    pending.remove(job)
                    self.record(job=job, status=RunStatus.SKIPPED, duration=timedelta())
                    continue
                if readiness is not None:
                    continue
                slot = next(
                    (each for each in free if each.uses_gpu == job.setting.uses_gpu),
                    None,
                )
                if slot is None:
                    continue
                free.remove(slot)
                pending.remove(job)
                running.append(self.start(job=job, slot=slot))
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
    parser.add_argument("--training-seconds", type=float, default=3600.0)
    parser.add_argument(
        "--datasets", nargs="+", default=[name.name for name in DatasetName]
    )
    parser.add_argument(
        "--methods", nargs="+", default=[name.name for name in LearningMethod]
    )
    parser.add_argument("--gpu-slots", type=int, default=3)
    parser.add_argument("--cpu-slots", type=int, default=7)
    parser.add_argument("--first-settings-only", action="store_true")
    parser.add_argument("--job")
    parser.add_argument("--memory-budget-bytes", type=int, default=0)
    arguments = parser.parse_args()
    training_duration = timedelta(seconds=arguments.training_seconds)
    if Command(arguments.command) is Command.EXECUTE:
        JobExecution(
            job=from_json(json.loads(arguments.job)),
            database=BenchmarkDatabase(path=arguments.output / OutputFile.DATABASE),
            cache_directory=arguments.cache_directory,
            limits=JobLimits(
                training_duration=training_duration,
                memory_budget_bytes=arguments.memory_budget_bytes,
            ),
        ).execute()
        return
    BenchmarkRunner(
        datasets=[DatasetName[name] for name in arguments.datasets],
        methods=[LearningMethod[name] for name in arguments.methods],
        output=arguments.output,
        cache_directory=arguments.cache_directory,
        training_duration=training_duration,
        gpu_slots=arguments.gpu_slots,
        cpu_slots=arguments.cpu_slots,
        first_settings_only=arguments.first_settings_only,
    ).run()


if __name__ == "__main__":
    main()
