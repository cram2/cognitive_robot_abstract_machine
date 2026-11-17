"""
Runs the circuit learning benchmark: every method on every dataset in every setting it
supports, and the queries of every circuit learned in a method's first setting, each run
in a process of its own on a share of the GPU or the processor cores.

A restarted benchmark resumes after its finished runs.
"""

from __future__ import annotations

# %% imports
import argparse
import enum
import functools
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import jax
import psutil
import torch
from typing_extensions import Any

from experiments.circuit_learning.datasets import (
    BenchmarkDataset,
    DatasetName,
    DatasetSources,
)
from experiments.circuit_learning.measurements import (
    QueryMeasurement,
    TrainingMeasurement,
)
from experiments.circuit_learning.methods import (
    BenchmarkMethods,
    LearningMethod,
    TrainingContext,
)
from experiments.circuit_learning.rows import BenchmarkConfiguration
from experiments.circuit_learning.settings import (
    ArrayLibrary,
    ComputationSetting,
    Device,
    EdgeRepresentation,
    FloatingPointType,
)
from experiments.circuit_learning.storage import CircuitFile, PickledFile
from experiments.experiment_definitions import ExperimentsTable
from probabilistic_model.learning.training_schedule import Convergence

# %% names


class JobKind(enum.Enum):
    """
    What a run measures.
    """

    TRAINING = "training"
    QUERIES = "queries"


class RunStatus(enum.Enum):
    """
    How a run ended.
    """

    FINISHED = "finished"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    OUT_OF_MEMORY = "out_of_memory"
    SKIPPED = "skipped"


class ThreadVariable(enum.Enum):
    """
    The environment variables that cap the threads of the numeric libraries.
    """

    OPENMP = "OMP_NUM_THREADS"
    OPENBLAS = "OPENBLAS_NUM_THREADS"
    MKL = "MKL_NUM_THREADS"


class ResultFile(enum.Enum):
    """
    The files and directories of a benchmark's output directory.
    """

    RUNS = "runs.jsonl"
    ROWS = "rows"
    CIRCUITS = "circuits"
    LOGS = "logs"


# %% jobs


@dataclass(kw_only=True, frozen=True)
class Job:
    """
    One run of the benchmark.
    """

    kind: JobKind
    """
    What the run measures.
    """

    dataset: DatasetName
    """
    The dataset.
    """

    method: LearningMethod
    """
    The method that learns, or learned, the circuit.
    """

    setting: ComputationSetting
    """
    The setting the run computes in.
    """

    @property
    def key(self) -> str:
        """
        :return: A name of the run unique within a benchmark, usable as a file name.
        """
        return (
            f"{self.kind.value}.{self.dataset.value}.{self.method.name.lower()}."
            f"{self.setting.key}"
        )

    def to_json(self) -> dict[str, str]:
        """
        :return: The run as a JSON object.
        """
        return {
            "kind": self.kind.name,
            "dataset": self.dataset.name,
            "method": self.method.name,
            "array_library": self.setting.array_library.name,
            "device": self.setting.device.name,
            "floating_point_type": self.setting.floating_point_type.name,
            "edge_representation": self.setting.edge_representation.name,
        }

    @classmethod
    def from_json(cls, *, data: dict[str, str]) -> Job:
        """
        :param data: A run as :meth:`to_json` gives it.
        :return: The run.
        """
        return cls(
            kind=JobKind[data["kind"]],
            dataset=DatasetName[data["dataset"]],
            method=LearningMethod[data["method"]],
            setting=ComputationSetting(
                array_library=ArrayLibrary[data["array_library"]],
                device=Device[data["device"]],
                floating_point_type=FloatingPointType[data["floating_point_type"]],
                edge_representation=EdgeRepresentation[data["edge_representation"]],
            ),
        )


@dataclass(kw_only=True)
class JobLimits:
    """
    The limits every run of a benchmark keeps to.
    """

    training_seconds: float = 3600.0
    """
    The time after which iterative training starts no further epoch.
    """

    query_seconds: float = 3600.0
    """
    The time after which a run of queries is stopped.
    """

    memory_budget_bytes: int = 0
    """
    The most memory of its device a training step may allocate.
    """

    @property
    def training_timeout_seconds(self) -> float:
        """
        :return: The time after which a training run is stopped, which leaves time for
            building the structure and evaluating the circuit.
        """
        return 2 * self.training_seconds + 1800.0


# %% one run


@dataclass(kw_only=True)
class JobExecution:
    """
    Carries out one run in the current process and writes its rows.
    """

    job: Job
    """
    The run.
    """

    output: Path
    """
    The output directory of the benchmark.
    """

    cache_directory: Path
    """
    The directory of the downloaded and prepared datasets.
    """

    limits: JobLimits
    """
    The limits of the run.
    """

    def circuit_file(self, *, job: Job) -> CircuitFile:
        """
        :param job: A training run.
        :return: The file its circuit is stored in.
        """
        return CircuitFile(
            path=self.output / ResultFile.CIRCUITS.value / f"{job.key}.pickle"
        )

    def execute(self) -> None:
        """
        Measure, and write one JSON line per row to the rows file of the run.
        """
        jax.config.update("jax_enable_x64", True)
        dataset = PreparedDatasets(cache_directory=self.cache_directory).load(
            name=self.job.dataset
        )
        method = BenchmarkMethods().find(name=self.job.method)
        if self.job.kind is JobKind.TRAINING:
            rows = [self.training_row(dataset=dataset, method=method)]
        else:
            rows = self.query_rows(dataset=dataset, method=method)
        path = self.output / ResultFile.ROWS.value / f"{self.job.key}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as file:
            for row in rows:
                record = ExperimentsTable([row]).as_json_rows()[0]
                record["row_type"] = type(row).__name__
                file.write(json.dumps(record) + "\n")

    def training_row(self, *, dataset: BenchmarkDataset, method: Any) -> Any:
        """
        :return: The training measurements of the run.
        """
        return TrainingMeasurement(
            dataset=dataset,
            method=method,
            setting=self.job.setting,
            context=TrainingContext(
                convergence=Convergence(maximum_seconds=self.limits.training_seconds),
                memory_budget_bytes=self.limits.memory_budget_bytes,
            ),
            circuit_file=self.circuit_file(job=self.job),
        ).run()

    def query_rows(self, *, dataset: BenchmarkDataset, method: Any) -> list[Any]:
        """
        :return: The query measurements of the run.
        """
        trained = Job(
            kind=JobKind.TRAINING,
            dataset=self.job.dataset,
            method=self.job.method,
            setting=method.settings()[0],
        )
        return QueryMeasurement(
            dataset=dataset,
            configuration=BenchmarkConfiguration.from_setting(
                learning_method=self.job.method,
                dataset=self.job.dataset,
                setting=self.job.setting,
                seed=method.seed,
            ),
            setting=self.job.setting,
            circuit=self.circuit_file(job=trained).load(),
        ).run()


# %% prepared datasets


@dataclass(kw_only=True)
class PreparedDatasets:
    """
    The datasets of the benchmark, loaded once from their sources and kept as files.
    """

    cache_directory: Path
    """
    The directory of the downloaded and prepared datasets.
    """

    def file(self, *, name: DatasetName) -> PickledFile:
        """
        :param name: A dataset.
        :return: The file it is kept in.
        """
        return PickledFile(
            path=self.cache_directory / "prepared" / f"{name.value}.pickle"
        )

    def load(self, *, name: DatasetName) -> BenchmarkDataset:
        """
        :param name: A dataset.
        :return: The dataset, prepared from its source on first use.
        """
        file = self.file(name=name)
        if not file.path.exists():
            dataset = (
                DatasetSources(cache_directory=self.cache_directory)
                .find(name=name)
                .load()
            )
            file.save_object(value=dataset)
            return dataset
        return file.load()


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
    When it started.
    """

    timeout_seconds: float
    """
    The time after which it is stopped.
    """


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

    training_seconds: float = 3600.0
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

    poll_seconds: float = 5.0
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

    def trained_job(self, *, job: Job) -> Job:
        """
        :param job: A query run.
        :return: The training run whose circuit it queries.
        """
        return Job(
            kind=JobKind.TRAINING,
            dataset=job.dataset,
            method=job.method,
            setting=BenchmarkMethods().find(name=job.method).settings()[0],
        )

    def readiness(self, *, job: Job) -> RunStatus | None:
        """
        :param job: A run that has not started.
        :return: ``None`` if it can start, :attr:`RunStatus.SKIPPED` if it never can,
            :attr:`RunStatus.FINISHED` if it has to wait.
        """
        if job.kind is JobKind.TRAINING:
            return None
        trained = self.finished.get(self.trained_job(job=job).key)
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

    def record(self, *, job: Job, status: RunStatus, seconds: float) -> None:
        """
        Append the end of a run to the runs file.
        """
        self.finished[job.key] = status
        record: dict[str, Any] = {
            "key": job.key,
            **job.to_json(),
            "status": status.value,
            "wall_seconds": seconds,
        }
        log = self.output / ResultFile.LOGS.value / f"{job.key}.log"
        if status is not RunStatus.FINISHED and log.exists():
            record["log_tail"] = log.read_text(errors="replace")[
                -self.log_tail_characters :
            ]
        with (self.output / ResultFile.RUNS.value).open("a") as file:
            file.write(json.dumps(record) + "\n")
        print(
            json.dumps(
                {key: value for key, value in record.items() if key != "log_tail"}
            ),
            flush=True,
        )

    def load_finished(self) -> None:
        """
        Read the runs that ended in an earlier start of the benchmark.
        """
        path = self.output / ResultFile.RUNS.value
        if not path.exists():
            return
        for line in path.read_text().splitlines():
            record = json.loads(line)
            self.finished[record["key"]] = RunStatus(record["status"])

    def start(self, *, job: Job, slot: Slot) -> RunningJob:
        """
        :return: The run, started in a process of its own on the share.
        """
        environment = dict(os.environ)
        for variable in ThreadVariable:
            environment[variable.value] = str(len(slot.cores))
        environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        # autotuning makes compiling a step take minutes on large batches
        environment["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
        if not slot.uses_gpu:
            environment["CUDA_VISIBLE_DEVICES"] = ""
        limits = JobLimits(
            training_seconds=self.training_seconds,
            memory_budget_bytes=slot.memory_budget_bytes,
        )
        log = self.output / ResultFile.LOGS.value / f"{job.key}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "experiments.circuit_learning.runner",
                "execute",
                "--job",
                json.dumps(job.to_json()),
                "--output",
                str(self.output),
                "--cache-directory",
                str(self.cache_directory),
                "--training-seconds",
                str(limits.training_seconds),
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
            timeout_seconds=(
                limits.training_timeout_seconds
                if job.kind is JobKind.TRAINING
                else limits.query_seconds
            ),
        )

    def ended_status(self, *, running: RunningJob) -> RunStatus | None:
        """
        :return: How the run ended, stopping it if it ran out of time or memory,
            ``None`` while it keeps running.
        """
        exit_code = running.process.poll()
        if exit_code is not None:
            rows = self.output / ResultFile.ROWS.value / f"{running.job.key}.jsonl"
            if exit_code == 0 and rows.exists():
                return RunStatus.FINISHED
            return RunStatus.FAILED
        if time.perf_counter() - running.start > running.timeout_seconds:
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
        self.output.mkdir(parents=True, exist_ok=True)
        self.load_finished()
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
                self.record(
                    job=current.job,
                    status=status,
                    seconds=time.perf_counter() - current.start,
                )
            for job in list(pending):
                readiness = self.readiness(job=job)
                if readiness is RunStatus.SKIPPED:
                    pending.remove(job)
                    self.record(job=job, status=RunStatus.SKIPPED, seconds=0.0)
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
            time.sleep(self.poll_seconds)


# %% command line


class Command(enum.Enum):
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
    parser.add_argument("command", choices=[command.value for command in Command])
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
    if Command(arguments.command) is Command.EXECUTE:
        JobExecution(
            job=Job.from_json(data=json.loads(arguments.job)),
            output=arguments.output,
            cache_directory=arguments.cache_directory,
            limits=JobLimits(
                training_seconds=arguments.training_seconds,
                memory_budget_bytes=arguments.memory_budget_bytes,
            ),
        ).execute()
        return
    BenchmarkRunner(
        datasets=[DatasetName[name] for name in arguments.datasets],
        methods=[LearningMethod[name] for name in arguments.methods],
        output=arguments.output,
        cache_directory=arguments.cache_directory,
        training_seconds=arguments.training_seconds,
        gpu_slots=arguments.gpu_slots,
        cpu_slots=arguments.cpu_slots,
        first_settings_only=arguments.first_settings_only,
    ).run()


if __name__ == "__main__":
    main()
