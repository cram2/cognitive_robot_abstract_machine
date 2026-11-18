"""
The learning methods of the circuit learning benchmark, each producing a layered circuit
from the training rows of a dataset in a computation setting, iterative methods training
until their loss converges.
"""

from __future__ import annotations

# %% imports
import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

import jax
import numpy as np
import optax
import pandas as pd
from krrood_compute_hardware.devices import Device
from krrood_compute_hardware.jax_memory import JaxDeviceMemory
from krrood_compute_hardware.memory import BudgetedMemory, DeviceMemory, HostMemory

from experiments.circuit_learning.datasets import BenchmarkDataset
from experiments.circuit_learning.settings import (
    ArrayLibrary,
    ComputationSetting,
    EdgeRepresentation,
)
from probabilistic_model.adapters.circuit_representations import CircuitRepresentations
from probabilistic_model.learning.expectation_maximization import (
    ExpectationMaximization,
    ExpectationMaximizationTraining,
    LearnableUnits,
)
from probabilistic_model.learning.gradient_descent import AverageNegativeLogLikelihood
from probabilistic_model.learning.jax_gradient_descent import JaxGradientDescent
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.learning.region_graph.region_graph import RegionGraph
from probabilistic_model.learning.torch_gradient_descent import TorchGradientDescent
from probabilistic_model.learning.training_schedule import (
    Convergence,
    ConvergenceHistory,
    LearningRatePlateau,
    MiniBatches,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)

# %% names


class LearningMethod(StrEnum):
    """
    A learning method of the benchmark, baselines included.
    """

    RANDOM_TENSORIZED_SUM_PRODUCT_NETWORK_GRADIENT_DESCENT = "RAT-SPN, gradient descent"
    RANDOM_TENSORIZED_SUM_PRODUCT_NETWORK_EXPECTATION_MAXIMIZATION = (
        "RAT-SPN, expectation maximization"
    )
    JOINT_PROBABILITY_TREE = "joint probability tree"


# %% training context


@dataclass(kw_only=True)
class LearnedCircuit:
    """
    A circuit a method learned and how its iterative training ended.
    """

    circuit: LayeredProbabilisticCircuit
    """
    The learned circuit, its parameters stored as the setting it was learned in says.
    """

    history: ConvergenceHistory | None
    """
    The losses of the epochs of the iterative training, ``None`` for methods without.
    """


@dataclass(kw_only=True)
class TrainingContext:
    """
    The limits a method trains within.
    """

    convergence: Convergence
    """
    When iterative training stops.
    """

    memory_budget_bytes: int
    """
    The most memory of its device a training step may allocate.
    """

    def device_memory(self, *, setting: ComputationSetting) -> DeviceMemory:
        """
        :param setting: The setting that trains.
        :return: The memory of its device within :attr:`memory_budget_bytes`.
        """
        memory = (
            JaxDeviceMemory(device=jax.devices(Device.GRAPHICS_PROCESSOR)[0])
            if setting.uses_graphics_processor
            else HostMemory()
        )
        return BudgetedMemory(memory=memory, budget_bytes=self.memory_budget_bytes)

    def rows_per_step(
        self,
        *,
        circuit: LayeredProbabilisticCircuit,
        setting: ComputationSetting,
        rows: np.ndarray,
    ) -> int:
        """
        :param circuit: The circuit, with host parameters.
        :param setting: The setting that trains.
        :param rows: The encoded training rows.
        :return: The most rows whose log-likelihood gradient fits in the memory budget,
            as XLA measures it for the circuit in jax with dense edges on the same
            device, at most all rows.
        """
        jax_setting = dataclasses.replace(
            setting,
            array_library=ArrayLibrary.JAX,
            edge_representation=EdgeRepresentation.DENSE,
        )
        placed = jax_setting.placed(circuit=circuit)
        step_memory = JaxGradientDescent(
            placed.root, AverageNegativeLogLikelihood()
        ).step_memory(jax_setting.asarray(rows=placed.encoded(rows)))
        batch_size = MiniBatches.fitting_in_memory(
            memory=self.device_memory(setting=setting), step_memory=step_memory
        ).batch_size
        return min(batch_size, len(rows))


# %% methods


@dataclass(kw_only=True)
class BenchmarkMethod(ABC):
    """
    A way of learning a circuit from the training rows of a dataset.
    """

    seed: int = 69
    """
    The seed of the structure, the initial parameters and the order of the rows.
    """

    @property
    @abstractmethod
    def name(self) -> LearningMethod:
        """
        :return: The name of the method.
        """

    @abstractmethod
    def settings(self) -> list[ComputationSetting]:
        """
        :return: The settings the method can learn in, the one it is measured in
            first leading.
        """

    def supports(self, *, dataset: BenchmarkDataset) -> bool:
        """
        :param dataset: A dataset.
        :return: Whether the method learns from the kinds of variables of the dataset.
        """
        return True

    @abstractmethod
    def learn(
        self,
        *,
        dataset: BenchmarkDataset,
        setting: ComputationSetting,
        context: TrainingContext,
    ) -> LearnedCircuit:
        """
        :param dataset: The dataset, of which only the training and validation rows are
            read.
        :param setting: The setting to learn in, one of :meth:`settings`.
        :param context: The limits of training.
        :return: The learned circuit.
        """


@dataclass(kw_only=True)
class RandomTensorizedSumProductNetworkLearning(BenchmarkMethod, ABC):
    """
    A way of learning the parameters of a RAT-SPN :cite:`peharz2020random`, built from a
    random region graph.
    """

    depth: int = 3
    """
    The number of splits of a region graph repetition, at most the base two logarithm of
    the number of variables.
    """

    repetitions: int = 10
    """
    The number of random repetitions of the region graph.
    """

    units: int = 10
    """
    The number of input distributions of a leaf region and sum units of a region.
    """

    def random_circuit(
        self, *, dataset: BenchmarkDataset
    ) -> LayeredProbabilisticCircuit:
        """
        :param dataset: The dataset.
        :return: The randomly initialized RAT-SPN over the variables of the dataset.
        """
        depth = min(self.depth, max(int(np.log2(len(dataset.variables))), 1))
        return (
            RegionGraph(
                variables=dataset.variables,
                partitions=2,
                depth=depth,
                repetitions=self.repetitions,
                seed=self.seed,
            )
            .create_random_region_graph()
            .to_probabilistic_circuit(input_units=self.units, sum_units=self.units)
        )


@dataclass(kw_only=True)
class RandomTensorizedSumProductNetworkByGradientDescent(
    RandomTensorizedSumProductNetworkLearning
):
    """
    A RAT-SPN trained generatively by mini-batch gradient descent with Adam, whose
    learning rate is reduced on plateaus of the training loss.
    """

    learning_rate: float = 1e-2
    """
    The initial learning rate of Adam.
    """

    batch_size: int = 1024
    """
    The most rows per step, fewer if they do not fit in memory; the largest batch size
    :cite:`liu2025rethinking` tune Adam over.
    """

    learning_rate_plateau: LearningRatePlateau = field(
        default_factory=LearningRatePlateau
    )
    """
    The schedule of the learning rate, copied for every training.
    """

    @property
    def name(self) -> LearningMethod:
        return LearningMethod.RANDOM_TENSORIZED_SUM_PRODUCT_NETWORK_GRADIENT_DESCENT

    def settings(self) -> list[ComputationSetting]:
        return ComputationSetting.grid(
            array_libraries=(ArrayLibrary.JAX, ArrayLibrary.TORCH)
        )

    def learn(
        self,
        *,
        dataset: BenchmarkDataset,
        setting: ComputationSetting,
        context: TrainingContext,
    ) -> LearnedCircuit:
        host = self.random_circuit(dataset=dataset)
        rows = dataset.encoded(frame=dataset.train)
        mini_batches = MiniBatches(
            batch_size=min(
                self.batch_size,
                context.rows_per_step(circuit=host, setting=setting, rows=rows),
            ),
            seed=self.seed,
        )
        circuit = setting.placed(circuit=host)
        data = setting.asarray(rows=circuit.encoded(rows))
        if setting.array_library is ArrayLibrary.TORCH:
            trainer = TorchGradientDescent(
                circuit.root,
                AverageNegativeLogLikelihood(),
                learning_rate=self.learning_rate,
            )
        else:
            trainer = JaxGradientDescent(
                circuit.root,
                AverageNegativeLogLikelihood(),
                optax.inject_hyperparams(optax.adam)(learning_rate=self.learning_rate),
            )
        run = trainer.minimize_until_converged(
            data,
            convergence=context.convergence,
            mini_batches=mini_batches,
            learning_rate_plateau=dataclasses.replace(self.learning_rate_plateau),
        )
        circuit.root = run.trained_root
        return LearnedCircuit(circuit=circuit, history=run.history)


@dataclass(kw_only=True)
class RandomTensorizedSumProductNetworkByExpectationMaximization(
    RandomTensorizedSumProductNetworkLearning
):
    """
    A RAT-SPN trained by full-batch expectation maximization on its rustworkx form, on
    the host, until the training log-likelihood converges.
    """

    expectation_maximization: ExpectationMaximization = field(
        default_factory=ExpectationMaximization
    )
    """
    The expectation maximization the parameters are learned with.
    """

    @property
    def name(self) -> LearningMethod:
        return LearningMethod.RANDOM_TENSORIZED_SUM_PRODUCT_NETWORK_EXPECTATION_MAXIMIZATION

    def settings(self) -> list[ComputationSetting]:
        return [ComputationSetting.host()]

    def learn(
        self,
        *,
        dataset: BenchmarkDataset,
        setting: ComputationSetting,
        context: TrainingContext,
    ) -> LearnedCircuit:
        representations = CircuitRepresentations()
        circuit = representations.convert(
            self.random_circuit(dataset=dataset), ProbabilisticCircuit
        )
        rows = dataset.train[
            [variable.name for variable in circuit.variables]
        ].to_numpy()
        history = context.convergence.history_until_converged(
            training=ExpectationMaximizationTraining(
                arguments=(rows,),
                mini_batches=MiniBatches(),
                circuit=circuit,
                learnable_units=LearnableUnits.from_units(circuit.nodes()),
                expectation_maximization=self.expectation_maximization,
            )
        )
        return LearnedCircuit(
            circuit=setting.placed(
                circuit=representations.convert(circuit, LayeredProbabilisticCircuit)
            ),
            history=history,
        )


@dataclass(kw_only=True)
class JointProbabilityTreeBaseline(BenchmarkMethod):
    """
    A joint probability tree :cite:`nyga2023joint`.
    """

    minimum_samples_per_leaf: float = 0.1
    """
    The fraction of the training rows a leaf holds at least.
    """

    @property
    def name(self) -> LearningMethod:
        return LearningMethod.JOINT_PROBABILITY_TREE

    def settings(self) -> list[ComputationSetting]:
        return [ComputationSetting.host()]

    def learn(
        self,
        *,
        dataset: BenchmarkDataset,
        setting: ComputationSetting,
        context: TrainingContext,
    ) -> LearnedCircuit:
        variables = infer_variables_from_dataframe(
            pd.concat([dataset.train, dataset.validation, dataset.test])
        )
        tree = JointProbabilityTree(
            minimum_samples_per_leaf=self.minimum_samples_per_leaf
        ).fit(dataset.train, variables)
        return LearnedCircuit(
            circuit=CircuitRepresentations().convert(tree, LayeredProbabilisticCircuit),
            history=None,
        )


# %% all methods


@dataclass(kw_only=True)
class BenchmarkMethods:
    """
    Every method of the benchmark in its default configuration.
    """

    methods: list[BenchmarkMethod] = field(
        default_factory=lambda: [
            RandomTensorizedSumProductNetworkByGradientDescent(),
            RandomTensorizedSumProductNetworkByExpectationMaximization(),
            JointProbabilityTreeBaseline(),
        ]
    )
    """
    The methods.
    """

    def find(self, *, name: LearningMethod) -> BenchmarkMethod:
        """
        :param name: The name of a method.
        :return: The method of that name.
        """
        return next(method for method in self.methods if method.name is name)
