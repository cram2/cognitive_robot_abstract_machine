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
import numpy.typing as npt
import optax
import pandas as pd
from typing_extensions import Self
from krrood_compute_resources.devices import Device
from krrood_compute_resources.jax_memory import JaxDeviceMemory
from krrood_compute_resources.memory import BudgetedMemory, DeviceMemory, HostMemory

from experiments.circuit_learning.datasets import BenchmarkDataset
from experiments.circuit_learning.settings import (
    ArrayLibrary,
    ComputationSetting,
)
from probabilistic_model.adapters.circuit_representations import CircuitRepresentations
from probabilistic_model.learning.parameters.expectation_maximization.expectation_maximization import (
    ExpectationMaximization,
    ExpectationMaximizationTraining,
    LearnableUnits,
    MaximizationRegularization,
)
from probabilistic_model.learning.parameters.circuit_parameter_learning import (
    CircuitParameterLearning,
    TrainingRun,
)
from probabilistic_model.learning.parameters.gradient_descent.gradient_descent import (
    AverageNegativeLogLikelihood,
)
from probabilistic_model.learning.parameters.expectation_maximization.jax_expectation_maximization import (
    JaxExpectationMaximization,
)
from probabilistic_model.learning.parameters.gradient_descent.jax_gradient_descent import (
    JaxGradientDescent,
)
from probabilistic_model.learning.structure.hidden_chow_liu_tree.hidden_chow_liu_tree import (
    HiddenChowLiuTree,
)
from probabilistic_model.learning.structure.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.structure.mutual_information import (
    BinnedMutualInformation,
    MutualInformationEstimator,
)
from probabilistic_model.learning.structure.jpt.variables import (
    infer_variables_from_dataframe,
)
from probabilistic_model.learning.structure.region_graph.region_graph import RegionGraph
from probabilistic_model.learning.parameters.expectation_maximization.torch_expectation_maximization import (
    TorchExpectationMaximization,
)
from probabilistic_model.learning.parameters.gradient_descent.torch_gradient_descent import (
    TorchGradientDescent,
)
from probabilistic_model.learning.parameters.training_schedule import (
    Convergence,
    ConvergenceHistory,
    LearningRatePlateau,
    MiniBatches,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import (
    Layer,
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
    RANDOM_TENSORIZED_SUM_PRODUCT_NETWORK_ONLINE_EXPECTATION_MAXIMIZATION = (
        "RAT-SPN, online expectation maximization"
    )
    JOINT_PROBABILITY_TREE = "joint probability tree"
    HIDDEN_CHOW_LIU_TREE = "hidden Chow-Liu tree"


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
        rows: npt.NDArray[np.float64],
    ) -> int:
        """
        :param circuit: The circuit, with host parameters.
        :param setting: The setting that trains.
        :param rows: The encoded training rows.
        :return: The most rows whose log-likelihood gradient fits in the memory budget,
            as XLA measures it for the circuit in jax on the setting's device with the
            setting's edges, at most all rows.
        """
        jax_setting = dataclasses.replace(setting, array_library=ArrayLibrary.JAX)
        placed = jax_setting.placed(circuit=circuit)
        step_memory = JaxGradientDescent(
            layer=placed.root, loss=AverageNegativeLogLikelihood()
        ).step_memory(jax_setting.asarray(rows=placed.encoded(rows)))
        batch_size = MiniBatches.fitting_in_memory(
            memory=self.device_memory(setting=setting), step_memory=step_memory
        ).batch_size
        return min(batch_size, len(rows))

    def train_until_converged(
        self,
        *,
        learner: CircuitParameterLearning,
        circuit: LayeredProbabilisticCircuit,
        setting: ComputationSetting,
        rows: npt.NDArray[np.float64],
        most_rows_per_step: int,
        seed: int,
    ) -> TrainingRun:
        """
        :param learner: The learner of the circuit's root.
        :param circuit: The circuit placed as the setting says.
        :param setting: The setting that trains.
        :param rows: The training rows, as the datasets encode them.
        :param most_rows_per_step: The most rows a step may train on.
        :param seed: The seed of the order of the rows.
        :return: The run of shuffled mini-batches until :attr:`convergence` stops it.
        """
        mini_batches = MiniBatches(
            batch_size=learner.largest_even_batch_size(rows=most_rows_per_step),
            seed=seed,
        )
        return learner.minimize_until_converged(
            setting.asarray(rows=circuit.encoded(rows)),
            convergence=self.convergence,
            mini_batches=mini_batches,
        )


# %% expectation maximization phases


@dataclass(kw_only=True)
class ExpectationMaximizationPhase:
    """
    Tensorized expectation maximization with one schedule of updates, trained until the
    training context's convergence stops it.
    """

    batch_size: int | None = None
    """
    The most rows per step, fewer if they do not fit in memory; as many as fit if
    ``None``.
    """

    batches_per_update: int | None = None
    """
    The batches after which the parameters are updated, once per epoch if ``None``.
    """

    step_size: float = 1.0
    """
    The share of the way to the maximizing parameters that an update moves.
    """

    regularization: MaximizationRegularization = field(
        default_factory=MaximizationRegularization
    )
    """
    The regularization of the maximizing parameters.
    """

    @classmethod
    def mini_batch(cls) -> Self:
        """
        :return: The mini-batch expectation maximization of :cite:`peharz2020einsum`: a
            step of a tenth of the way after every batch of 512 rows.
        """
        return cls(batch_size=512, batches_per_update=1, step_size=0.1)

    def learner(
        self, *, root: Layer, setting: ComputationSetting
    ) -> CircuitParameterLearning:
        """
        :param root: The root of the circuit placed as the setting says.
        :param setting: The setting that trains, with jax or torch arrays.
        :return: The expectation maximization of the setting's array library.
        """
        if setting.array_library is ArrayLibrary.TORCH:
            return TorchExpectationMaximization(
                layer=root,
                batches_per_update=self.batches_per_update,
                step_size=self.step_size,
                regularization=self.regularization,
                records_cuda_graphs=True,
            )
        return JaxExpectationMaximization(
            layer=root,
            batches_per_update=self.batches_per_update,
            step_size=self.step_size,
            regularization=self.regularization,
        )


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
class RandomTensorizedSumProductNetworkByMiniBatches(
    RandomTensorizedSumProductNetworkLearning, ABC
):
    """
    A RAT-SPN whose parameters a learner of the setting's array library trains on
    shuffled mini-batches until the training loss converges.
    """

    batch_size: int = 1024
    """
    The most rows per step, fewer if they do not fit in memory.
    """

    @abstractmethod
    def learner(
        self, *, root: Layer, setting: ComputationSetting
    ) -> CircuitParameterLearning:
        """
        :param root: The root of the circuit placed as the setting says.
        :param setting: The setting that trains.
        :return: The learner that trains the circuit in the setting.
        """
        raise NotImplementedError

    def learn(
        self,
        *,
        dataset: BenchmarkDataset,
        setting: ComputationSetting,
        context: TrainingContext,
    ) -> LearnedCircuit:
        host = self.random_circuit(dataset=dataset)
        rows = dataset.encoded(frame=dataset.train)
        circuit = setting.placed(circuit=host)
        run = context.train_until_converged(
            learner=self.learner(root=circuit.root, setting=setting),
            circuit=circuit,
            setting=setting,
            rows=rows,
            most_rows_per_step=min(
                self.batch_size,
                context.rows_per_step(circuit=host, setting=setting, rows=rows),
            ),
            seed=self.seed,
        )
        circuit.root = run.trained_root
        return LearnedCircuit(circuit=circuit, history=run.history)


@dataclass(kw_only=True)
class RandomTensorizedSumProductNetworkByGradientDescent(
    RandomTensorizedSumProductNetworkByMiniBatches
):
    """
    A RAT-SPN trained generatively by mini-batch gradient descent with Adam, whose
    learning rate is reduced on plateaus of the training loss, on batches of at most the
    largest size :cite:`liu2025rethinking` tune Adam over.
    """

    learning_rate: float = 1e-2
    """
    The initial learning rate of Adam.
    """

    learning_rate_plateau: LearningRatePlateau = field(
        default_factory=LearningRatePlateau
    )
    """
    The schedule of the learning rate.
    """

    @property
    def name(self) -> LearningMethod:
        return LearningMethod.RANDOM_TENSORIZED_SUM_PRODUCT_NETWORK_GRADIENT_DESCENT

    def settings(self) -> list[ComputationSetting]:
        return ComputationSetting.grid(
            array_libraries=(ArrayLibrary.JAX, ArrayLibrary.TORCH)
        )

    def learner(
        self, *, root: Layer, setting: ComputationSetting
    ) -> CircuitParameterLearning:
        if setting.array_library is ArrayLibrary.TORCH:
            return TorchGradientDescent(
                layer=root,
                loss=AverageNegativeLogLikelihood(),
                learning_rate=self.learning_rate,
                learning_rate_plateau=self.learning_rate_plateau,
                records_cuda_graphs=True,
            )
        return JaxGradientDescent(
            layer=root,
            loss=AverageNegativeLogLikelihood(),
            optimizer=optax.inject_hyperparams(optax.adam)(
                learning_rate=self.learning_rate
            ),
            learning_rate_plateau=self.learning_rate_plateau,
        )


@dataclass(kw_only=True)
class RandomTensorizedSumProductNetworkByExpectationMaximization(
    RandomTensorizedSumProductNetworkByMiniBatches
):
    """
    A RAT-SPN trained by full-batch expectation maximization until the training
    log-likelihood converges: on its rustworkx form in the host setting, tensorized on
    mini-batches whose statistics accumulate over every epoch in the other settings.
    """

    expectation_maximization: ExpectationMaximization = field(
        default_factory=ExpectationMaximization
    )
    """
    The host expectation maximization, whose regularization the tensorized one shares.
    """

    batches_per_update: int | None = None
    """
    The batches after which the tensorized parameters are updated, once per epoch if
    ``None``.
    """

    step_size: float = 1.0
    """
    The share of the way to the maximizing parameters that a tensorized update moves.
    """

    @property
    def name(self) -> LearningMethod:
        return LearningMethod.RANDOM_TENSORIZED_SUM_PRODUCT_NETWORK_EXPECTATION_MAXIMIZATION

    def settings(self) -> list[ComputationSetting]:
        return [ComputationSetting.host()] + ComputationSetting.grid(
            array_libraries=(ArrayLibrary.JAX, ArrayLibrary.TORCH)
        )

    def learner(
        self, *, root: Layer, setting: ComputationSetting
    ) -> CircuitParameterLearning:
        return ExpectationMaximizationPhase(
            batches_per_update=self.batches_per_update,
            step_size=self.step_size,
            regularization=self.expectation_maximization.regularization,
        ).learner(root=root, setting=setting)

    def learn(
        self,
        *,
        dataset: BenchmarkDataset,
        setting: ComputationSetting,
        context: TrainingContext,
    ) -> LearnedCircuit:
        """
        Like :meth:`RandomTensorizedSumProductNetworkByMiniBatches.learn`, but in the
        host setting on the rustworkx form of the circuit, all rows in one batch.
        """
        if setting.array_library is not ArrayLibrary.NUMPY:
            return super().learn(dataset=dataset, setting=setting, context=context)
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
class RandomTensorizedSumProductNetworkByOnlineExpectationMaximization(
    RandomTensorizedSumProductNetworkByExpectationMaximization
):
    """
    A RAT-SPN trained by online expectation maximization, an update after every batch
    with the step size of the experiments of Einsum Networks :cite:`peharz2020einsum`.
    """

    batches_per_update: int | None = 1
    """
    An update after every batch.
    """

    step_size: float = 0.05
    """
    The step size of the online expectation maximization of Einsum Networks.
    """

    @property
    def name(self) -> LearningMethod:
        """
        The online variant, see
        :attr:`RandomTensorizedSumProductNetworkByExpectationMaximization.name`.
        """
        return LearningMethod.RANDOM_TENSORIZED_SUM_PRODUCT_NETWORK_ONLINE_EXPECTATION_MAXIMIZATION

    def settings(self) -> list[ComputationSetting]:
        """
        The tensorized settings alone, since the host learns all rows in one batch, see
        :meth:`RandomTensorizedSumProductNetworkByExpectationMaximization.settings`.
        """
        return ComputationSetting.grid(
            array_libraries=(ArrayLibrary.JAX, ArrayLibrary.TORCH)
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
            circuit=setting.placed(
                circuit=CircuitRepresentations().convert(
                    tree, LayeredProbabilisticCircuit
                )
            ),
            history=None,
        )


@dataclass(kw_only=True)
class HiddenChowLiuTreeByExpectationMaximization(BenchmarkMethod):
    """
    A hidden Chow-Liu tree circuit :cite:`liu2021tractable` trained by mini-batch and
    then full-batch expectation maximization, each phase until the training loss
    converges.
    """

    hidden_states: int = 12
    """
    The number of states of every latent variable.
    """

    mutual_information: MutualInformationEstimator = field(
        default_factory=BinnedMutualInformation
    )
    """
    The estimator of the mutual information the tree is spanned over.
    """

    phases: list[ExpectationMaximizationPhase] = field(
        default_factory=lambda: [
            ExpectationMaximizationPhase.mini_batch(),
            ExpectationMaximizationPhase(),
        ]
    )
    """
    The phases of training, in order.
    """

    @property
    def name(self) -> LearningMethod:
        return LearningMethod.HIDDEN_CHOW_LIU_TREE

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
        rows = dataset.encoded(frame=dataset.train)
        host = HiddenChowLiuTree(
            variables=dataset.variables,
            hidden_states=self.hidden_states,
            mutual_information=self.mutual_information,
            seed=self.seed,
        ).build_circuit(data=rows)
        circuit = setting.placed(circuit=host)
        rows_per_step = context.rows_per_step(circuit=host, setting=setting, rows=rows)
        histories = []
        for phase in self.phases:
            run = context.train_until_converged(
                learner=phase.learner(root=circuit.root, setting=setting),
                circuit=circuit,
                setting=setting,
                rows=rows,
                most_rows_per_step=min(phase.batch_size or len(rows), rows_per_step),
                seed=self.seed,
            )
            circuit.root = run.trained_root
            histories.append(run.history)
        return LearnedCircuit(
            circuit=circuit,
            history=ConvergenceHistory.from_consecutive(histories=histories),
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
            RandomTensorizedSumProductNetworkByOnlineExpectationMaximization(),
            JointProbabilityTreeBaseline(),
            HiddenChowLiuTreeByExpectationMaximization(),
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
