"""
Expectation maximization of the parameters of a layered circuit with jax: one compiled
step accumulates the statistics of a batch, one compiled update maximizes.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from krrood.patterns.called_positionally import called_positionally
from typing_extensions import Callable

from probabilistic_model.learning.circuit_parameter_learning import LayerType
from probabilistic_model.learning.jax_circuit_parameter_learning import (
    BatchSplitDevices,
    JaxCircuitParameterLearning,
)
from probabilistic_model.learning.tensorized_expectation_maximization import (
    TensorizedExpectationMaximization,
    TensorizedExpectationMaximizationTraining,
)
from probabilistic_model.learning.training_schedule import (
    LearningRatePlateau,
    MiniBatches,
)
from probabilistic_model.probabilistic_circuit.tensorized.expectation_maximization_parameters import (
    SufficientStatistics,
)

# %% training state


@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
class ExpectationMaximizationState:
    """
    The values one compiled step of expectation maximization reads and returns.
    """

    parameters: list[jax.Array]
    """
    The parameter arrays of the circuit, in the order of its ``circuit_parameters``.
    """

    statistics: list[SufficientStatistics]
    """
    The sufficient statistics of every learned layer since the last update.
    """

    loss: jax.Array
    """
    The negative average log-likelihood of the last batch before its step.
    """


# %% expectation maximization


@dataclass(kw_only=True)
class JaxExpectationMaximization(
    TensorizedExpectationMaximization[LayerType], JaxCircuitParameterLearning[LayerType]
):
    """
    Expectation maximization with jax, one compiled step per batch.
    """

    def __post_init__(self) -> None:
        """
        Make the statistics of every learned layer a jax pytree.
        """
        super().__post_init__()
        for statistics in self.zero_statistics(parameters=self.initial_parameters):
            [first_leaf, *_] = jax.tree_util.tree_leaves(statistics)
            if first_leaf is statistics:
                self.register_as_pytree(statistics_type=type(statistics))

    @staticmethod
    def register_as_pytree(*, statistics_type: type[SufficientStatistics]) -> None:
        """
        Let jax trace and place statistics of a type as the list of their arrays.

        :param statistics_type: A type of statistics that jax does not know yet.
        """
        jax.tree_util.register_pytree_node(
            statistics_type,
            lambda statistics: (statistics.arrays, None),
            lambda _, arrays: statistics_type.from_arrays(arrays=list(arrays)),
        )

    def zero_node_counts(self) -> list[jax.Array]:
        """
        :return: A zero for every node of every learned layer, in the floating point
            type of the parameters.
        """
        floating_point_type = jnp.result_type(*self.initial_parameters)
        return [
            jnp.zeros(layer.number_of_nodes, dtype=floating_point_type)
            for layer in self.learned_layers(root=self.layer)
        ]

    def zero_statistics(
        self, *, parameters: list[jax.Array]
    ) -> list[SufficientStatistics]:
        """
        :param parameters: Arrays in the order of the circuit's parameters.
        :return: The statistics of no rows, for every learned layer.
        """
        return self.statistics_from_gradients(
            parameters=parameters,
            parameter_gradients=[jnp.zeros_like(parameter) for parameter in parameters],
            node_counts=self.zero_node_counts(),
        )

    @called_positionally
    def step(
        self, state: ExpectationMaximizationState, *batch: jax.Array
    ) -> ExpectationMaximizationState:
        [data] = batch
        total, (parameter_gradients, node_counts) = jax.value_and_grad(
            lambda parameters, node_offsets: self.total_log_likelihood(
                parameters=parameters, node_offsets=node_offsets, data=data
            ),
            argnums=(0, 1),
        )(state.parameters, self.zero_node_counts())
        batch_statistics = self.statistics_from_gradients(
            parameters=state.parameters,
            parameter_gradients=parameter_gradients,
            node_counts=node_counts,
        )
        return ExpectationMaximizationState(
            parameters=state.parameters,
            statistics=[
                accumulated.added(other=added)
                for accumulated, added in zip(state.statistics, batch_statistics)
            ],
            loss=-total / data.shape[0],
        )

    @called_positionally
    def update(
        self, state: ExpectationMaximizationState
    ) -> ExpectationMaximizationState:
        """
        :param state: The state after the last step.
        :return: The state with the maximized parameters and no statistics.
        """
        return ExpectationMaximizationState(
            parameters=self.maximized_circuit_parameters(
                parameters=state.parameters, statistics=state.statistics
            ),
            statistics=jax.tree.map(jnp.zeros_like, state.statistics),
            loss=state.loss,
        )

    @called_positionally
    def step_and_update(
        self, state: ExpectationMaximizationState, *batch: jax.Array
    ) -> ExpectationMaximizationState:
        """
        :param state: The state before the step.
        :param batch: The same rows of every training array.
        :return: The state after :meth:`step` on the batch and :meth:`update`.
        """
        return self.update(self.step(state, *batch))

    def initial_state(self) -> ExpectationMaximizationState:
        """
        :return: The state before the first step: the parameters before training and no
            statistics.
        """
        return ExpectationMaximizationState(
            parameters=self.initial_parameters,
            statistics=self.zero_statistics(parameters=self.initial_parameters),
            loss=jnp.full((), jnp.nan, jnp.result_type(*self.initial_parameters)),
        )

    def start_training(
        self,
        *,
        arguments: tuple[jax.Array, ...],
        mini_batches: MiniBatches,
        learning_rate_plateau: LearningRatePlateau | None = None,
    ) -> JaxExpectationMaximizationTraining[LayerType]:
        return JaxExpectationMaximizationTraining(
            arguments=arguments,
            mini_batches=mini_batches,
            batches_per_update=self.batches_per_update,
            expectation_maximization=self,
        )


# %% training on mini-batches


@dataclass(kw_only=True)
class JaxExpectationMaximizationTraining(
    TensorizedExpectationMaximizationTraining[LayerType]
):
    """
    Mini-batch training by :class:`JaxExpectationMaximization`, one compiled step per
    batch and one compiled update.
    """

    expectation_maximization: JaxExpectationMaximization[LayerType]
    """
    The expectation maximization whose steps are taken.
    """

    compiled_step: Callable[..., ExpectationMaximizationState] = field(init=False)
    """
    :meth:`JaxExpectationMaximization.step`, compiled once for every shape of batch.
    """

    compiled_step_and_update: Callable[..., ExpectationMaximizationState] = field(
        init=False
    )
    """
    :meth:`JaxExpectationMaximization.step_and_update`, compiled once for every shape of
    batch.
    """

    compiled_update: Callable[..., ExpectationMaximizationState] = field(init=False)
    """
    :meth:`JaxExpectationMaximization.update`, compiled.
    """

    state: ExpectationMaximizationState = field(init=False)
    """
    The state after the last step or update.
    """

    batch_split_devices: BatchSplitDevices | None = field(init=False, default=None)
    """
    The devices that the rows of a batch are split across and the state is copied to,
    ``None`` if there is only one.
    """

    def __post_init__(self) -> None:
        """
        Compile the step and the update and start from the state before training, copied
        to every training device.
        """
        self.compiled_step = jax.jit(self.expectation_maximization.step)
        self.compiled_step_and_update = jax.jit(
            self.expectation_maximization.step_and_update
        )
        self.compiled_update = jax.jit(self.expectation_maximization.update)
        self.state = self.expectation_maximization.initial_state()
        self.batch_split_devices = BatchSplitDevices.from_devices(
            devices=self.expectation_maximization.training_devices
        )
        if self.batch_split_devices is not None:
            self.state = self.batch_split_devices.replicated(state=self.state)

    def accumulate_statistics(self, *, batch: tuple[jax.Array, ...]) -> jax.Array:
        self.state = self.compiled_step(self.state, *self.placed_batch(batch=batch))
        return self.state.loss

    def accumulate_statistics_and_update(
        self, *, batch: tuple[jax.Array, ...]
    ) -> jax.Array:
        self.state = self.compiled_step_and_update(
            self.state, *self.placed_batch(batch=batch)
        )
        return self.state.loss

    def placed_batch(self, *, batch: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        """
        :param batch: The same rows of every training array.
        :return: The batch as :attr:`batch_split_devices` place it, as it is without
            them.
        """
        if self.batch_split_devices is None:
            return batch
        return self.batch_split_devices.placed_batch(batch=batch)

    def update_parameters(self) -> None:
        self.state = self.compiled_update(self.state)

    def trained_root(self) -> LayerType:
        return self.expectation_maximization.layer.copy_with_circuit_parameters(
            parameters=jax.device_put(
                self.state.parameters, self.expectation_maximization.parameter_device
            )
        )
