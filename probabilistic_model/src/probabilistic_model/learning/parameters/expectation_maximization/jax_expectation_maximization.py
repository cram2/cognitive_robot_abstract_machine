"""
Expectation maximization of the parameters of a layered circuit with jax: one compiled
step accumulates the statistics of a batch, one compiled update maximizes.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import Callable

from probabilistic_model.learning.parameters.circuit_parameter_learning import LayerType
from probabilistic_model.learning.parameters.jax_circuit_parameter_learning import (
    CompiledStep,
    JaxCircuitParameterLearning,
    JaxCircuitParameterTraining,
    JaxTrainingState,
)
from probabilistic_model.learning.parameters.expectation_maximization.tensorized_expectation_maximization import (
    TensorizedExpectationMaximization,
    TensorizedExpectationMaximizationTraining,
)
from probabilistic_model.learning.parameters.training_schedule import (
    LearningRatePlateau,
    MiniBatches,
)
from probabilistic_model.learning.parameters.expectation_maximization.layer_maximization import (
    SufficientStatistics,
)

# %% training state


@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
class ExpectationMaximizationState(JaxTrainingState):
    """
    The values one compiled step of expectation maximization reads and returns, its loss
    the negative average log-likelihood of the last batch.
    """

    statistics: list[SufficientStatistics]
    """
    The sufficient statistics of every learned layer since the last update.
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

    def step(
        self, *, state: ExpectationMaximizationState, batch: tuple[jax.Array, ...]
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

    def update(
        self, *, state: ExpectationMaximizationState
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

    def step_and_update(
        self, *, state: ExpectationMaximizationState, batch: tuple[jax.Array, ...]
    ) -> ExpectationMaximizationState:
        """
        :param state: The state before the step.
        :param batch: The same rows of every training array.
        :return: The state after :meth:`step` on the batch and :meth:`update`.
        """
        return self.update(state=self.step(state=state, batch=batch))

    def initial_state(self) -> ExpectationMaximizationState:
        """
        :return: The state before the first step: the parameters before training and no
            statistics.
        """
        return ExpectationMaximizationState(
            parameters=self.initial_parameters,
            statistics=self.zero_statistics(parameters=self.initial_parameters),
            loss=self.loss_before_any_step(),
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
            learning=self,
        )


# %% training on mini-batches


@dataclass(kw_only=True)
class JaxExpectationMaximizationTraining(
    TensorizedExpectationMaximizationTraining[LayerType],
    JaxCircuitParameterTraining[LayerType],
):
    """
    Mini-batch training by :class:`JaxExpectationMaximization`, one compiled step per
    batch and one compiled update.
    """

    learning: JaxExpectationMaximization[LayerType]
    """
    The expectation maximization whose steps are taken.
    """

    compiled_step: CompiledStep = field(init=False)
    """
    :meth:`JaxExpectationMaximization.step`, compiled.
    """

    compiled_step_and_update: CompiledStep = field(init=False)
    """
    :meth:`JaxExpectationMaximization.step_and_update`, compiled.
    """

    compiled_update: Callable[..., ExpectationMaximizationState] = field(init=False)
    """
    :meth:`JaxExpectationMaximization.update`, compiled.
    """

    def __post_init__(self) -> None:
        """
        Compile the steps and the update.
        """
        super().__post_init__()
        self.compiled_step = CompiledStep.from_step(step=self.learning.step)
        self.compiled_step_and_update = CompiledStep.from_step(
            step=self.learning.step_and_update
        )
        self.compiled_update = jax.jit(self.learning.update)

    def accumulate_statistics(self, *, batch: tuple[jax.Array, ...]) -> jax.Array:
        return self.take_compiled_step(step=self.compiled_step, batch=batch)

    def accumulate_statistics_and_update(
        self, *, batch: tuple[jax.Array, ...]
    ) -> jax.Array:
        return self.take_compiled_step(step=self.compiled_step_and_update, batch=batch)

    def step_on_rows(self, *, rows: np.ndarray) -> jax.Array:
        """
        On a single training device, the rows are gathered by the compiled step itself,
        see :meth:`MiniBatchTraining.step_on_rows`.
        """
        return self.take_compiled_step_on_rows(
            step=(
                self.compiled_step_and_update
                if self.count_batch_toward_update()
                else self.compiled_step
            ),
            rows=rows,
        )

    def update_parameters(self) -> None:
        self.state = self.compiled_update(state=self.state)
