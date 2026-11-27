"""
Tests that a layered probabilistic circuit stored in a database is loaded back as an
equivalent circuit.
"""

from __future__ import annotations

import collections

import jax.numpy as jnp
import numpy as np
import optax
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from krrood.ormatic.data_access_objects.helper import to_dao
from probabilistic_model.learning.parameters.gradient_descent.jax_gradient_descent import (
    TrainingState,
)
from probabilistic_model.learning.structure.region_graph.region_graph import RegionGraph
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)

from ..test_region_graphs.test_region_graph import continuous_variables

# %% circuits


@pytest.fixture
def circuit() -> LayeredProbabilisticCircuit:
    """
    The layered circuit of a random region graph over six continuous variables, in which
    a layer is a child of several layers.
    """
    return (
        RegionGraph(
            variables=continuous_variables(6), partitions=2, depth=2, repetitions=3
        )
        .create_random_region_graph()
        .to_probabilistic_circuit(input_units=4, sum_units=3)
    )


def store_and_load(
    *, circuit: LayeredProbabilisticCircuit, session: Session
) -> LayeredProbabilisticCircuit:
    """
    :param circuit: The circuit to store.
    :param session: The session of the database to store it in.
    :return: The circuit loaded back from the database.
    """
    from probabilistic_model.orm.ormatic_interface import (
        LayeredProbabilisticCircuitDAO,
    )

    session.add(to_dao(circuit))
    session.commit()
    return session.scalars(select(LayeredProbabilisticCircuitDAO)).one().from_dao()


# %% round trip


def test_a_loaded_circuit_has_the_log_likelihoods_of_the_stored_one(
    circuit: LayeredProbabilisticCircuit, session: Session
):
    """
    Storing and loading a circuit keeps the log-likelihood of every row.
    """
    data = np.random.default_rng(0).normal(size=(50, 6))

    loaded = store_and_load(circuit=circuit, session=session)

    np.testing.assert_array_equal(
        loaded.log_likelihood(events=data), circuit.log_likelihood(events=data)
    )


def test_a_loaded_circuit_has_the_variables_of_the_stored_one(
    circuit: LayeredProbabilisticCircuit, session: Session
):
    """
    Storing and loading a circuit keeps its variables, in a sorted set.
    """
    loaded = store_and_load(circuit=circuit, session=session)

    assert type(loaded.variables) is type(circuit.variables)
    assert list(loaded.variables) == list(circuit.variables)


def test_a_layer_with_several_parents_is_stored_once(
    circuit: LayeredProbabilisticCircuit, session: Session
):
    """
    A layer that several layers point to gets one row, so the database holds as many
    layers as the circuit, and so does the loaded circuit.
    """
    from probabilistic_model.orm.ormatic_interface import LayerDAO

    parents_per_layer = collections.Counter(
        id(child) for layer in circuit.layers for child in layer.child_layers
    )
    assert max(parents_per_layer.values()) > 1

    loaded = store_and_load(circuit=circuit, session=session)

    assert session.scalar(select(func.count()).select_from(LayerDAO)) == len(
        circuit.layers
    )
    assert len(loaded.layers) == len(circuit.layers)


def test_parameters_of_another_backend_are_loaded_as_numpy_arrays(
    circuit: LayeredProbabilisticCircuit, session: Session
):
    """
    Parameters of another array backend are stored with their values and loaded as numpy
    arrays.
    """
    jax_circuit = circuit.to_namespace(jnp, floating_point_type=jnp.result_type(float))

    loaded = store_and_load(circuit=jax_circuit, session=session)

    assert all(isinstance(parameter, np.ndarray) for parameter in loaded.parameters)
    for loaded_parameter, parameter in zip(loaded.parameters, jax_circuit.parameters):
        np.testing.assert_array_equal(loaded_parameter, np.asarray(parameter))


# %% training state


def test_a_jax_training_state_is_loaded_with_its_parameters_and_loss(session: Session):
    """
    The parameters and the loss of a jax training state are stored and loaded back as
    numpy arrays of the same values.
    """
    from probabilistic_model.orm.ormatic_interface import TrainingStateDAO

    parameters = [jnp.ones(3), jnp.arange(2.0)]
    state = TrainingState(
        parameters=parameters,
        loss=jnp.asarray(1.5),
        optimizer_state=optax.adam(learning_rate=0.1).init(parameters),
    )

    session.add(to_dao(state))
    session.commit()
    loaded = session.scalars(select(TrainingStateDAO)).one().from_dao()

    for loaded_parameter, parameter in zip(loaded.parameters, parameters, strict=True):
        np.testing.assert_array_equal(loaded_parameter, np.asarray(parameter))
    np.testing.assert_array_equal(loaded.loss, np.asarray(state.loss))
