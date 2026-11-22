"""
Trains the mixture of products with jax on mini-batches and prints what the test of
splitting batches across devices compares, as one JSON object.

It runs in a process of its own, since jax fixes its number of CPU devices when it
starts.
"""

from __future__ import annotations

# %% imports
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
from random_events.variable import Continuous

from probabilistic_model.learning.gradient_descent import AverageNegativeLogLikelihood
from probabilistic_model.learning.jax_gradient_descent import JaxGradientDescent
from probabilistic_model.learning.training_schedule import MiniBatches
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.edge_matrix import (
    BlockEdgeMatrix,
)

from .test_inner_layers_on_array_backends import MixtureOfProductsCase

# %% training


def main() -> None:
    """
    Train four epochs of batches of 100 rows of 450 sampled rows, so that the last batch
    of every epoch has a number of rows no number of devices above two divides, and
    print the trained parameters and where the arrays lived.
    """
    np.random.seed(0)
    host = MixtureOfProductsCase(Continuous("x"), Continuous("y")).circuit(
        np, BlockEdgeMatrix
    )
    circuit = host.to_namespace(jnp)
    data = jnp.asarray(host.sample(450))
    gradient_descent = JaxGradientDescent(
        layer=circuit.root,
        loss=AverageNegativeLogLikelihood(),
        optimizer=optax.adam(5e-2),
    )
    training = gradient_descent.start_training(
        arguments=(data,), mini_batches=MiniBatches(batch_size=100, seed=0)
    )
    epoch_losses = [training.train_one_epoch() for _ in range(4)]
    trained = training.trained_root().circuit_parameters
    print(
        json.dumps(
            {
                "devices": len(jax.devices()),
                "devices_of_state": len(training.state.parameters[0].devices()),
                "devices_of_trained_parameters": len(trained[0].devices()),
                "even_batch_size": gradient_descent.largest_even_batch_size(rows=450),
                "epoch_losses": epoch_losses,
                "parameters": [np.asarray(p).tolist() for p in trained],
            }
        )
    )


if __name__ == "__main__":
    main()
