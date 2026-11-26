"""
Trains the mixture of products with jax on mini-batches and writes what the test of
splitting batches across devices compares to a JSON file.

It runs in a process of its own, since jax fixes its number of CPU devices when it
starts.
"""

from __future__ import annotations

# %% imports
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from krrood.adapters.json_serializer import to_json
from random_events.variable import Continuous

from probabilistic_model.learning.parameters.gradient_descent.gradient_descent import (
    AverageNegativeLogLikelihood,
)
from probabilistic_model.learning.parameters.jax_circuit_parameter_learning import (
    JaxPlatform,
)
from probabilistic_model.learning.parameters.gradient_descent.jax_gradient_descent import (
    JaxGradientDescent,
)
from probabilistic_model.learning.parameters.training_schedule import MiniBatches
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.edge_matrix import (
    BlockEdgeMatrix,
)

from .split_batch_training_result import SplitBatchTrainingResult
from .test_inner_layers_on_array_backends import MixtureOfProductsCase

# %% training


def train_on_split_batches() -> SplitBatchTrainingResult:
    """
    Train four epochs of batches of 100 rows of 450 sampled rows, so that the last batch
    of every epoch has a number of rows no number of devices above two divides.
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
    return SplitBatchTrainingResult(
        number_of_devices=len(jax.devices()),
        devices_of_state=len(training.state.parameters[0].devices()),
        devices_of_trained_parameters=len(trained[0].devices()),
        even_batch_size=gradient_descent.largest_even_batch_size(rows=450),
        epoch_losses=epoch_losses,
        parameters=[np.asarray(parameter) for parameter in trained],
    )


def main() -> None:
    """
    Train on the CPU devices in 64 bit and write the result to the file the command line
    names.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="The JSON file to write.")
    output = parser.parse_args().output
    jax.config.update("jax_platforms", JaxPlatform.CPU)
    jax.config.update("jax_enable_x64", True)
    output.write_text(json.dumps(to_json(train_on_split_batches())))


if __name__ == "__main__":
    main()
