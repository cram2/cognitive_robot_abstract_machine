"""
The result that the training process of :mod:`split_batch_training` writes and the test
of splitting batches across devices reads.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# %% result


@dataclass(kw_only=True)
class SplitBatchTrainingResult:
    """
    What training on mini-batches gave and where its arrays lived.
    """

    number_of_devices: int
    """
    The number of devices jax started with.
    """

    devices_of_state: int
    """
    The number of devices the training state was placed on.
    """

    devices_of_trained_parameters: int
    """
    The number of devices the parameters of the trained circuit are on.
    """

    even_batch_size: int
    """
    The largest batch size of at most 450 rows that the devices split evenly.
    """

    epoch_losses: list[float]
    """
    The average loss of every epoch.
    """

    parameters: list[npt.NDArray[np.float64]]
    """
    The parameters of the trained circuit.
    """
