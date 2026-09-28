from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import (
        Layer,
    )


@dataclass
class LayerWithDepth:
    """
    A layer of a circuit together with its distance from the root.
    """

    depth: int
    """
    The number of layers between the root layer and this layer.
    """

    layer: Layer
    """
    The layer at that depth.
    """
