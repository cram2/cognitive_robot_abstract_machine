from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException
from typing_extensions import TYPE_CHECKING, Type, Union

if TYPE_CHECKING:
    from probabilistic_model.probabilistic_circuit.tensorized.inner_layer import Layer


@dataclass
class BatchedTruncationUnsupported(DataclassException):
    """
    Raised when a layer cannot be truncated to several simple events in one pass.

    The circuit catches this and falls back to truncating once per simple event. It is an
    exception rather than a ``None`` return because the decision is made deep inside the
    recursion, by an input layer, and has to abort the whole pass.
    """

    layer: Union[Type[Layer], Layer]
    """
    The layer, or layer type, that does not support batched truncation.
    """

    def error_message(self) -> str:
        return f"{self.layer} does not support batched truncation."

    def suggest_correction(self) -> str:
        return ""
