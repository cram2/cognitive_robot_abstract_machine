"""The traceable out-of-distribution warning."""

from dataclasses import dataclass
from typing_extensions import Optional


@dataclass
class UnfamiliarSampleWarning:
    node_name: str
    log_p: Optional[float]
    reason: str

    def __str__(self) -> str:
        return f"UnfamiliarSampleWarning at '{self.node_name}': {self.reason}"
