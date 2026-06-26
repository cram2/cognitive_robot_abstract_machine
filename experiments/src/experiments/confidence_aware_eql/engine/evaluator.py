"""
ConfidenceAwareEvaluator — score an object and (maybe) warn.

Given a Domain, a fitted CircuitModel, and a threshold, `check()`:
  1. encodes the object (reporting a missing/unknown feature -> warn), then
  2. runs a full-evidence log_likelihood query, then
  3. compares to the threshold and records an UnfamiliarSampleWarning if below.

`node_name` is carried on every warning for traceability. In the standalone
demo there is a single check site; when wired into an EQL rule tree, the same
call is made per node with that node's name (and, later, a marginal/conditional
view for contextual scoring).
"""

from dataclasses import dataclass, field
from typing_extensions import Dict, List, Optional, Tuple
import numpy as np

from .domain import Domain
from .circuit_model import CircuitModel
from .warning import UnfamiliarSampleWarning


@dataclass
class ConfidenceAwareEvaluator:
    domain: Domain
    model: CircuitModel
    threshold: float
    warnings: List[UnfamiliarSampleWarning] = field(default_factory=list)

    def check(self, obj: Dict, node_name: str = "root") -> Tuple[Optional[float], Optional[UnfamiliarSampleWarning]]:
        """Score one object dict. Returns (log_p, warning_or_None).

        log_p is None when the object could not be scored (missing features).
        The warning (if any) is also appended to self.warnings.
        """
        row, missing = self.domain.encode_row(obj)

        if missing:
            w = UnfamiliarSampleWarning(
                node_name=node_name, log_p=None,
                reason=f"incomplete features {missing} (missing/unknown tag)",
            )
            self.warnings.append(w)
            return None, w

        log_p = float(self.model.log_likelihood(row)[0])
        if log_p < self.threshold:
            w = UnfamiliarSampleWarning(
                node_name=node_name, log_p=log_p,
                reason=f"log P(x)={log_p:.2f} < threshold {self.threshold:.2f}",
            )
            self.warnings.append(w)
            return log_p, w

        return log_p, None

    def is_familiar(self, obj: Dict) -> bool:
        log_p, w = self.check(obj, node_name="_probe_")
                                          
        if self.warnings and self.warnings[-1].node_name == "_probe_":
            self.warnings.pop()
        return w is None
