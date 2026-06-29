"""
Domain definition — the only thing that changes between domains.

A Domain is an ordered list of Features. Each Feature is either:
  - continuous  (e.g. weight, size), or
  - categorical (e.g. material, is_waterproof) with a {label: code} map.

Objects are plain dicts keyed by feature name, e.g.
    {"weight": 2.5, "size": 0.25, "material": "glass"}

encode_row() turns an object into a numeric row in the domain's feature order,
and reports any missing features (the "unset flag" case) so the caller can warn
instead of scoring an incomplete vector.

NOTE (known simplification): categorical features are encoded as numeric codes
and modelled by the continuous circuit. This works well when codes are distinct,
and keeps the engine fully general. A future version can swap in a proper
SymbolicDistribution leaf per categorical without changing the public API.
"""

from dataclasses import dataclass
from typing_extensions import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class Feature:
    name: str
    kind: str = "continuous"                                              
    categories: Optional[Dict[str, float]] = None                            

    def code(self, value) -> float:
        """Map a raw value to the numeric value the circuit sees."""
        if self.kind == "continuous":
            return float(value)
                     
        if value not in self.categories:
            raise KeyError(f"unknown category {value!r} for feature {self.name!r}")
        return float(self.categories[value])


@dataclass
class Domain:
    name: str
    features: List[Feature]

    @property
    def names(self) -> List[str]:
        return [f.name for f in self.features]

    def feature(self, name: str) -> Feature:
        for f in self.features:
            if f.name == name:
                return f
        raise KeyError(name)

    def encode_row(self, obj: Dict) -> Tuple[Optional[np.ndarray], List[str]]:
        """Encode one object dict into a numeric row (domain feature order).

        Returns (row, missing). If any feature is missing/None, row is None and
        `missing` lists the offending feature names. Unknown categorical labels
        are also reported as 'missing' (the value cannot be placed).
        """
        missing: List[str] = []
        values = []
        for f in self.features:
            v = obj.get(f.name, None)
            if v is None:
                missing.append(f.name)
                continue
            try:
                values.append(f.code(v))
            except KeyError:
                missing.append(f.name)
        if missing:
            return None, missing
        return np.array(values, dtype=float), missing
