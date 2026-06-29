"""
Synthetic dataset generation — stands in for "a dataset from the internet".

A spec describes each object class as per-feature parameters:
  - continuous feature:  (mean, std)
  - categorical feature:  a label string (the class's fixed category)

generate_dataset() samples `n_per_class` objects per class and returns a numeric
matrix in the domain's feature order, ready to fit a CircuitModel. Because the
spec is just data, the SAME function builds kitchen, bathroom, or any domain.

This is genuine learning-from-data: the circuit is fit to these samples by EM,
not hand-specified. (Swapping in a real CSV later is a one-line change: load it
into a matrix in domain order and pass it to CircuitModel.fit.)
"""

from typing_extensions import Dict
import numpy as np

from .domain import Domain


def generate_dataset(domain: Domain, spec: Dict[str, Dict],
                     n_per_class: int = 80, seed: int = 0,
                     categorical_jitter: float = 0.02) -> np.ndarray:
    """Return an (n_classes * n_per_class, n_features) matrix in domain order."""
    rng = np.random.default_rng(seed)
    rows = []
    for class_name, class_spec in spec.items():
        for _ in range(n_per_class):
            values = []
            for f in domain.features:
                p = class_spec[f.name]
                if f.kind == "continuous":
                    mean, std = p
                    values.append(rng.normal(mean, std))
                else:                                                                 
                    code = f.categories[p]
                    values.append(code + rng.normal(0.0, categorical_jitter))
            rows.append(values)
    return np.array(rows, dtype=float)


def sample_objects(domain: Domain, spec: Dict[str, Dict],
                   n: int = 20, seed: int = 1) -> list:
    """Sample `n` familiar OBJECTS (as dicts) for evaluation/testing."""
    rng = np.random.default_rng(seed)
    classes = list(spec.keys())
    objs = []
    for i in range(n):
        cname = classes[i % len(classes)]
        cspec = spec[cname]
        obj = {}
        for f in domain.features:
            p = cspec[f.name]
            if f.kind == "continuous":
                mean, std = p
                obj[f.name] = float(rng.normal(mean, std))
            else:
                obj[f.name] = p             
        objs.append(obj)
    return objs
