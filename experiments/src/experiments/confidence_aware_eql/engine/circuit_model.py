"""
CircuitModel — learn a tractable probabilistic circuit FROM data.

We fit a Gaussian Mixture (EM) to the training matrix, then compile it into a
probabilistic_model sum-product circuit:

        SumUnit                      (the mixture)
        ├── w_1 · ProductUnit_1      Π_d Gaussian(mean_1d, std_1d)
        ├── ...
        └── w_K · ProductUnit_K      Π_d Gaussian(mean_Kd, std_Kd)

This is mathematically a tractable probabilistic circuit and supports
log_likelihood / marginal / conditional. The number of components K can be
chosen automatically by BIC, so the engine adapts to any domain without being
told how many object types exist.

IMPORTANT: probabilistic_model sorts a circuit's variables alphabetically, so
input columns must be permuted into circuit-variable order before querying.
CircuitModel hides this: callers always pass rows in DOMAIN feature order.
"""

from typing_extensions import List, Optional, Self
import numpy as np
from sklearn.mixture import GaussianMixture

from random_events.variable import Continuous
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit, SumUnit, ProductUnit, leaf,
)


class CircuitModel:
    def __init__(self, circuit: ProbabilisticCircuit, domain_order: List[str]):
        self.circuit = circuit
        self.domain_order = list(domain_order)
        circuit_names = [v.name for v in circuit.variables]
        self._perm = [self.domain_order.index(n) for n in circuit_names]

    def log_likelihood(self, rows) -> np.ndarray:
        """rows: array (n, n_features) in DOMAIN feature order."""
        rows = np.asarray(rows, dtype=float)
        if rows.ndim == 1:
            rows = rows[None, :]
        return self.circuit.log_likelihood(rows[:, self._perm])

    def marginal(self, feature_names: List[str]):
        """Return a sub-circuit over a subset of features (for per-node checks)."""
        variables = [v for v in self.circuit.variables if v.name in feature_names]
        return self.circuit.marginal(variables)

    @classmethod
    def fit(cls, data: np.ndarray, domain_order: List[str],
            n_components="auto", max_components: int = 8,
            seed: int = 0, reg_covar: float = 1e-4) -> Self:
        data = np.asarray(data, dtype=float)
        n_features = data.shape[1]
        assert n_features == len(domain_order), "data columns must match domain"

        if n_components == "auto":
            n_components = cls._select_k(data, max_components, seed, reg_covar)

        gmm = GaussianMixture(
            n_components=n_components, covariance_type="diag",
            random_state=seed, reg_covar=reg_covar,
        ).fit(data)

        variables = [Continuous(name) for name in domain_order]

        circuit = ProbabilisticCircuit()
        mixture = SumUnit(probabilistic_circuit=circuit)
        for k in range(gmm.n_components):
            product = ProductUnit(probabilistic_circuit=circuit)
            for d, var in enumerate(variables):
                mean = float(gmm.means_[k, d])
                std = float(np.sqrt(gmm.covariances_[k, d]))
                product.add_subcircuit(leaf(GaussianDistribution(location=mean, scale=std, variable=var), circuit))
            mixture.add_subcircuit(product, float(np.log(gmm.weights_[k])))

        model = cls(circuit, domain_order)
        model.n_components = gmm.n_components
        return model

    @staticmethod
    def _select_k(data, max_components, seed, reg_covar) -> int:
        n = len(data)
        best_k, best_bic = 1, np.inf
        for k in range(1, min(max_components, n) + 1):
            try:
                g = GaussianMixture(n_components=k, covariance_type="diag",
                                    random_state=seed, reg_covar=reg_covar).fit(data)
                bic = g.bic(data)
            except Exception:
                continue
            if bic < best_bic:
                best_bic, best_k = bic, k
        return best_k
