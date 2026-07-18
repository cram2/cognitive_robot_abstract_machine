from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture

from experiments.confidence_aware_eql.engine.circuit_model import (
    fit_gaussian_mixture_circuit,
)
from experiments.confidence_aware_eql.engine.evaluator import ConfidenceAwareEvaluator
from experiments.confidence_aware_eql.engine.schema import FeatureSchema
from experiments.confidence_aware_eql.engine.threshold import PercentileThreshold


class SampleMaterial(Enum):
    """A categorical feature used to exercise enum encoding in the schema."""

    FIRST = 0
    SECOND = 1


@dataclass
class TwoFeatureSample:
    """A minimal instance exercising one continuous and one categorical feature."""

    weight: float
    material: SampleMaterial


@pytest.fixture
def two_cluster_data() -> np.ndarray:
    """Two well-separated clusters of light and heavy instances."""
    generator = np.random.default_rng(0)
    light = np.column_stack(
        [generator.normal(0.25, 0.05, 400), generator.normal(0.10, 0.02, 400)]
    )
    heavy = np.column_stack(
        [generator.normal(3.0, 0.30, 400), generator.normal(0.30, 0.03, 400)]
    )
    return np.vstack([light, heavy])


def test_circuit_likelihood_matches_gaussian_mixture(two_cluster_data):
    """The compiled circuit reproduces the likelihood of the fitted mixture."""
    feature_names = ["weight", "size"]
    circuit = fit_gaussian_mixture_circuit(
        two_cluster_data, feature_names, number_of_components=2
    )
    reference = GaussianMixture(
        n_components=2, covariance_type="diag", random_state=0, reg_covar=1e-4
    ).fit(two_cluster_data)
    points = two_cluster_data[:5]
    assert np.allclose(
        circuit.log_likelihood(points), reference.score_samples(points), atol=1e-4
    )


def test_percentile_threshold_matches_numpy_percentile():
    """The percentile threshold equals the requested percentile of the input."""
    log_likelihoods = np.linspace(-10.0, 0.0, 101)
    threshold = PercentileThreshold(percentile=1.0)
    assert threshold.fit(log_likelihoods) == pytest.approx(
        np.percentile(log_likelihoods, 1.0)
    )


def test_schema_derives_features_from_dataclass():
    """The schema takes its variables from the public fields of the dataclass."""
    schema = FeatureSchema.from_dataclass(TwoFeatureSample)
    assert schema.feature_names == ["weight", "material"]


def test_enum_feature_is_encoded_by_value():
    """An enumeration feature is encoded as the numeric value of its member."""
    schema = FeatureSchema.from_dataclass(TwoFeatureSample)
    encoded = schema.encode(TwoFeatureSample(1.0, SampleMaterial.SECOND))
    assert encoded.is_complete
    assert encoded.row[1] == 1.0


def test_missing_feature_is_flagged():
    """An absent feature marks the encoded instance as incomplete."""
    schema = FeatureSchema.from_dataclass(TwoFeatureSample)
    encoded = schema.encode(TwoFeatureSample(1.0, None))
    assert not encoded.is_complete
    assert "material" in encoded.missing_features


def test_impossible_instance_flagged_by_evaluator(two_cluster_data):
    """A far out-of-distribution instance is reported as unfamiliar."""
    @dataclass
    class WeightSizeSample:
        """An instance with two continuous features."""

        weight: float
        size: float

    schema = FeatureSchema.from_dataclass(WeightSizeSample)
    circuit = fit_gaussian_mixture_circuit(
        two_cluster_data, schema.feature_names, number_of_components=2
    )
    threshold = PercentileThreshold(percentile=1.0)
    threshold.fit(circuit.log_likelihood(two_cluster_data))
    evaluator = ConfidenceAwareEvaluator(schema, circuit, threshold)

    familiar = evaluator.check(WeightSizeSample(0.25, 0.10), node_name="test_node")
    assert familiar.is_familiar

    impossible = evaluator.check(WeightSizeSample(50.0, 0.10), node_name="test_node")
    assert not impossible.is_familiar
    assert impossible.warning.node_name == "test_node"
