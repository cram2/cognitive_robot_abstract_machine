import numpy as np
import pytest

import semantic_digital_twin.orm.ormatic_interface  # type: ignore  # noqa: F401
from krrood.entity_query_language.factories import a
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from semantic_digital_twin.semantic_annotations.semantic_annotations import Cup, Pot
from semantic_digital_twin.world_description.inertial_properties import Inertial
from semantic_digital_twin.world_description.world_entity import Body

from experiments.confidence_aware_eql.confidence_model import ConfidenceModel
from experiments.confidence_aware_eql.data_generation import (
    MassDistribution,
    generate_familiar_objects,
)
from experiments.confidence_aware_eql.feature_pipeline import (
    ObjectClass,
    extract_feature_dataframe,
)

# %% mass variable extraction assumption


def test_underspecified_query_has_exactly_one_mass_variable():
    """The nested Cup query has exactly one variable to unpack the mass from.

    `generate_familiar_objects` takes the mass variable directly from this single
    entry (`[mass_variable] = parameters.variables.values()`); if EQL ever produced
    zero or more than one underspecified variable for this query shape, that unpack
    would raise loudly instead of silently leaving the mass unset, and this test
    would catch the change first.
    """
    query = a(Cup)(root=a(Body)(inertial=a(Inertial)(mass=...)))
    parameters = UnderspecifiedParameters(query)
    assert len(parameters.variables) == 1


# %% generation


@pytest.fixture
def mass_distributions():
    """Familiar mass distributions for twenty cups and twenty pots."""
    return [
        MassDistribution(Cup, mean=0.25, standard_deviation=0.03, number_of_samples=20),
        MassDistribution(Pot, mean=2.50, standard_deviation=0.20, number_of_samples=20),
    ]


def test_generate_familiar_objects_returns_the_requested_count_per_class(
    mass_distributions,
):
    """One generated object per requested sample, of the matching class, in order."""
    np.random.seed(0)
    generated = generate_familiar_objects(mass_distributions)
    assert [type(instance) for instance in generated] == [Cup] * 20 + [Pot] * 20


def test_generated_masses_are_sampled_within_a_plausible_range(mass_distributions):
    """Generated masses are real floats close to their class' familiar mean."""
    np.random.seed(0)
    generated = generate_familiar_objects(mass_distributions)
    cup_distribution, pot_distribution = mass_distributions

    for cup in generated[:20]:
        assert isinstance(cup.root.inertial.mass, float)
        assert (
            abs(cup.root.inertial.mass - cup_distribution.mean)
            < 5 * cup_distribution.standard_deviation
        )
    for pot in generated[20:]:
        assert isinstance(pot.root.inertial.mass, float)
        assert (
            abs(pot.root.inertial.mass - pot_distribution.mean)
            < 5 * pot_distribution.standard_deviation
        )


# %% wiring into the existing pipeline


def test_generated_objects_flow_into_the_feature_pipeline_unchanged(mass_distributions):
    """Generated objects are consumable by `extract_feature_dataframe` unchanged."""
    np.random.seed(0)
    generated = generate_familiar_objects(mass_distributions)
    dataframe = extract_feature_dataframe(generated)
    assert dataframe.shape == (40, 2)
    assert list(dataframe["class"]) == [ObjectClass.CUP] * 20 + [ObjectClass.POT] * 20


def test_generated_objects_flow_into_the_confidence_model_unchanged(mass_distributions):
    """A model fitted on generated objects accepts a familiar probe, rejects an outlier."""
    np.random.seed(0)
    generated = generate_familiar_objects(mass_distributions)
    model = ConfidenceModel.fit_from_instances(generated)

    familiar_probe = Cup(root=Body(inertial=Inertial(mass=0.25)))
    outlier_probe = Cup(root=Body(inertial=Inertial(mass=50.0)))

    assert model.is_familiar(familiar_probe)
    assert not model.is_familiar(outlier_probe)
