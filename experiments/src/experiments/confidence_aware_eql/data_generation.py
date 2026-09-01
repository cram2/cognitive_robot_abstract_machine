"""Generate real SDT objects with sampled masses for the confidence-aware pipeline.

The confidence model needs a population of "familiar" objects to fit on. Rather than
hand-authoring them, this module drives EQL's :class:`ProbabilisticBackend` with a
nested query on a class' root body inertial mass, so it constructs real objects
(``Cup``, ``Pot``, ...) whose mass is drawn from a chosen Gaussian.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import a
from krrood.parametrization.model_registries import DictRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from semantic_digital_twin.world_description.inertial_properties import Inertial
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Any, List, Type


class MassVariableMissing(KeyError):
    """Raised when a semantic-annotation class has no ``root.inertial.mass`` leaf.

    Without this check, a class whose query does not produce the expected mass
    variable would leave its mass as an unresolved ``Ellipsis`` instead of a sampled
    value, since the parameterizer silently skips variables it cannot map back.
    """


@dataclass
class MassDistribution:
    """The familiar mass distribution to sample one class' objects from."""

    object_class: Type
    """The semantic-annotation class to generate, e.g. ``Cup`` or ``Pot``."""

    mean: float
    """The mean mass, in kilograms, familiar objects of this class are sampled around."""

    standard_deviation: float
    """The standard deviation of the mass, in kilograms, familiar objects are sampled with."""

    number_of_samples: int
    """How many objects of this class to generate."""


def generate_familiar_objects(mass_distributions: List[MassDistribution]) -> List[Any]:
    """Generate real SDT objects with sampled masses via EQL's probabilistic backend.

    :param mass_distributions: The per-class distributions to sample objects from.
    :return: The generated objects, grouped by distribution in the given order.
    """
    generated_objects = []
    for distribution in mass_distributions:
        generated_objects.extend(_generate_objects_for_distribution(distribution))
    return generated_objects


def _mass_variable_name(object_class: Type) -> str:
    """
    :param object_class: The semantic-annotation class whose mass leaf is queried.
    :return: The dotted access-path name EQL assigns to that class' root inertial mass.
    """
    return f"{object_class.__name__}.root.inertial.mass"


def _generate_objects_for_distribution(distribution: MassDistribution) -> List[Any]:
    """
    Sample ``distribution.number_of_samples`` objects of ``distribution.object_class``.

    :param distribution: The mass distribution to sample objects from.
    :return: The generated objects.
    :raises MassVariableMissing: If the object class' query has no
        ``root.inertial.mass`` variable to parameterize.
    """
    query = a(distribution.object_class)(root=a(Body)(inertial=a(Inertial)(mass=...)))
    parameters = UnderspecifiedParameters(query)

    mass_variable_name = _mass_variable_name(distribution.object_class)
    if mass_variable_name not in parameters.variables:
        raise MassVariableMissing(mass_variable_name)
    mass_variable = parameters.variables[mass_variable_name]

    circuit = fully_factorized(
        parameters.variables.values(),
        means={mass_variable: distribution.mean},
        variances={mass_variable: distribution.standard_deviation},
    )
    registry = DictRegistry({distribution.object_class: circuit})
    backend = ProbabilisticBackend(
        registry, number_of_samples=distribution.number_of_samples
    )
    return list(backend.evaluate(query))
