from dataclasses import dataclass
from enum import Enum, auto
from typing_extensions import List

from krrood.entity_query_language.factories import a, cause, confounder
from krrood.parametrization.feature_extraction.aggregations import (
    AggregationStatistic,
    aggregation_statistic,
)
from krrood.parametrization.parameterizer import UnderspecifiedParameters


class Status(Enum):
    SUCCESS = auto()
    FAILURE = auto()


@dataclass
class Pick:
    arm: float
    status: Status


@dataclass
class Component:
    weight: float


@dataclass
class Assembly:
    """
    A class whose relevant statistic is not one of its own fields, but an aggregate over
    an exchangeable part -- the shape ``cause``/``confounder`` needs to resolve without
    a literal field to read a type off.
    """

    components: List[Component]


@dataclass
class AssemblyAggregations(AggregationStatistic[Assembly]):
    @aggregation_statistic("components")
    def component_count(self) -> int:
        """
        Count of components.
        """
        return len(self.instance.components)


def test_cause_is_registered_as_a_search_cause_variable():
    match = a(Pick)(arm=cause, status=...)
    parameters = UnderspecifiedParameters(match)
    assert len(parameters.search_cause_variables) == 1
    assert parameters.search_cause_variables[0].name == "Pick.arm"


def test_cause_variable_is_also_a_registered_variable():
    match = a(Pick)(arm=cause, status=...)
    parameters = UnderspecifiedParameters(match)
    assert "Pick.arm" in parameters.variables


def test_without_cause_no_search_cause_variables_are_registered():
    match = a(Pick)(arm=0.3, status=...)
    parameters = UnderspecifiedParameters(match)
    assert parameters.search_cause_variables == []


def test_causes_effect_condition_registers_its_effect_variable():
    match = a(Pick)(arm=cause, status=...)
    match.causes_effect(match.status == Status.SUCCESS)
    parameters = UnderspecifiedParameters(match)
    assert len(parameters.effect_variables_from_causes_effect) == 1
    assert parameters.effect_variables_from_causes_effect[0].name == "Pick.status"


def test_without_causes_effect_no_effect_variables_are_registered():
    match = a(Pick)(arm=cause, status=...)
    parameters = UnderspecifiedParameters(match)
    assert parameters.effect_variables_from_causes_effect == []


def test_causes_effect_conjunction_registers_every_effect_variable():
    match = a(Pick)(arm=cause, status=...)
    match.causes_effect(match.status == Status.SUCCESS, match.arm == 0.3)
    parameters = UnderspecifiedParameters(match)
    names = {v.name for v in parameters.effect_variables_from_causes_effect}
    assert names == {"Pick.status", "Pick.arm"}


# %% cause/confounder on an aggregation statistic, not a literal field


def test_cause_on_an_aggregation_statistic_resolves_its_return_type():
    """
    The resolved name must match ``"{AggregationClass}.{method}()"``, the same
    convention EQL's own ``variable(AggregationClass).method()`` attribute access
    produces, not ``Assembly.component_count`` -- that is not how grounding actually
    names this variable on the circuit, so a search variable under that name would never
    match anything a real ``RelationalCircuitRegistry`` grounds.
    """
    match = a(Assembly)(
        component_count=cause,
        components=[a(Component)(weight=...)],
    )
    parameters = UnderspecifiedParameters(match)
    assert len(parameters.search_cause_variables) == 1
    assert (
        parameters.search_cause_variables[0].name
        == "AssemblyAggregations.component_count()"
    )


def test_confounder_on_an_aggregation_statistic_resolves_its_return_type():
    match = a(Assembly)(
        component_count=confounder,
        components=[a(Component)(weight=...)],
    )
    parameters = UnderspecifiedParameters(match)
    assert len(parameters.search_confounder_variables) == 1
    assert (
        parameters.search_confounder_variables[0].name
        == "AssemblyAggregations.component_count()"
    )
