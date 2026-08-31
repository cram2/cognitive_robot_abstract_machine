from dataclasses import dataclass
from enum import Enum, auto

from krrood.entity_query_language.factories import a, cause
from krrood.parametrization.parameterizer import UnderspecifiedParameters


class Status(Enum):
    SUCCESS = auto()
    FAILURE = auto()


@dataclass
class Pick:
    arm: float
    status: Status


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
    match.causes_effect(match.variable.status == Status.SUCCESS)
    parameters = UnderspecifiedParameters(match)
    assert len(parameters.effect_variables_from_causes_effect) == 1
    assert parameters.effect_variables_from_causes_effect[0].name == "Pick.status"


def test_without_causes_effect_no_effect_variables_are_registered():
    match = a(Pick)(arm=cause, status=...)
    parameters = UnderspecifiedParameters(match)
    assert parameters.effect_variables_from_causes_effect == []


def test_causes_effect_conjunction_registers_every_effect_variable():
    match = a(Pick)(arm=cause, status=...)
    match.causes_effect(
        match.variable.status == Status.SUCCESS, match.variable.arm == 0.3
    )
    parameters = UnderspecifiedParameters(match)
    names = {v.name for v in parameters.effect_variables_from_causes_effect}
    assert names == {"Pick.status", "Pick.arm"}
