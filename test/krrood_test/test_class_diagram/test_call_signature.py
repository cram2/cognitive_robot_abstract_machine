"""
Reading which parameters a call passes its arguments to.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from typing_extensions import Any, Callable

from krrood.class_diagrams.call_arguments import CallArguments
from krrood.class_diagrams.call_signature import CallSignature
from krrood.class_diagrams.exceptions import ArgumentsPassableOnlyByPosition

# %% mimic callees


def scale(value: float, factor: float = 2.0, *, offset: float = 0.0) -> float:
    return value * factor + offset


def scale_by_options(value: float, **options: float) -> float:
    return value * options["factor"]


def scale_by_position(value: float, /, factor: float = 2.0) -> float:
    return value * factor


def scale_all(*values: float) -> list[float]:
    return [value * 2.0 for value in values]


@dataclass
class Scaler:
    """
    A mutable dataclass, so neither it nor its bound methods can be hashed.
    """

    factor: float = 2.0

    def scale(self, value: float) -> float:
        return value * self.factor


# %% finding the signature of a callee


def test_a_callee_has_one_signature():
    assert CallSignature.from_callable(scale) is CallSignature.from_callable(scale)


def test_a_callee_that_cannot_be_hashed_has_a_signature():
    signature = CallSignature.from_callable(Scaler().scale)

    assert signature.bind_by_name(CallArguments((3.0,))) == {"value": 3.0}


# %% binding arguments


def test_a_call_binds_the_same_however_its_arguments_are_passed():
    signature = CallSignature.from_callable(scale)

    assert signature.bind(CallArguments((3.0, 4.0))) == signature.bind(
        CallArguments(keyword={"factor": 4.0, "value": 3.0})
    )


def test_a_default_that_was_left_out_binds_as_if_it_was_passed():
    signature = CallSignature.from_callable(scale)

    assert signature.bind(CallArguments((3.0,))) == signature.bind(
        CallArguments((3.0, 2.0), {"offset": 0.0})
    )


def test_arguments_that_fit_no_parameter_are_refused():
    with pytest.raises(TypeError):
        CallSignature.from_callable(scale).bind(CallArguments((1.0, 2.0, 3.0)))


# %% passing every argument by name


def test_each_passed_argument_is_named_by_the_parameter_that_receives_it():
    passed_by_name = CallSignature.from_callable(scale).bind_by_name(
        CallArguments((3.0,), {"offset": 1.0})
    )

    assert passed_by_name == {"value": 3.0, "offset": 1.0}


def test_the_arguments_passed_by_name_make_the_same_call():
    passed_by_name = CallSignature.from_callable(scale).bind_by_name(
        CallArguments((3.0, 4.0), {"offset": 1.0})
    )

    assert scale(**passed_by_name) == scale(3.0, 4.0, offset=1.0)


def test_a_class_is_called_without_self():
    passed_by_name = CallSignature.from_callable(Scaler).bind_by_name(
        CallArguments((3.0,))
    )

    assert passed_by_name == {"factor": 3.0}


def test_a_method_read_from_its_class_is_called_with_self():
    scaler = Scaler()

    passed_by_name = CallSignature.from_callable(Scaler.scale).bind_by_name(
        CallArguments((scaler, 3.0))
    )

    assert passed_by_name == {"self": scaler, "value": 3.0}


def test_a_method_read_from_its_class_can_take_self_by_name():
    scaler = Scaler()

    bound = CallSignature.from_callable(Scaler.scale).bind(
        CallArguments(keyword={"self": scaler, "value": 3.0})
    )

    assert bound == CallArguments((scaler, 3.0))


def test_arbitrary_keywords_keep_their_own_names():
    passed_by_name = CallSignature.from_callable(scale_by_options).bind_by_name(
        CallArguments((3.0,), {"factor": 4.0})
    )

    assert passed_by_name == {"value": 3.0, "factor": 4.0}


@pytest.mark.parametrize("callee", [scale_by_position, scale_all])
def test_an_argument_passable_only_by_position_cannot_be_named(
    callee: Callable[..., Any],
):
    with pytest.raises(ArgumentsPassableOnlyByPosition):
        CallSignature.from_callable(callee).bind_by_name(CallArguments((3.0,)))


# %% accepting a keyword


def test_a_parameter_accepts_its_name_as_a_keyword():
    assert CallSignature.from_callable(scale).accepts_keyword("offset")


def test_a_name_no_parameter_has_is_not_accepted():
    assert not CallSignature.from_callable(scale).accepts_keyword("shift")


def test_arbitrary_keywords_accept_any_name():
    assert CallSignature.from_callable(scale_by_options).accepts_keyword("shift")


def test_a_parameter_passable_only_by_position_does_not_accept_its_name():
    assert not CallSignature.from_callable(scale_by_position).accepts_keyword("value")
