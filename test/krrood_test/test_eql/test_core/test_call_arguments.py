from __future__ import annotations

import pytest

from krrood.entity_query_language.call_arguments import CallArguments

# %% mimic callees


def scale(value: float, factor: float = 2.0, *, offset: float = 0.0) -> float:
    return value * factor + offset


class Scaler:
    def scale(self, value: float, factor: float = 2.0) -> float:
        return value * factor


# %% naming arguments by parameter


def test_arguments_are_named_by_the_parameter_that_receives_them():
    named = CallArguments.named_by_parameter(scale, (3.0,), {"offset": 1.0})
    assert named == {"value": 3.0, "offset": 1.0}


def test_a_default_that_was_left_out_is_not_named():
    named = CallArguments.named_by_parameter(scale, (3.0, 4.0), {})
    assert "offset" not in named


def test_the_first_parameter_is_skipped_when_it_is_not_passed():
    named = CallArguments.named_by_parameter(
        Scaler.scale, (3.0,), {"factor": 5.0}, without_first=True
    )
    assert named == {"value": 3.0, "factor": 5.0}


def test_arguments_that_fit_no_parameter_are_refused():
    with pytest.raises(TypeError):
        CallArguments.named_by_parameter(scale, (1.0, 2.0, 3.0), {})
