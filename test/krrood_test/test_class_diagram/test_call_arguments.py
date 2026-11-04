"""
Passing the arguments of one call to a callee.
"""

from __future__ import annotations

from krrood.class_diagrams.call_arguments import CallArguments
from krrood.class_diagrams.call_signature import CallSignature

# %% mimic callees


def scale(value: float, factor: float = 2.0) -> float:
    return value * factor


# %% passing arguments to a callee


def test_passing_arguments_calls_the_callee_with_them_as_they_were_passed():
    arguments = CallArguments((3.0,), {"factor": 4.0})
    assert arguments.pass_to(scale) == scale(3.0, factor=4.0)


def test_passing_arguments_to_a_signature_binds_them():
    arguments = CallArguments((), {"value": 3.0})
    assert arguments.pass_to(CallSignature.of(scale).bind) == CallSignature.of(
        scale
    ).bind(value=3.0)
