from __future__ import annotations

import pytest

from krrood.parametrization.exceptions import (
    AmbiguousVariableName,
    DoRequiresCausalCircuitModel,
    DomainElementsIndistinguishableInSamples,
    JointQueryAcrossClassesNotSupported,
    ModelVariableNotBound,
    MultipleEffectVariablesNotSupported,
    RelationalCircuitRegistryRequiresMatch,
    UnboundedParameterError,
    ValueOutsideVariableDomain,
)
from krrood.parametrization.feature_extraction.exceptions import (
    OutOfDomainValueError,
    UnsupportedFeatureTypeError,
)
from probabilistic_model.exceptions import ProbabilisticModelException

# %% exceptions about a probabilistic model


@pytest.mark.parametrize(
    "exception_class",
    [
        ModelVariableNotBound,
        ValueOutsideVariableDomain,
        DoRequiresCausalCircuitModel,
        MultipleEffectVariablesNotSupported,
        JointQueryAcrossClassesNotSupported,
        RelationalCircuitRegistryRequiresMatch,
        AmbiguousVariableName,
        DomainElementsIndistinguishableInSamples,
        UnboundedParameterError,
        OutOfDomainValueError,
        UnsupportedFeatureTypeError,
    ],
    ids=lambda exception_class: exception_class.__qualname__,
)
def test_exception_about_a_probabilistic_model_is_a_probabilistic_model_exception(
    exception_class: type[BaseException],
):
    assert issubclass(exception_class, ProbabilisticModelException)
