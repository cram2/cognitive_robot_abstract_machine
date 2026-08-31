from dataclasses import dataclass
from unittest.mock import patch

import pytest

from krrood.entity_query_language.backends import (
    EntityQueryLanguageBackend,
    EntityQueryLanguageGenerativeBackend,
)
from krrood.entity_query_language.exceptions import (
    BackendCannotEvaluateCause,
    SelectiveBackendCannotResolveEllipsisMatch,
    UnderspecifiedStatementInfeasibleForEntityQueryLanguageGeneration,
)
from krrood.entity_query_language.factories import a, cause


@dataclass
class Pick:
    arm: float
    grasped: bool


# %% SelectiveBackend


def test_selective_backend_warns_and_finds_nothing_by_default():
    apple = Pick(0.3, True)
    match = a(Pick)(arm=cause, grasped=True).from_([apple])

    with patch("krrood.entity_query_language.backends.logger.warning") as warning:
        results = list(match.evaluate(backend=EntityQueryLanguageBackend()))

    assert results == []
    warning.assert_called_once()
    assert isinstance(warning.call_args.args[0], BackendCannotEvaluateCause)


def test_selective_backend_raises_when_configured_to_raise():
    apple = Pick(0.3, True)
    match = a(Pick)(arm=cause, grasped=True).from_([apple])

    backend = EntityQueryLanguageBackend(raise_on_unresolvable_cause=True)
    with pytest.raises(BackendCannotEvaluateCause):
        list(match.evaluate(backend=backend))


def test_selective_backend_still_rejects_a_plain_ellipsis_attribute():
    match = a(Pick)(arm=..., grasped=True).from_([Pick(0.3, True)])
    with pytest.raises(SelectiveBackendCannotResolveEllipsisMatch):
        list(match.evaluate(backend=EntityQueryLanguageBackend()))


# %% EntityQueryLanguageGenerativeBackend


def test_generative_backend_warns_and_raises_the_existing_infeasibility_error():
    # `arm` is a non-enum type, so degrading `cause` to `...` for a backend with no
    # causal reasoning hits the same infeasibility guard a bare `arm=...` would.
    match = a(Pick)(arm=cause, grasped=True)

    with patch("krrood.entity_query_language.backends.logger.warning") as warning:
        with pytest.raises(
            UnderspecifiedStatementInfeasibleForEntityQueryLanguageGeneration
        ):
            list(match.evaluate(backend=EntityQueryLanguageGenerativeBackend()))

    warning.assert_called_once()
    assert isinstance(warning.call_args.args[0], BackendCannotEvaluateCause)


def test_generative_backend_raises_when_configured_to_raise():
    match = a(Pick)(arm=cause, grasped=True)
    backend = EntityQueryLanguageGenerativeBackend(raise_on_unresolvable_cause=True)
    with pytest.raises(BackendCannotEvaluateCause):
        list(match.evaluate(backend=backend))
