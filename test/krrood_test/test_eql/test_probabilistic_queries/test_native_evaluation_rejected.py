import pytest

from krrood.entity_query_language.backends import EntityQueryLanguageBackend
from krrood.entity_query_language.exceptions import BackendCannotEvaluateProbabilisticQuery
from krrood.entity_query_language.factories import a, distribution_of

from ._fixtures import Coin

# distribution_of never resolves natively -- there's no row-based equivalent for a
# whole distribution the way probability_of has counting (see test_probability.py's
# native-evaluation tests). average(...) isn't covered here either: it's not a
# ProbabilisticQuery -- it has an ordinary native meaning (a per-row average) and is
# only reinterpreted probabilistically by ProbabilisticBackend, so it belongs in
# test_average.py.


def test_distribution_of_native_evaluation_rejected():
    match = a(Coin)(a=..., b=..., c=...)
    query = distribution_of(match)

    with pytest.raises(BackendCannotEvaluateProbabilisticQuery):
        list(query._evaluate_natively_())

    with pytest.raises(BackendCannotEvaluateProbabilisticQuery):
        query.first()

    with pytest.raises(BackendCannotEvaluateProbabilisticQuery):
        query.first(backend=EntityQueryLanguageBackend())
