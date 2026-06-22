from __future__ import annotations

from typing_extensions import Dict, Type

from krrood.entity_query_language.operators.aggregators import (
    Aggregator,
    Average,
    Count,
    CountAll,
    Max,
    Min,
    Mode,
    MultiMode,
    Sum,
)
from krrood.entity_query_language.verbalization.vocabulary.english import Aggregations

#: Maps each standard aggregator subtype to its lexicon phrase.
AGGREGATION_KIND: Dict[Type[Aggregator], Aggregations] = {
    Count: Aggregations.COUNT,
    CountAll: Aggregations.COUNT_ALL,
    Sum: Aggregations.SUM,
    Average: Aggregations.AVERAGE,
    Max: Aggregations.MAX,
    Min: Aggregations.MIN,
    Mode: Aggregations.MODE,
    MultiMode: Aggregations.MULTI_MODE,
}
