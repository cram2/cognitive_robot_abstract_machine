from __future__ import annotations

import inspect

from typing_extensions import List

from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    PhraseRule,
)
from krrood.utils import recursive_subclasses

# Import every construct's rules module so its ``PhraseRule`` subclasses are loaded (and therefore
# discoverable) before ``RULES`` is built below. This module is the single registration point: a
# new construct is registered by adding its ``rules`` import here. The rules themselves live next
# to their construct's planner/assembler, not in one central module.
from krrood.entity_query_language.verbalization.grammar.terms import rules as _terms
from krrood.entity_query_language.verbalization.grammar.chain import rules as _chain
from krrood.entity_query_language.verbalization.grammar.conditions import rules as _conditions
from krrood.entity_query_language.verbalization.grammar.query import rules as _query
from krrood.entity_query_language.verbalization.grammar.inference import rules as _inference
from krrood.entity_query_language.verbalization.grammar.aggregation import rules as _aggregation
from krrood.entity_query_language.verbalization.grammar.clauses import rules as _clauses
from krrood.entity_query_language.verbalization.grammar.instantiated import rules as _instantiated

# Auto-discovered: one instance of every concrete ``PhraseRule`` subclass. Order is irrelevant —
# ``select`` decides specificity — and a new rule is registered simply by defining its class in one
# of the imported ``rules`` modules. This mirrors how the ``SpecificityRule`` families discover
# their alternatives (``recursive_subclasses`` + abstract filtering).
RULES: List[PhraseRule] = [
    rule_cls()
    for rule_cls in recursive_subclasses(PhraseRule)
    if not inspect.isabstract(rule_cls)
]
