"""
Unit tests for :class:`ReferringExpressions` after coreference moved into the document-
order :class:`~krrood.entity_query_language.verbalization.rendering.coreference_processo
r.CoreferenceProcessor`.

What remains here is *pre-computed* / *cross-build* state: the disambiguation map and
the set of introduced referents (consulted only to seed a later build sharing the same
context).  The first/subsequent/pronoun decision itself is tested in
``test_coreference_phase.py``.
"""

from __future__ import annotations

from krrood.entity_query_language.verbalization.microplanning.referring import (
    ReferringExpressions,
)


def test_seen_starts_empty():
    assert ReferringExpressions().seen == set()
