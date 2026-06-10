"""
Unit tests for :class:`ReferringExpressions` after coreference moved into the document-order
:class:`~krrood.entity_query_language.verbalization.rendering.coreference_processor.CoreferenceProcessor`.

What remains here is *pre-computed* / *cross-build* state: the disambiguation map and the set of
introduced referents (consulted only to seed a later build sharing the same context).  The
first/subsequent/pronoun decision itself is tested in ``test_coreference_phase.py``.
"""

from __future__ import annotations

from krrood.entity_query_language.verbalization.microplanning.referring import (
    ReferringExpressions,
)


class _Ref:
    """Minimal stand-in carrying an ``_id_`` (what the coreference store keys on)."""

    def __init__(self, identifier: int) -> None:
        self._id_ = identifier


def test_mark_introduced_records_the_referent():
    refer = ReferringExpressions()
    robot = _Ref(1)
    assert robot._id_ not in refer.seen
    refer.mark_introduced(robot)
    # Recorded so a later build sharing this context seeds it as already-mentioned.
    assert robot._id_ in refer.seen


def test_seen_starts_empty():
    assert ReferringExpressions().seen == set()
