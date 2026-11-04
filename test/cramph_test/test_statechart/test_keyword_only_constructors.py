from __future__ import annotations

import cramph

from .keyword_only_constructors import KeywordOnlyConstructorAudit

# %% keyword-only constructors


def test_every_cramph_node_takes_keyword_arguments_only() -> None:
    audit = KeywordOnlyConstructorAudit([cramph])
    assert audit.nodes_with_positional_parameters() == []
