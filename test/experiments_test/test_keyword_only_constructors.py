from __future__ import annotations

import experiments.sage_10k

from ..cramph_test.test_statechart.keyword_only_constructors import (
    KeywordOnlyConstructorAudit,
)

# %% keyword-only constructors


def test_every_experiment_node_takes_keyword_arguments_only() -> None:
    audit = KeywordOnlyConstructorAudit([experiments.sage_10k])
    assert audit.nodes_with_positional_parameters() == []
