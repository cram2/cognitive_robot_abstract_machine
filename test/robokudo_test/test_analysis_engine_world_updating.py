"""
Tests that every analysis engine puts the objects it perceives into the world, before it
answers a query.
"""

from __future__ import annotations

import pytest

from robokudo.annotators.query import GenerateQueryResult, QueryReply
from robokudo.annotators.world_updating import WorldUpdatingAnnotator

from .test_analysis_engine_query_composition import (
    built_analysis_engine_pipeline,
    discovered_analysis_engine_module_names,
    flattened_pipeline_nodes,
)

# %% world updating in every analysis engine


@pytest.mark.parametrize("module_name", discovered_analysis_engine_module_names())
def test_analysis_engine_updates_the_world(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The pipeline of every analysis engine puts the objects it perceives into the world.
    """
    pipeline = built_analysis_engine_pipeline(module_name, monkeypatch)

    assert any(
        isinstance(node, WorldUpdatingAnnotator)
        for node in flattened_pipeline_nodes(pipeline)
    )


@pytest.mark.parametrize("module_name", discovered_analysis_engine_module_names())
def test_analysis_engine_updates_the_world_right_before_answering_a_query(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Every query answer of an analysis engine directly follows a world update, so the
    answer reflects what the pass perceived.
    """
    pipeline = built_analysis_engine_pipeline(module_name, monkeypatch)
    query_answers = [
        node
        for node in flattened_pipeline_nodes(pipeline)
        if isinstance(node, (GenerateQueryResult, QueryReply))
    ]

    behaviours_before_the_answers = [
        answer.parent.children[answer.parent.children.index(answer) - 1]
        for answer in query_answers
    ]

    assert [type(behaviour) for behaviour in behaviours_before_the_answers] == [
        WorldUpdatingAnnotator
    ] * len(query_answers)
