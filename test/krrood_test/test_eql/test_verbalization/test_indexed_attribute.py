"""
Tests for indexed-collection attributes: an integer subscript folds its ordinal into the
collection noun, which singularizes — *"the first task of a Worker"* rather than *"the
first of the tasks of a Worker"*.
"""

from __future__ import annotations

from krrood.entity_query_language.factories import variable
from krrood.entity_query_language.verbalization.example_domain import Worker
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression


def test_first_index_folds_into_singular_noun():
    worker = variable(Worker, [])
    assert verbalize_expression(worker.tasks[0]) == "the first task of a Worker"


def test_negative_index_reads_as_last():
    worker = variable(Worker, [])
    assert verbalize_expression(worker.tasks[-1]) == "the last task of a Worker"


def test_second_to_last_index():
    worker = variable(Worker, [])
    assert (
        verbalize_expression(worker.tasks[-2]) == "the second to last task of a Worker"
    )


def test_attribute_of_indexed_element():
    worker = variable(Worker, [])
    assert (
        verbalize_expression(worker.tasks[0].name)
        == "the name of the first task of a Worker"
    )


def test_boolean_attribute_of_indexed_element():
    worker = variable(Worker, [])
    assert (
        verbalize_expression(worker.tasks[0].completed)
        == "the first task of a Worker is completed"
    )
