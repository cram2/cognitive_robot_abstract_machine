"""Tests that memoizing structural facts during evaluation never pins the expression tree."""

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.evaluation_context import get_evaluation_context
from krrood.entity_query_language.factories import an, entity, variable
from krrood.entity_query_language.predicate import symbolic_function


def test_structural_cache_never_strongly_references_expressions():
    """
    The evaluation context is captured by long-lived consumers (an inference explanation keeps the
    evaluation's result). Its per-evaluation structural cache must therefore never strongly reference
    an expression node -- neither directly nor inside a plain container -- otherwise it would pin the
    whole tree, and every variable's domain, for as long as the consumer lives. Node indices are held
    through weak references instead.
    """
    captured_contexts = []

    @symbolic_function
    def capture_context(value):
        captured_contexts.append(get_evaluation_context())
        return True

    subject = variable(int, [1, 2, 3])
    an(entity(subject).where(capture_context(subject))).first()

    assert captured_contexts, "the capturing predicate never ran during evaluation"
    context = captured_contexts[0]
    assert context is not None
    cached_values = (
        *context.subtree_containment_cache.values(),
        *context.expression_index_cache.values(),
    )
    for cached_value in cached_values:
        assert not isinstance(cached_value, SymbolicExpression)
        if isinstance(cached_value, dict):
            assert not any(
                isinstance(entry, SymbolicExpression) for entry in cached_value.values()
            )


def test_evaluation_releases_its_context_when_it_finishes():
    """A top-level evaluation must release its :class:`EvaluationContext` once it finishes."""
    subject = variable(int, [1, 2, 3])
    an(entity(subject)).tolist()
    assert get_evaluation_context() is None
