"""
Tests that memoizing structural facts during evaluation never pins the expression tree.
"""

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.evaluation_context import (
    EvaluationContext,
    get_evaluation_context,
)
from krrood.entity_query_language.factories import an, entity, variable
from krrood.entity_query_language.predicate import symbolic_function


def test_structural_cache_never_strongly_references_expressions():
    """
    The evaluation context is captured by long-lived consumers (an inference explanation
    keeps the evaluation's result).

    Its per-evaluation structural cache must therefore never strongly reference an
    expression node -- neither directly nor inside a plain container -- otherwise it
    would pin the whole tree, and every variable's domain, for as long as the consumer
    lives. Node indices are held through weak references instead.
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
    """
    A top-level evaluation must release its :class:`EvaluationContext` once it finishes.
    """
    subject = variable(int, [1, 2, 3])
    an(entity(subject)).tolist()
    assert get_evaluation_context() is None


def test_suspended_evaluation_does_not_leave_its_context_set():
    """
    An evaluation whose caller keeps its iterator suspended between results must not
    leave its :class:`EvaluationContext` set for the caller.

    The context holds the query it evaluates, so leaving it set pins that query, and
    every object its variables reach, for as long as the thread lives; it would also be
    picked up by the next, unrelated evaluation.
    """
    subject = variable(int, [1, 2, 3])
    results = an(entity(subject)).evaluate()

    next(results)

    assert get_evaluation_context() is None
    results.close()


def test_resumed_evaluation_sees_its_own_context_on_every_result():
    """
    An evaluation resumed after being suspended evaluates each result within the context
    it started with.
    """
    captured_contexts = []

    @symbolic_function
    def capture_context(value):
        captured_contexts.append(get_evaluation_context())
        return True

    subject = variable(int, [1, 2, 3])
    results = an(entity(subject).where(capture_context(subject))).evaluate()

    next(results)
    next(results)
    results.close()

    first_context, second_context = captured_contexts
    assert first_context is not None
    assert second_context is first_context


def test_interleaved_evaluations_of_one_query_each_keep_their_own_context():
    """
    Two evaluations of the same query advanced in turn each evaluate within a context of
    their own, rather than the second joining the one the first is suspended in.
    """
    captured_contexts = []

    @symbolic_function
    def capture_context(value):
        captured_contexts.append(get_evaluation_context())
        return True

    subject = variable(int, [1, 2])
    query = an(entity(subject).where(capture_context(subject)))
    first_results = query.evaluate()
    second_results = query.evaluate()

    next(first_results)
    next(second_results)
    next(first_results)
    next(second_results)
    first_results.close()
    second_results.close()

    first_context, second_context, first_again, second_again = captured_contexts
    assert first_context is not second_context
    assert first_again is first_context
    assert second_again is second_context


def test_evaluation_started_while_another_is_suspended_gets_its_own_context():
    """
    A query evaluated while an unrelated evaluation is suspended evaluates within a new
    context instead of joining the suspended one.
    """
    captured_contexts = []

    @symbolic_function
    def capture_context(value):
        captured_contexts.append(get_evaluation_context())
        return True

    suspended_subject = variable(int, [1, 2])
    suspended_results = an(
        entity(suspended_subject).where(capture_context(suspended_subject))
    ).evaluate()
    next(suspended_results)

    other_subject = variable(int, [3])
    an(entity(other_subject).where(capture_context(other_subject))).tolist()
    suspended_results.close()

    suspended_context, other_context = captured_contexts
    assert other_context is not suspended_context


def test_closing_suspended_results_keeps_their_context_current_for_the_cleanup():
    """
    Closing results part way runs their cleanup with the context they were produced in
    current, as it was while they advanced.
    """
    evaluation_context = EvaluationContext()
    contexts_seen_by_cleanup = []

    def results_with_cleanup():
        try:
            yield from [1, 2]
        finally:
            contexts_seen_by_cleanup.append(get_evaluation_context())

    results = evaluation_context.iterate_as_current(results_with_cleanup())
    next(results)
    results.close()

    assert contexts_seen_by_cleanup == [evaluation_context]
    assert get_evaluation_context() is None


def test_evaluation_root_query_falls_back_to_the_structural_root_outside_an_evaluation():
    """
    With no evaluation active there is no outermost query to resolve from, so
    :attr:`SymbolicExpression._evaluation_root_query_` must fall back to
    :attr:`SymbolicExpression._root_`.
    """
    subject = variable(int, [1, 2, 3])
    entity(subject)

    assert get_evaluation_context() is None
    assert subject._evaluation_root_query_ is subject._root_
