"""
Shared evaluation interface for the Entity Query Language.

Both queries (which select existing data) and structural matches (which can generate new
instances) are evaluable. Defining a single interface lets backends treat them uniformly
and gives them a consistent ``evaluate``/``tolist``/``first`` surface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from typing_extensions import Iterator, TYPE_CHECKING

from krrood.entity_query_language.evaluation_context import EvaluationContext
from krrood.entity_query_language.exceptions import BackendCannotReadState
from krrood.entity_query_language.utils import make_list

if TYPE_CHECKING:
    from krrood.entity_query_language.backends import QueryBackend
    from krrood.entity_query_language.ground_term import TermReader


class Evaluable(ABC):
    """
    Interface for objects that can be evaluated to produce results.

    Implementations provide :meth:`_evaluate_natively_`, the in-process (backend-free)
    evaluation strategy. The public :meth:`evaluate` dispatches to a chosen backend,
    which by default is the native ``EntityQueryLanguageBackend``.
    """

    @abstractmethod
    def _evaluate_natively_(self) -> Iterator:
        """
        Produce results by evaluating this object in the current python process.

        :return: An iterator over the results.
        """
        ...

    def evaluate(
        self, backend: QueryBackend | None = None, state: TermReader | None = None
    ) -> Iterator:
        """
        Evaluate using the given backend, returning an iterator over the results.

        :param backend: The query backend to evaluate with. Defaults to the
            ``EntityQueryLanguageBackend`` (native in-process evaluation / generation).
        :param state: What the evaluation reads the values of attributes, indexes, calls
            and predicates from; their live values when omitted.
        :return: An iterator over the results.
        :raises BackendCannotReadState: If a state is given to a backend that cannot
            read one.
        """
        if backend is None:
            from krrood.entity_query_language.backends import EntityQueryLanguageBackend

            backend = EntityQueryLanguageBackend()
        if state is not None and not backend.reads_state:
            raise BackendCannotReadState(type(backend))
        return EvaluationContext.iterate_reading_from(state, backend.evaluate(self))

    def tolist(
        self, backend: QueryBackend | None = None, state: TermReader | None = None
    ) -> list:
        """
        Evaluate and return the results as a list.

        :param backend: The query backend to evaluate with; forwarded to
            :meth:`evaluate`.
        :param state: Forwarded to :meth:`evaluate`.
        """
        return make_list(self.evaluate(backend=backend, state=state))

    def first(
        self, backend: QueryBackend | None = None, state: TermReader | None = None
    ):
        """
        Evaluate and return the first result.

        :param backend: The query backend to evaluate with; forwarded to
            :meth:`evaluate`.
        :param state: Forwarded to :meth:`evaluate`.
        :raises StopIteration: If no results are found.
        """
        return next(self.evaluate(backend=backend, state=state))
