"""
The modifiers that narrow and shape a query's results.

Both a :class:`~krrood.entity_query_language.query.query.Query` and a
:class:`~krrood.entity_query_language.query.match.Match` are written by narrowing them
step by step, so the verbs that do the narrowing are declared here once instead of
being a convention each class is trusted to have followed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from typing_extensions import (
    Any,
    Callable,
    Generic,
    Optional,
    Self,
    TYPE_CHECKING,
    Union,
)

from krrood.entity_query_language.utils import T

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import Selectable
    from krrood.entity_query_language.factories import ConditionType


class HasQueryModifiers(Generic[T], ABC):
    """
    Something whose results can be narrowed and shaped by the query modifiers.

    Every modifier returns the receiver, so a chain stays on the object it started on.
    The return type is the instance-mimicking union the rest of the language uses: the
    receiver also stands for the value of type ``T`` it describes, so a chain may end in
    that value's own attributes.
    """

    @abstractmethod
    def where(self, *conditions: ConditionType) -> Union[T, Self]:
        """
        Constrain the described object by conditions, chained using AND.

        :param conditions: The conditions the described object must satisfy.
        :return: The receiver.
        """
        ...

    @abstractmethod
    def having(self, *conditions: ConditionType) -> Union[T, Self]:
        """
        Constrain the grouped results by conditions, chained using AND.

        :param conditions: The conditions a group must satisfy.
        :return: The receiver.
        """
        ...

    @abstractmethod
    def ordered_by(
        self,
        variable: Union[Selectable[T], Any],
        descending: bool = False,
        key: Optional[Callable] = None,
    ) -> Union[T, Self]:
        """
        Order the results by the given expression.

        :param variable: The expression to order by.
        :param descending: Whether to order the results in descending order.
        :param key: A function to extract the key from the expression's value.
        :return: The receiver.
        """
        ...

    @abstractmethod
    def distinct(self, *on: Union[Selectable, Any]) -> Union[T, Self]:
        """
        Keep only results that differ in the given expressions.

        :param on: The expressions the results must differ in; the selection by default.
        :return: The receiver.
        """
        ...

    @abstractmethod
    def grouped_by(
        self, *variables_to_group_by: Union[Selectable, Any]
    ) -> Union[T, Self]:
        """
        Group the results by the given expressions.

        :param variables_to_group_by: The expressions to group the results by.
        :return: The receiver.
        """
        ...

    @abstractmethod
    def limit(self, n: int) -> Union[T, Self]:
        """
        Return at most ``n`` results.

        :param n: The maximum number of results to return.
        :return: The receiver.
        """
        ...
