from __future__ import annotations

from dataclasses import dataclass, field, MISSING
from functools import cached_property

from typing_extensions import (
    Type,
    Any,
    Dict,
    Optional,
    Iterable,
    List,
    Union as TypingUnion,
    Union,
    Callable,
)

from .base_expressions import (
    Bindings,
    OperationResult,
    SymbolicExpression,
    Selectable,
)
from .domain_mapping import CanBehaveLikeAVariable
from ..cache_data import ReEnterableLazyIterable
from ..failures import VariableCannotBeEvaluated
from ..operators.set_operations import MultiArityExpressionThatPerformsACartesianProduct
from ..utils import (
    T,
    is_iterable,
    make_list,
)


@dataclass(eq=False, repr=False)
class Variable(CanBehaveLikeAVariable[T]):
    """
    A Variable that queries will assign. The Variable produces results of type `T`.
    """

    _type_: Union[Type, Callable] = field(default=MISSING)
    """
    The result type of the variable. (The value of `T`)
    """
    _name__: str
    """
    The name of the variable.
    """
    _domain_source_: Optional[DomainType] = field(
        default=None, kw_only=True, repr=False
    )
    """
    An optional source for the variable domain. If not given, the global cache of the variable class type will be used
    as the domain, or if kwargs are given the type and the kwargs will be used to inference/infer new values for the
    variable.
    """
    _domain_: ReEnterableLazyIterable = field(
        default_factory=ReEnterableLazyIterable, kw_only=True, repr=False
    )
    """
    The iterable domain of values for this variable.
    """
    _is_inferred_: bool = field(default=False, repr=False)
    """
    Whether this variable domain is inferred or not.
    """

    def __post_init__(self):
        super().__post_init__()

        if self._domain_source_ is not None:
            self._update_domain_(self._domain_source_)

        self._var_ = self

    def _update_domain_(self, domain):
        """
        Set the domain and ensure it is a lazy re-enterable iterable.
        """
        if isinstance(domain, CanBehaveLikeAVariable):
            self._update_children_(domain)
        if isinstance(domain, (ReEnterableLazyIterable, CanBehaveLikeAVariable)):
            self._domain_ = domain
            return
        if not is_iterable(domain):
            domain = [domain]
        self._domain_.set_iterable(domain)

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        """
        A variable either is already bound in sources by other constraints (Symbolic Expressions).,
        or will yield from current domain if exists,
        or has no domain and will instantiate new values by constructing the type if the type is given.
        """

        if self._domain_source_ is not None:
            yield from self._iterator_over_domain_values_(sources)
        elif self._is_inferred_:
            # Means that the variable gets its values from conclusions only.
            return
        else:
            raise VariableCannotBeEvaluated(self)

    def _iterator_over_domain_values_(
        self, sources: Bindings
    ) -> Iterable[OperationResult]:
        """
        Iterate over the values in the variable's domain, yielding OperationResult instances.

        :param sources: The current bindings.
        :return: An Iterable of OperationResults for each value in the domain.
        """
        if isinstance(self._domain_, CanBehaveLikeAVariable):
            yield from self._iterator_over_variable_domain_values_(sources)
        else:
            yield from self._iterator_over_iterable_domain_values_(sources)

    def _iterator_over_variable_domain_values_(self, sources: Bindings):
        """
        Iterate over the values in the variable's domain, where the domain is another variable.

        :param sources: The current bindings.
        :return: An Iterable of OperationResults for each value in the domain.
        """
        for domain in self._domain_._evaluate_(sources, parent=self):
            for v in domain.value:
                bindings = {**sources, **domain.bindings, self._binding_id_: v}
                yield self._build_operation_result_and_update_truth_value_(bindings)

    def _iterator_over_iterable_domain_values_(self, sources: Bindings):
        """
        Iterate over the values in the variable's domain, where the domain is an iterable.

        :param sources: The current bindings.
        :return: An Iterable of OperationResults for each value in the domain.
        """
        for v in self._domain_:
            bindings = {**sources, self._binding_id_: v}
            yield self._build_operation_result_and_update_truth_value_(bindings)

    def _build_operation_result_and_update_truth_value_(
        self, bindings: Bindings
    ) -> OperationResult:
        """
        Build an OperationResult instance and update the truth value based on the bindings.

        :param bindings: The bindings of the result.
        :return: The OperationResult instance with updated truth value.
        """
        self._update_truth_value_(bindings[self._binding_id_])
        return OperationResult(bindings, self._is_false_, self)

    def _replace_child_field_(
        self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        raise ValueError(f"class {self.__class__} does not have children")

    @property
    def _name_(self):
        return self._name__

    @property
    def _is_iterable_(self):
        return is_iterable(next(iter(self._domain_), None))


@dataclass(eq=False, init=False, repr=False)
class Literal(Variable[T]):
    """
    Literals are variables that are not constructed by their type but by their given data.
    """

    def __init__(
        self,
        data: Any,
        name: Optional[str] = None,
        type_: Optional[Type] = None,
        wrap_in_iterator: bool = True,
    ):
        original_data = data
        if wrap_in_iterator:
            data = [data]
        if not type_:
            original_data_lst = make_list(original_data)
            first_value = original_data_lst[0] if len(original_data_lst) > 0 else None
            type_ = type(first_value) if first_value else None
        if name is None:
            if type_:
                name = type_.__name__
            else:
                if isinstance(data, Selectable):
                    name = data._name_
                else:
                    name = type(original_data).__name__
        super().__init__(_name__=name, _type_=type_, _domain_source_=data)


@dataclass(eq=False, repr=False)
class InstantiatedVariable(
    MultiArityExpressionThatPerformsACartesianProduct, Variable[T]
):
    """
    A variable which does not have an explicit domain, but creates new instances using the `_type_` and `_kwargs_`
    that are provided. The `_kwargs_` are variables that can be used to generate combinations of bindings to create
    instances for each combination. By definition this variable is inferred. It also represents Predicates and symbolic
    functions.
    """

    _kwargs_: Dict[str, Any] = field(default_factory=dict)
    """
    The properties of the variable as keyword arguments.
    """
    _child_vars_: Dict[str, SymbolicExpression] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    A dictionary mapping child variable names to variables, these are from the _kwargs_ dictionary. 
    """
    _child_var_id_name_map_: Dict[int, str] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    A dictionary mapping child variable ids to their names. 
    """

    def __post_init__(self):
        self._is_inferred_ = True
        Variable.__post_init__(self)
        self._update_child_vars_from_kwargs_()
        self._operation_children_ = tuple(self._child_vars_.values())
        # This is done here as it uses `_operation_children_`
        MultiArityExpressionThatPerformsACartesianProduct.__post_init__(self)

    def _update_child_vars_from_kwargs_(self):
        """
        Set the child variables from the kwargs dictionary.
        """
        for k, v in self._kwargs_.items():
            self._child_vars_[k] = (
                v if isinstance(v, SymbolicExpression) else Literal(v, name=k)
            )
            self._child_var_id_name_map_[self._child_vars_[k]._binding_id_] = k

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        yield from self._instantiate_using_child_vars_and_yield_results_(sources)

    def _instantiate_using_child_vars_and_yield_results_(
        self, sources: Bindings
    ) -> Iterable[OperationResult]:
        """
        Create new instances of the variable type and using as keyword arguments the child variables values.
        """
        for result in self._evaluate_product_(sources):
            # Build once: unwrapped hashed kwargs for already provided child vars
            kwargs = {
                self._child_var_id_name_map_[id_]: v
                for id_, v in result.bindings.items()
                if id_ in self._child_var_id_name_map_
            }
            instance = self._type_(**kwargs)
            yield self._process_output_and_update_values_(instance, result.bindings)

    def _process_output_and_update_values_(
        self, instance: Any, bindings: Bindings
    ) -> OperationResult:
        """
        Process the predicate/variable instance and get the results.

        :param instance: The created instance.
        :param bindings: The bindings from the child variables.
        :return: The results' dictionary.
        """
        values = {self._binding_id_: instance} | bindings
        return self._build_operation_result_and_update_truth_value_(values)

    def _replace_child_field_(
        self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        for k, v in self._child_vars_.items():
            if v is old_child:
                self._child_vars_[k] = new_child
                break


DomainType = TypingUnion[Iterable[T], None]
"""
The type of the domain used for the variable.
"""
