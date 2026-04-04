from __future__ import annotations

import inspect
import uuid
from dataclasses import dataclass, field

from colorama import Fore, Style
from omegaconf import MISSING
from sqlalchemy.orm import DeclarativeBase as SQLTable
from typing_extensions import Any, Optional, Dict, Type, Tuple, Union, List, Set, Callable, TYPE_CHECKING

from krrood.ripple_down_rules.datastructures.callable_expression import CallableExpression
from krrood.ripple_down_rules.datastructures.case import create_case, Case
from krrood.ripple_down_rules.utils import copy_case, make_list, make_set, get_origin_and_args_from_type_hint, render_tree, \
    get_function_representation, get_method_object_from_pytest_request, typing_to_python_type

if TYPE_CHECKING:
    from ..rdr import RippleDownRules
    from ..rules import Rule


@dataclass
class CaseQuery:
    """
    This is a dataclass that represents an attribute of an object and its target value. If attribute name is
    not provided, it will be inferred from the attribute itself or from the attribute type or from the target value,
    depending on what is provided.
    """
    original_case: Any
    """
    The case that the attribute belongs to.
    """
    attribute_name: str
    """
    The name of the attribute.
    """
    _attribute_types: Tuple[Type, ...]
    """
    The type(s) of the attribute.
    """
    mutually_exclusive: bool
    """
    Whether the attribute can only take one value (i.e. True) or multiple values (i.e. False).
    """
    case_factory: Optional[Callable[[], Any]] = None
    """
    The factory method that can be used to recreate the original case.
    """
    case_factory_idx: Optional[int] = None
    """
    This is used when the case factory is a list of cases, this index is used to select the case from the list.
    """
    case_conf: Optional[CaseConf] = None
    """
    The case configuration that is used to (re)create the original case, recommended to be used when you want to 
    the case to persist in the rule base, this would allow it to be used for merging with other similar conclusion RDRs.
    """
    scenario: Optional[Callable] = None
    """
    The executable scenario is the root callable that recreates the situation that the case is 
    created in, for example, when the case is created from a test function, this would be the test function itself.
    """
    this_case_target_value: Optional[Any] = None
    """
    The non relational case query instance target value.
    """
    _target: Optional[CallableExpression] = None
    """
    The relational target (the evaluatable conclusion of the rule) which is a callable expression that varies with
     the case.
    """
    default_value: Optional[Any] = None
    """
    The default value of the attribute. This is used when the target value is not provided.
    """
    scope: Optional[Dict[str, Any]] = field(default_factory=lambda: inspect.currentframe().f_back.f_back.f_globals)
    """
    The global scope of the case query. This is used to evaluate the conditions and prediction, and is what is available
    to the user when they are prompted for input. If it is not provided, it will be set to the global scope of the
    caller.
    """
    _case: Optional[Union[Case, SQLTable]] = None
    """
    The created case from the original case that the attribute belongs to.
    """
    _target_value: Optional[Any] = None
    """
    The target value of the case query. (This is the result of the target expression evaluation on the case.)
    """
    conditions: Optional[CallableExpression] = None
    """
    The conditions that must be satisfied for the target value to be valid.
    """
    is_function: bool = False
    """
    Whether the case is a dict representing the arguments of an actual function or not,
    most likely means it came from RDRDecorator, the the rdr takes function arguments and outputs the function output.
    """
    function_args_type_hints: Optional[Dict[str, Type]] = None
    """
    The type hints of the function arguments. This is used to recreate the function signature.
    """
    rdr: Optional[RippleDownRules] = None
    """
    The Ripple Down Rules that was used to answer the case query.
    """

    def render_rule_tree(self, filepath: Optional[str] = None, view: bool = False):
        if self.rdr is None:
            return
        render_tree(self.rdr.start_rule, use_dot_exporter=True, filename=filepath, view=view)

    @property
    def current_value_str(self):
        return (f"{Fore.MAGENTA}Current value of {Fore.CYAN}{self.name}{Fore.MAGENTA} of type(s) "
                f"{Fore.CYAN}({self.core_attribute_type_str}){Fore.MAGENTA}: "
                f"{Fore.WHITE}{self.current_value}{Style.RESET_ALL}")

    @property
    def current_value(self) -> Any:
        """
        :return: The current value of the attribute.
        """
        if not hasattr(self.case, self.attribute_name):
            return None

        attr_value = getattr(self.case, self.attribute_name)

        if attr_value is None:
            return attr_value
        elif self.mutually_exclusive:
            return attr_value
        else:
            return list({v for v in make_list(attr_value) if isinstance(v, self.core_attribute_type)})

    @property
    def case_type(self) -> Type:
        """
        :return: The type of the case that the attribute belongs to.
        """
        if self.is_function:
            if self.function_args_type_hints is not None:
                func_args = [arg for name, arg in self.function_args_type_hints.items() if name != 'return']
                case_type_args = Union[tuple(func_args)]
            else:
                case_type_args = Any
            return Dict[str, case_type_args]
        else:
            return self.original_case._obj_type if isinstance(self.original_case, Case) else type(self.original_case)

    @property
    def case(self) -> Any:
        """
        :return: The case that the attribute belongs to.
        """
        if self._case is not None:
            return self._case
        elif not isinstance(self.original_case, Case):
            self._case = create_case(self.original_case, max_recursion_idx=3)
        else:
            self._case = self.original_case
        return self._case

    @case.setter
    def case(self, value: Any):
        """
        Set the case that the attribute belongs to.
        """
        if not isinstance(value, (Case, SQLTable)):
            raise ValueError("The case must be a Case or SQLTable object.")
        self._case = value

    @property
    def attribute_type_hint(self) -> str:
        """
        :return: The type hint of the attribute as a typing object.
        """
        if len(self.core_attribute_type) > 1:
            attribute_types_str = f"Union[{', '.join([t.__name__ for t in self.core_attribute_type])}]"
        else:
            attribute_types_str = self.core_attribute_type[0].__name__
        if not self.mutually_exclusive:
            return f"List[{attribute_types_str}]"
        else:
            return attribute_types_str

    @property
    def core_attribute_type_str(self) -> str:
        """
        :return: The names of the core types of the attribute.
        """
        return ','.join([t.__name__ for t in self.core_attribute_type])

    @property
    def core_attribute_type(self) -> Tuple[Type, ...]:
        """
        :return: The core type of the attribute.
        """
        return tuple(t for t in self.attribute_type if t not in (set, list))

    @property
    def attribute_type(self) -> Tuple[Type, ...]:
        """
        :return: The type of the attribute.
        """
        if not isinstance(self._attribute_types, tuple):
            self._attribute_types = tuple(make_set(self._attribute_types))
        att_types = set()
        for att_type in self._attribute_types:
            origin, args = get_origin_and_args_from_type_hint(att_type)
            if origin is not None:
                att_types.add(origin)
                if origin in (list, set, tuple, type, List, Set, Union, Tuple, Type):
                    att_types.update(make_set(args))
                elif origin in (dict, Dict):
                    # ignore the key type
                    if args and len(args) > 1:
                        att_types.update(make_set(args[1]))
            else:
                att_types.add(typing_to_python_type(att_type))
        self._attribute_types = tuple(att_types)
        if not self.mutually_exclusive and (list not in self._attribute_types):
            self._attribute_types = tuple(make_list(self._attribute_types) + [set, list])
        return self._attribute_types

    @attribute_type.setter
    def attribute_type(self, value: Type):
        """
        Set the type of the attribute.
        """
        self._attribute_types = tuple(make_list(value))

    @property
    def name(self):
        """
        :return: The name of the case query.
        """
        return f"{self.case_name}.{self.attribute_name}"

    @property
    def case_name(self) -> str:
        """
        :return: The name of the case.
        """
        return self.case._name if isinstance(self.case, Case) else self.case.__class__.__name__

    @property
    def target(self) -> Optional[CallableExpression]:
        """
        :return: The target expression of the attribute.
        """
        if (self._target is not None) and (not isinstance(self._target, CallableExpression)):
            self._target = CallableExpression(conclusion=self._target, conclusion_type=self.attribute_type,
                                              scope=self.scope, mutually_exclusive=self.mutually_exclusive)
        return self._target

    @target.setter
    def target(self, value: Optional[CallableExpression]):
        """
        Set the target expression of the attribute.
        """
        if value is not None and not isinstance(value, (CallableExpression, str)):
            raise ValueError("The target must be a CallableExpression or a string.")
        self._target = value
        self.update_target_value()

    @property
    def target_value(self) -> Any:
        """
        :return: The target value of the case query.
        """
        if self._target_value is None:
            self.update_target_value()
        return self._target_value

    def update_target_value(self):
        """
        Update the target value of the case query.
        """
        if isinstance(self.target, CallableExpression):
            self._target_value = self.target(self.case)
        else:
            self._target_value = self.target

    def __str__(self):
        header = f"CaseQuery: {self.name}"
        target = f"Target: {self.name} |= {self.target if self.target is not None else '?'}"
        conditions = f"Conditions: {self.conditions if self.conditions is not None else '?'}"
        return "\n".join([header, target, conditions])

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return CaseQuery(self.original_case, self.attribute_name, self.attribute_type,
                         self.mutually_exclusive, _target=self.target, default_value=self.default_value,
                         scope=self.scope, _case=copy_case(self.case), _target_value=self.target_value,
                         conditions=self.conditions, is_function=self.is_function,
                         function_args_type_hints=self.function_args_type_hints,
                         case_factory=self.case_factory, case_factory_idx=self.case_factory_idx,
                         case_conf=self.case_conf, scenario=self.scenario, rdr=self.rdr)


@dataclass
class CaseConf:
    factory_method: Callable[[Any], Any] = MISSING

    def create(self) -> Any:
        return self.factory_method()


@dataclass
class CaseFactoryMetaData:
    factory_method: Optional[Callable[[Optional[CaseConf]], Any]] = None
    factory_idx: Optional[int] = None
    case_conf: Optional[CaseConf] = None
    scenario: Optional[Callable] = None
    pytest_request: Optional[Callable] = field(hash=False, compare=False, default=None)
    this_case_target_value: Optional[Any] = None

    def __post_init__(self):
        if self.pytest_request is not None and self.scenario is None:
            self.scenario = get_method_object_from_pytest_request(self.pytest_request)

    @classmethod
    def from_case_query(cls, case_query: CaseQuery) -> CaseFactoryMetaData:
        return cls(factory_method=case_query.case_factory, factory_idx=case_query.case_factory_idx,
                   case_conf=case_query.case_conf, scenario=case_query.scenario)

    def __repr__(self):
        factory_method_repr = None
        scenario_repr = None
        if self.factory_method is not None:
            factory_method_repr = get_function_representation(self.factory_method)
        if self.scenario is not None:
            scenario_repr = get_function_representation(self.scenario)
        return (f"CaseFactoryMetaData("
                f"factory_method={factory_method_repr}, "
                f"factory_idx={self.factory_idx}, "
                f"case_conf={self.case_conf}, "
                f"scenario={scenario_repr}, "
                f"this_case_target_value={self.this_case_target_value})")

    def __str__(self):
        return self.__repr__()


@dataclass
class RDRConclusion:
    """
    This dataclass represents a conclusion of a Ripple Down Rule.
    It contains the conclusion expression, the type of the conclusion, and the scope in which it is evaluated.
    """
    _conclusion: Any
    """
    The conclusion value.
    """
    _frozen_case: Any
    """
    The frozen case that the conclusion was made for.
    """
    _rule: Rule
    """
    The rule that gave this conclusion.
    """
    _rdr: RippleDownRules
    """
    The Ripple Down Rules that classified the case and produced this conclusion.
    """
    _id: int = field(default_factory=lambda: uuid.uuid4().int)
    """
    The unique identifier of the conclusion.
    """
    def __getattribute__(self, name: str) -> Any:
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        else:
            conclusion = object.__getattribute__(self, "_conclusion")

            value = getattr(conclusion, name)

            self._record_dependency(name)

            return value

    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            setattr(self._wrapped, name, value)

    def _record_dependency(self, attr_name):
        # Inspect stack to find instance of CallableExpression
        for frame_info in inspect.stack():
            func_name = frame_info.function
            local_self = frame_info.frame.f_locals.get("self", None)
            if (
                    func_name == "__call__" and
                    local_self is not None and
                    type(local_self) is CallableExpression
            ):
                self._used_in_tracker = True
                print("RDRConclusion used inside CallableExpression")
                break

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, RDRConclusion):
            return False
        return self.id == other.id
