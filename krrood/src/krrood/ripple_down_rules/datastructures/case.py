from __future__ import annotations

from collections import UserDict
from copy import copy, deepcopy
from dataclasses import dataclass, is_dataclass
from enum import Enum

from pandas import DataFrame
from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase as SQLTable, registry
from typing_extensions import Any, Optional, Dict, Type, Set, Hashable, Union, List, TYPE_CHECKING

from krrood.ripple_down_rules.utils import make_set, row_to_dict, table_rows_as_str, get_value_type_from_type_hint, SubclassJSONSerializer, \
    get_full_class_name, get_type_from_string, make_list, is_iterable, serialize_dataclass, dataclass_to_dict, copy_case

if TYPE_CHECKING:
    from ..rules import Rule
    from .callable_expression import CallableExpression


class Case(UserDict, SubclassJSONSerializer):
    """
    A collection of attributes that represents a set of attributes of a case. This is a dictionary where the keys are
    the names of the attributes and the values are the attributes. All are stored in lower case, and can be accessed
    using the dot notation as well as the dictionary access notation.
    """

    def __init__(self, _obj_type: Type, _id: Optional[Hashable] = None,
                 _name: Optional[str] = None, original_object: Optional[Any] = None, **kwargs):
        """
        Create a new case.

        :param _obj_type: The original type of the object that the case represents.
        :param _id: The id of the case.
        :param _name: The semantic name that describes the case.
        :param kwargs: The attributes of the case.
        """
        super().__init__(kwargs)
        self._original_object = original_object
        self._obj_type: Type = _obj_type
        self._id: Hashable = _id if _id is not None else id(self)
        self._name: str = _name if _name is not None else self._obj_type.__name__

    @classmethod
    def from_obj(cls, obj: Any, obj_name: Optional[str] = None, max_recursion_idx: int = 3) -> Case:
        """
        Create a case from an object.

        :param obj: The object to create a case from.
        :param max_recursion_idx: The maximum recursion index to prevent infinite recursion.
        :param obj_name: The name of the object.
        :return: The case that represents the object.
        """
        return create_case(obj, max_recursion_idx=max_recursion_idx, obj_name=obj_name)

    def __getitem__(self, item: str) -> Any:
        return super().__getitem__(item.lower())

    def __setitem__(self, name: str, value: Any):
        name = name.lower()
        if name in self:
            if isinstance(self[name], list):
                self[name].extend(make_list(value))
            elif isinstance(value, list):
                new_list = make_list(self[name])
                new_list.extend(make_list(value))
                super().__setitem__(name, new_list)
            else:
                super().__setitem__(name, value)
        else:
            super().__setitem__(name, value)
        setattr(self, name, self[name])

    def __contains__(self, item):
        if isinstance(item, (type, Enum)):
            item = item.__name__
        return super().__contains__(item.lower())

    def __delitem__(self, key):
        super().__delitem__(key.lower())

    def __hash__(self):
        return self._id

    def _to_json(self) -> Dict[str, Any]:
        serializable = {k: v for k, v in self.items() if not k.startswith("_")}
        serializable["_id"] = self._id
        serializable["_obj_type"] = get_full_class_name(self._obj_type) if self._obj_type is not None else None
        serializable["_name"] = self._name
        for k, v in serializable.items():
            if isinstance(v, set):
                serializable[k] = {'_type': get_full_class_name(set), 'value': serialize_dataclass(list(v))}
            else:
                serializable[k] = serialize_dataclass(v)
        return {k: v.to_json() if isinstance(v, SubclassJSONSerializer) else v for k, v in serializable.items()}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Case:
        id_ = data.pop("_id")
        obj_type = get_type_from_string(data.pop("_obj_type")) if data["_obj_type"] is not None else None
        name = data.pop("_name")
        for k, v in data.items():
            data[k] = SubclassJSONSerializer.from_json(v)
        return cls(_obj_type=obj_type, _id=id_, _name=name, **data)

    def __deepcopy__(self, memo: Dict[Hashable, Any]) -> Case:
        """
        Create a deep copy of the case.

        :param memo: A dictionary to keep track of objects that have already been copied.
        :return: A deep copy of the case.
        """
        new_case = Case(self._obj_type, _id=self._id, _name=self._name, original_object=self._original_object)
        for k, v in self.items():
            new_case[k] = deepcopy(v)
        return new_case

    def __copy__(self) -> Case:
        """
        Create a shallow copy of the case.

        :return: A shallow copy of the case.
        """
        new_case = Case(self._obj_type, _id=self._id, _name=self._name, original_object=self._original_object)
        for k, v in self.items():
            new_case[k] = copy(v)
        return new_case


@dataclass
class CaseAttributeValue(SubclassJSONSerializer):
    """
    Encapsulates a single value of a case attribute, it adds an id to the value.
    """
    id: Hashable
    """
    The row id of the column value.
    """
    value: Any
    """
    The value of the column.
    """

    def __eq__(self, other):
        if not isinstance(other, CaseAttributeValue):
            return False
        return self.value == other.value

    def __hash__(self):
        return self.id

    def _to_json(self) -> Dict[str, Any]:
        return {"id": self.id, "value": self.value}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> CaseAttributeValue:
        return cls(id=data["id"], value=data["value"])


class CaseAttribute(list, SubclassJSONSerializer):
    nullable: bool = True
    """
    A boolean indicating whether the case attribute can be None or not.
    """
    mutually_exclusive: bool = False
    """
    A boolean indicating whether the case attribute is mutually exclusive or not. (i.e. can only have one value)
    """

    @classmethod
    def from_obj(cls, values: List[Any]) -> CaseAttribute:
        return cls(make_list(values))

    @property
    def as_dict(self) -> Dict[str, Any]:
        """
        Get the case attribute as a dictionary.

        :return: The case attribute as a dictionary.
        """
        return {self.__class__.__name__: self}

    def filter_by(self, condition: CallableExpression) -> CaseAttribute:
        """
        Filter the column by a condition.

        :param condition: The condition to filter by.
        :return: The filtered column.
        """
        return self.__class__([v for v in self if condition(v)])

    def __eq__(self, other):
        if not isinstance(other, list):
            return super().__eq__(make_list(other))
        return super().__eq__(other)

    def __hash__(self):
        return hash(id(self))

    def __str__(self):
        if len(self) == 0:
            return "None"
        return str([v for v in self]) if len(self) > 1 else str(next(iter(self)))

    def _to_json(self) -> Dict[str, Any]:
        return {str(i): v.to_json() if isinstance(v, SubclassJSONSerializer) else v
                for i, v in enumerate(self)}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> CaseAttribute:
        return cls([SubclassJSONSerializer.from_json(v) for _, v in data.items()])


def create_cases_from_dataframe(df: DataFrame, name: Optional[str] = None) -> List[Case]:
    """
    Create cases from a pandas DataFrame.

    :param df: The DataFrame to create cases from.
    :param name: The semantic name of the DataFrame that describes the DataFrame.
    :return: The cases of the DataFrame.
    """
    cases = []
    attribute_names = list(df.columns)
    for row_id, case in df.iterrows():
        case = {col_name: case[col_name].item() for col_name in attribute_names}
        cases.append(Case(DataFrame, _id=row_id, _name=name, **case))
    return cases


def create_case(obj: Any, recursion_idx: int = 0, max_recursion_idx: int = 0,
                obj_name: Optional[str] = None, parent_is_iterable: bool = False) -> Case:
    """
    Create a case from an object.

    :param obj: The object to create a case from.
    :param recursion_idx: The current recursion index.
    :param max_recursion_idx: The maximum recursion index to prevent infinite recursion.
    :param obj_name: The name of the object.
    :param parent_is_iterable: Boolean indicating whether the parent object is iterable or not.
    :return: The case that represents the object.
    """
    obj_name = obj_name or obj.__class__.__name__
    if isinstance(obj, DataFrame):
        return create_cases_from_dataframe(obj, obj_name)
    if isinstance(obj, Case) or (is_dataclass(obj) and not isinstance(obj, SQLTable)):
        return obj
    if ((recursion_idx > max_recursion_idx)
            or (obj.__class__.__module__ == "builtins" and not isinstance(obj, (list, set, dict)))
            or (obj.__class__ in [MetaData, registry])):
        return Case(type(obj), _id=id(obj), _name=obj_name, original_object=obj,
                    **{obj_name or obj.__class__.__name__: make_list(obj) if parent_is_iterable else obj})
    case = Case(type(obj), _id=id(obj), _name=obj_name, original_object=obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            case = create_or_update_case_from_attribute(v, k, obj, obj_name, recursion_idx,
                                                        max_recursion_idx, parent_is_iterable, case)
    for attr in dir(obj):
        if attr.startswith("_") or callable(getattr(obj, attr)):
            continue
        attr_value = getattr(obj, attr)
        case = create_or_update_case_from_attribute(attr_value, attr, obj, obj_name, recursion_idx,
                                                    max_recursion_idx, parent_is_iterable, case)
    return case


def create_or_update_case_from_attribute(attr_value: Any, name: str, obj: Any, obj_name: Optional[str] = None,
                                         recursion_idx: int = 0, max_recursion_idx: int = 1,
                                         parent_is_iterable: bool = False,
                                         case: Optional[Case] = None) -> Case:
    """
    Create or update a case from an attribute of the object that the case represents.

    :param attr_value: The attribute value.
    :param name: The name of the attribute.
    :param obj: The parent object of the attribute.
    :param obj_name: The parent object name.
    :param recursion_idx: The recursion index to prevent infinite recursion.
    :param max_recursion_idx: The maximum recursion index.
    :param parent_is_iterable: Boolean indicating whether the parent object is iterable or not.
    :param case: The case to update.
    :return: The updated/created case.
    """
    if case is None:
        case = Case(type(obj), _id=id(obj), _name=obj_name, original_object=obj)
    if isinstance(attr_value, (dict, UserDict)):
        case.update({f"{obj_name}.{k}": v for k, v in attr_value.items()})
    if hasattr(attr_value, "__iter__") and not isinstance(attr_value, str):
        column = create_case_attribute_from_iterable_attribute(attr_value, name, obj, obj_name,
                                                               recursion_idx=recursion_idx + 1,
                                                               max_recursion_idx=max_recursion_idx)
        case[name] = column
    else:
        case[name] = make_list(attr_value) if parent_is_iterable else attr_value
    return case


def create_case_attribute_from_iterable_attribute(attr_value: Any, name: str, obj: Any, obj_name: Optional[str] = None,
                                                  recursion_idx: int = 0,
                                                  max_recursion_idx: int = 1) -> CaseAttribute:
    """
    Get a case attribute from an iterable attribute.

    :param attr_value: The iterable attribute to get the case from.
    :param name: The name of the case.
    :param obj: The parent object of the iterable.
    :param obj_name: The parent object name.
    :param recursion_idx: The recursion index to prevent infinite recursion.
    :param max_recursion_idx: The maximum recursion index.
    :return: A case attribute that represents the original iterable attribute.
    """
    values = list(attr_value.values()) if isinstance(attr_value, (dict, UserDict)) else attr_value
    try:
        _type = type(list(values)[0]) if len(values) > 0 else get_value_type_from_type_hint(name, obj)
    except ValueError:
        _type = None
    attr_case = Case(_type, _id=id(attr_value), _name=name, original_object=attr_value)
    case_attr = CaseAttribute(values)
    for idx, val in enumerate(values):
        sub_attr_case = create_case(val, recursion_idx=recursion_idx,
                                    max_recursion_idx=max_recursion_idx,
                                    obj_name=name, parent_is_iterable=True)
        attr_case.update(sub_attr_case)
    for sub_attr, val in attr_case.items():
        try:
            setattr(case_attr, sub_attr, val)
        except AttributeError:
            pass
    return case_attr


def show_current_and_corner_cases(case: Any, targets: Optional[Dict[str, Any]] = None,
                                  current_conclusions: Optional[Dict[str, Any]] = None,
                                  last_evaluated_rule: Optional[Rule] = None) -> str:
    """
    Get the the data to show of the new case and if last evaluated rule exists also show that of the corner case.

    :param case: The new case.
    :param targets: The target attribute of the case.
    :param current_conclusions: The current conclusions of the case.
    :param last_evaluated_rule: The last evaluated rule in the RDR.
    """
    corner_case = None
    targets = {f"target_{name}": value for name, value in targets.items()} if targets else {}
    current_conclusions = {name: value for name, value in current_conclusions.items()} if current_conclusions else {}
    information = ""
    if last_evaluated_rule:
        action = "Refinement" if last_evaluated_rule.fired else "Alternative"
        information += f"{action} needed for rule: {last_evaluated_rule}\n"
        corner_case = last_evaluated_rule.corner_case

    corner_row_dict = None
    if isinstance(case, SQLTable):
        case_dict = row_to_dict(case)
        if last_evaluated_rule and last_evaluated_rule.fired:
            corner_row_dict = row_to_dict(last_evaluated_rule.corner_case)
    elif is_dataclass(case):
        case_dict = dataclass_to_dict(case)
        if last_evaluated_rule and last_evaluated_rule.fired:
            corner_row_dict = dataclass_to_dict(last_evaluated_rule.corner_case)
    else:
        case_dict = copy_case(case)
        if last_evaluated_rule and last_evaluated_rule.fired:
            corner_row_dict = copy_case(corner_case)

    case_dict.update(targets)
    case_dict.update(current_conclusions)
    all_table_rows = [case_dict]
    if corner_row_dict:
        corner_conclusion = last_evaluated_rule.conclusion(case)
        corner_row_dict.update({corner_conclusion.__class__.__name__: corner_conclusion})
        all_table_rows.append(corner_row_dict)
    information += "\n" + "=" * 50 + "\n"
    information += "\n" + table_rows_as_str(all_table_rows) + "\n"
    return information
