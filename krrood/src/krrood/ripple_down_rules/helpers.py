from __future__ import annotations

import importlib
import inspect
import os
from functools import wraps
from types import ModuleType

from typing_extensions import Tuple, Sequence, get_type_hints, Set
from typing_extensions import Type, Optional, Callable, Any, Dict, TYPE_CHECKING, Union, Iterable

from krrood.ripple_down_rules.datastructures.case import create_case, Case
from krrood.ripple_down_rules.datastructures.dataclasses import CaseQuery
from krrood.ripple_down_rules.utils import calculate_precision_and_recall, get_method_args_as_dict
from krrood.ripple_down_rules.utils import get_func_rdr_model_name, copy_case, make_set, update_case_in_case_query

if TYPE_CHECKING:
    from .rdr import RippleDownRules


def general_rdr_classify(classifiers_dict: Dict[str, Union[ModuleType, RippleDownRules]],
                         case: Any, modify_original_case: bool = False,
                         case_query: Optional[CaseQuery] = None) -> Dict[str, Any]:
    """
    Classify a case by going through all classifiers and adding the categories that are classified,
     and then restarting the classification until no more categories can be added.

    :param classifiers_dict: A dictionary mapping conclusion types to the classifiers that produce them.
    :param case: The case to classify.
    :param modify_original_case: Whether to modify the original case or create a copy and modify it.
    :param case_query: The case query to extract metadata from if needed.
    :return: The categories that the case belongs to.
    """
    conclusions = {}
    case = create_case(case)
    case_cp = copy_case(case) if not modify_original_case else case
    while True:
        new_conclusions = {}
        for attribute_name, rdr in classifiers_dict.items():
            pred_atts = rdr.classify(case_cp, case_query=case_query)
            if pred_atts is None and type(None) not in rdr.conclusion_type:
                continue
            if rdr.mutually_exclusive:
                if attribute_name not in conclusions or \
                        (attribute_name in conclusions and conclusions[attribute_name] != pred_atts):
                    conclusions[attribute_name] = pred_atts
                    new_conclusions[attribute_name] = pred_atts
            else:
                pred_atts = make_set(pred_atts)
                if attribute_name in conclusions:
                    pred_atts = pred_atts - conclusions[attribute_name]
                if len(pred_atts) > 0:
                    new_conclusions[attribute_name] = pred_atts
                    if attribute_name not in conclusions:
                        conclusions[attribute_name] = set()
                    conclusions[attribute_name].update(pred_atts)
            if attribute_name in new_conclusions:
                temp_case_query = CaseQuery(case_cp, attribute_name, rdr.conclusion_type, rdr.mutually_exclusive)
                update_case_in_case_query(temp_case_query, new_conclusions)
        if len(new_conclusions) == 0 or len(classifiers_dict) == 1 and list(classifiers_dict.values())[
            0].mutually_exclusive:
            break
    return conclusions


def is_matching(classifier: Callable[[Any], Any], case_query: CaseQuery,
                pred_cat: Optional[Dict[str, Any]] = None) -> bool:
    """
    :param classifier: The RDR classifier to check the prediction of.
    :param case_query: The case query to check.
    :param pred_cat: The predicted category.
    :return: Whether the classifier prediction is matching case_query target or not.
    """
    if case_query.target is None:
        return False
    if pred_cat is None:
        pred_cat = classifier(case_query.case)
    if not isinstance(pred_cat, dict):
        pred_cat = {case_query.attribute_name: pred_cat}
    target = {case_query.attribute_name: case_query.target_value}
    precision, recall = calculate_precision_and_recall(pred_cat, target)
    return all(recall) and all(precision)


def load_or_create_func_rdr_model(func, model_dir: str, rdr_type: Type[RippleDownRules],
                                  **rdr_kwargs) -> RippleDownRules:
    """
    Load the RDR model of the function if it exists, otherwise create a new one.

    :param func: The function to load the model for.
    :param model_dir: The directory where the model is stored.
    :param rdr_type: The type of the RDR model to load.
    :param rdr_kwargs: Additional arguments to pass to the RDR constructor in the case of a new model.
    """
    model_name = get_func_rdr_model_name(func)
    model_path = os.path.join(model_dir, model_name, f"{model_name}.py")
    if os.path.exists(model_path):
        rdr = rdr_type.load(load_dir=model_dir, model_name=model_name)
    else:
        rdr = rdr_type(**rdr_kwargs)
    return rdr


def update_case_and_conclusions_with_rule_output(case: Case, conclusions: Set[Any],
                                                 output: Any, attribute_name: str,
                                                 conclusion_type: Tuple[Type, ...], mutually_exclusive: bool):
    """
    Updates the case and conclusions with the rule output.

    :param case: The case to update.
    :param conclusions: The current conclusions of the case.
    :param output: The output of the rule.
    :param attribute_name: The name of the attribute to update.
    :param conclusion_type: The type of the conclusion to update.
    :param mutually_exclusive: Whether the rule belongs to a mutually exclusive RDR.
    """
    output = make_set(output)
    new_conclusions = update_case_with_conclusion_output(
        case, output, attribute_name, conclusion_type, mutually_exclusive
    )
    if len(new_conclusions) > 0:
        conclusions.update(new_conclusions)


def update_case_with_conclusion_output(case: Case, output: Iterable[Any], attribute_name: str,
                                       conclusion_type: Tuple[Type, ...],
                                       mutually_exclusive: bool) -> Set[Any]:
    """
    :param case: The case to update.
    :param output: The output of the conclusion to add to the case.
    :param attribute_name: The name of the attribute to update.
    :param conclusion_type: The type of the conclusion to update.
    :param mutually_exclusive: Whether the rule belongs to a mutually exclusive RDR.
    :return: Whether the case was updated or not.
    """
    new_conclusions = make_set(output) - make_set(getattr(case, attribute_name, []))
    if len(new_conclusions) == 0:
        return new_conclusions
    temp_case_query = CaseQuery(case, attribute_name, conclusion_type,
                                mutually_exclusive=mutually_exclusive)
    if not isinstance(output, Dict):
        output = {attribute_name: output}
    update_case_in_case_query(temp_case_query, output)
    return new_conclusions


def get_an_updated_case_copy(case: Case, conclusion: Callable, attribute_name: str, conclusion_type: Tuple[Type, ...],
                             mutually_exclusive: bool) -> Case:
    """
    :param case: The case to copy and update.
    :param conclusion: The conclusion to add to the case.
    :param attribute_name: The name of the attribute to update.
    :param conclusion_type: The type of the conclusion to update.
    :param mutually_exclusive: Whether the rule belongs to a mutually exclusive RDR.
    :return: A copy of the case updated with the given conclusion.
    """
    case_cp = copy_case(case)
    temp_case_query = CaseQuery(case_cp, attribute_name, conclusion_type,
                                mutually_exclusive=mutually_exclusive)
    output = conclusion(case_cp)
    if not isinstance(output, Dict):
        output = {attribute_name: output}
    update_case_in_case_query(temp_case_query, output)
    return case_cp


def enable_gui():
    """
    Enable the GUI for Ripple Down Rules if available.
    """
    try:
        from .user_interface.gui import RDRCaseViewer
        viewer = RDRCaseViewer()
    except ImportError:
        pass


def create_case_from_method(func: Callable,
                            func_output: Dict[str, Any],
                            *args, **kwargs) -> Tuple[Case, Dict[str, Any]]:
    """
    Create a Case from the function and its arguments.

    :param func: The function to create a case from.
    :param func_output: A dictionary containing the output of the function, where the key is the output name.
    :param args: The positional arguments of the function.
    :param kwargs: The keyword arguments of the function.
    :return: A Case object representing the case.
    """
    case_dict = get_method_args_as_dict(func, *args, **kwargs)
    case_dict.update(func_output)
    case_name = get_func_rdr_model_name(func)
    return Case(dict, id(case_dict), case_name, case_dict, **case_dict), case_dict


class MockRDRDecorator:
    def __init__(self, models_dir: str):
        self.models_dir = models_dir

    def decorator(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Optional[Any]:
            model_dir = get_func_rdr_model_name(func, include_file_name=True)
            model_name = get_func_rdr_model_name(func, include_file_name=False)
            rdr = importlib.import_module(os.path.join(self.models_dir, model_dir, f"{model_name}_rdr.py"))
            func_output = {"output_": func(*args, **kwargs)}
            case, case_dict = create_case_from_method(func, func_output, *args, **kwargs)
            return rdr.classify(case)

        return wrapper


def create_case_query_from_method(func: Callable,
                                  func_output: Dict[str, Any],
                                  output_type: Sequence[Type],
                                  mutual_exclusive: bool,
                                  func_args: Tuple[Any, ...], func_kwargs: Dict[str, Any],
                                  case: Optional[Case] = None,
                                  case_dict: Optional[Dict[str, Any]] = None,
                                  scenario: Optional[Callable] = None,
                                  this_case_target_value: Optional[Any] = None, ) -> CaseQuery:
    """
    Create a CaseQuery from the function and its arguments.

    :param func: The function to create a case from.
    :param func_output: The output of the function as a dictionary, where the key is the output name.
    :param output_type: The type of the output as a sequence of types.
    :param mutual_exclusive: If True, the output types are mutually exclusive.
    :param func_args: The positional arguments of the function.
    :param func_kwargs: The keyword arguments of the function.
    :param case: The case to create.
    :param case_dict: The dictionary of the case.
    :param scenario: The scenario that produced the given case.
    :param this_case_target_value: The target value for the case.
    :return: A CaseQuery object representing the case.
    """
    output_type = make_set(output_type)
    if case is None or case_dict is None:
        case, case_dict = create_case_from_method(func, func_output, *func_args, **func_kwargs)
    scope = func.__globals__
    scope.update(case_dict)
    try:
        func_args_type_hints = get_type_hints(func)
    except NameError:
        # use inspect to get the type hints if get_type_hints fails
        func_args_type_hints = {k: v.annotation for k, v in inspect.signature(func).parameters.items()
                                if v.annotation is not inspect._empty}
        # add return type hint from the function signature
        return_annotation = inspect.signature(func).return_annotation
        if return_annotation is not inspect._empty:
            func_args_type_hints['return'] = return_annotation
        types_dict = {t.__name__: t for t in output_type}
        scope.update(types_dict)
        types_names = list(types_dict.keys())
        for k, v in func_args_type_hints.items():
            for t_name in types_names:
                if isinstance(v, str) and t_name in v:
                    func_args_type_hints[k] = eval(v, scope)
    output_name = list(func_output.keys())[0]
    func_args_type_hints.update({output_name: Union[tuple(output_type)]})
    return CaseQuery(case, output_name, tuple(output_type),
                     mutual_exclusive, scope=scope, scenario=scenario, this_case_target_value=this_case_target_value,
                     is_function=True, function_args_type_hints=func_args_type_hints)
