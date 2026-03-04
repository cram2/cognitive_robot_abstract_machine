from __future__ import annotations

import builtins
import codecs
import copyreg
import importlib
import itertools
import json
import os
import re
import shutil
import sys
import threading
import uuid
from collections import UserDict, defaultdict
from collections.abc import Iterator
from copy import deepcopy, copy
from dataclasses import is_dataclass, fields
from enum import Enum
from functools import lru_cache
from os.path import dirname
from pathlib import Path
from subprocess import check_call
from tempfile import NamedTemporaryFile
from textwrap import dedent
from types import NoneType, ModuleType
import inspect

import six
from graphviz import Source
from sqlalchemy.exc import NoInspectionAvailable
from krrood.ripple_down_rules import logger

try:
    import matplotlib
    from matplotlib import pyplot as plt

    Figure = plt.Figure
except ImportError as e:
    matplotlib = None
    plt = None
    Figure = None
    logger.debug(f"{e}: matplotlib is not installed")

try:
    import networkx as nx
except ImportError as e:
    nx = None
    logger.debug(f"{e}: networkx is not installed")

import requests
from anytree import Node, RenderTree, PreOrderIter
from sqlalchemy import MetaData, inspect as sql_inspect
from sqlalchemy.orm import (
    Mapped,
    registry,
    class_mapper,
    DeclarativeBase as SQLTable,
    Session,
)
from tabulate import tabulate
from typing_extensions import (
    Callable,
    Set,
    Any,
    Type,
    Dict,
    TYPE_CHECKING,
    get_type_hints,
    get_origin,
    get_args,
    Tuple,
    Optional,
    List,
    Union,
    Self,
    ForwardRef,
    Iterable,
    Sequence,
)

if TYPE_CHECKING:
    from .datastructures.case import Case
    from .datastructures.dataclasses import CaseQuery

import ast


class IDGenerator:
    """
    A class that generates incrementing, unique IDs and caches them for every object this is called on.
    """

    _counter = 0
    """
    The counter of the unique IDs.
    """

    # @lru_cache(maxsize=None)
    def __call__(self, obj: Any) -> int:
        """
        Creates a unique ID and caches it for every object this is called on.

        :param obj: The object to generate a unique ID for, must be hashable.
        :return: The unique ID.
        """
        self._counter += 1
        return self._counter


def filter_data(data, selected_indices):
    data = iter(data)
    prev = -1
    encountered_indices = set()
    for idx in selected_indices:
        if idx in encountered_indices:
            continue
        encountered_indices.add(idx)
        skip = idx - prev - 1
        data = itertools.islice(data, skip, None)
        try:
            yield next(data)
        except StopIteration:
            break
        prev = idx


def fill_in_missing_kwargs(func: Callable, **kwargs) -> Dict[str, Any]:
    """
    Fill in missing kwargs with kwargs from the function signature.

    :param func: The function object.
    :param kwargs: The kwargs to complete.
    :return: The complete kwargs.
    """
    original_kwargs = {
        pname: p
        for pname, p in inspect.signature(func).parameters.items()
        if p.default != inspect._empty
    }
    for og_kwarg in original_kwargs:
        if og_kwarg not in kwargs:
            kwargs[og_kwarg] = original_kwargs[og_kwarg].default
    return kwargs


def get_and_import_python_modules_in_a_package(
    file_paths: List[str], parent_package_name: Optional[str] = None
) -> List[Optional[ModuleType]]:
    """
    :param file_paths: The paths to the python files to import.
    :param parent_package_name: The name of the parent package to use for relative imports.
    :return: The imported modules.
    """
    package_path = dirname(file_paths[0])
    package_import_path = get_import_path_from_path(package_path)
    file_names = [Path(file_path).name.replace(".py", "") for file_path in file_paths]
    module_import_paths = [
        f"{package_import_path}.{file_name}" if package_import_path else file_name
        for file_name in file_names
    ]
    modules = [
        (
            importlib.import_module(module_import_path, package=parent_package_name)
            if os.path.exists(file_paths[i])
            else None
        )
        for i, module_import_path in enumerate(module_import_paths)
    ]
    for module in modules:
        if module is not None:
            importlib.reload(module)
    return modules


def get_and_import_python_module(
    python_file_path: str,
    package_import_path: Optional[str] = None,
    parent_package_name: Optional[str] = None,
) -> ModuleType:
    """
    :param python_file_path: The path to the python file to import.
    :param package_import_path: The import path of the package that contains the python file.
    :param parent_package_name: The name of the parent package to use for relative imports.
    :return: The imported module.
    """
    if package_import_path is None:
        package_path = dirname(python_file_path)
        package_import_path = get_import_path_from_path(package_path)
    file_name = Path(python_file_path).name.replace(".py", "")
    module_import_path = (
        f"{package_import_path}.{file_name}" if package_import_path else file_name
    )
    module = importlib.import_module(module_import_path, package=parent_package_name)
    importlib.reload(module)
    return module


def str_to_snake_case(snake_str: str) -> str:
    """
    Convert a string to snake case.

    :param snake_str: The string to convert.
    :return: The converted string.
    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", snake_str)
    s1 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    # remove redundant underscores
    s1 = re.sub(r"_{2,}", "_", s1)
    # remove leading and trailing underscores
    s1 = re.sub(r"^_|_$", "", s1)
    return s1


def are_results_subclass_of_types(result_types: List[Any], types_: List[Type]) -> bool:
    """
    Check if all results are subclasses of the given types.

    :param result_types: The list of result types to check.
    :param types_: The list of types to check against.
    :return: True if all results are subclasses of the given types, False otherwise.
    """
    for rt in result_types:
        if not any(issubclass(rt, t) for t in types_):
            return False
    return True


def get_origin_type_of_function_output(func: Callable) -> Optional[Type]:
    """
    Get the origin type of a function return type.

    :param func: The function object.
    """
    origin_type: Optional[Type] = None
    try:
        origin_type = get_origin(get_type_hints(func)["return"])
    except NameError:
        return_annotation = inspect.signature(func).return_annotation
        if any(
            return_annotation.startswith(t) for t in ["List", "list", "typing.List"]
        ):
            origin_type = list
        elif any(
            return_annotation.startswith(t) for t in ["Type", "type", "typing.Type"]
        ):
            origin_type = type
    if origin_type:
        origin_type = get_type_from_type_hint(origin_type)
    return origin_type


def get_arg_type_of_function_output(
    func: Callable, output_type: Type, *func_args, package_name: Optional[str] = None
):
    if output_type is Self and len(func_args) > 0:
        func_class = get_method_class_if_exists(func, *func_args)
        if func_class is not None:
            return func_class
        else:
            raise ValueError(
                f"The function {func} is not a method of a class,"
                f" and the output type is {Self}."
            )
    elif type(output_type) is str:
        # If the type is a string, it might be a forward reference or a type hint.
        # We can try to resolve it using the current module's globals.
        if output_type in func.__globals__:
            return func.__globals__[output_type]
        else:
            if package_name:
                # If a package name is provided, try to resolve it as a module import.
                try:
                    module = sys.modules[package_name]
                    return module.__dict__[output_type]
                except (ImportError, KeyError):
                    pass
            try:
                module_name = ".".join(output_type.split(".")[:-1])
                if package_name:
                    module_name = f"{package_name}.{module_name}"
                type_name = output_type.split(".")[-1]
                return importlib.import_module(module_name).__dict__[type_name]
            except (ImportError, KeyError):
                raise ValueError(
                    f"Output type '{output_type}' could not be resolved in the current scope."
                )
    else:
        return output_type


def get_imports_from_scope(scope: Dict[str, Any]) -> List[str]:
    """
    Get the imports from the given scope.

    :param scope: The scope to get the imports from.
    :return: The imports as a string.
    """
    return get_imports_from_types(list(scope.values()))


def extract_imports(
    file_path: Optional[str] = None,
    tree: Optional[ast.AST] = None,
    package_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract imports from a Python file or an AST tree.

    :param file_path: The path to the Python file to extract imports from.
    :param tree: An AST tree to extract imports from. If provided, file_path is ignored.
    :param package_name: The name of the package to use for relative imports.
    """
    if tree is None:
        if file_path is None:
            raise ValueError("Either file_path or tree must be provided")
        with open(file_path, "r") as f:
            tree = ast.parse(f.read(), filename=file_path)

    scope = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                asname = alias.asname or alias.name
                try:
                    scope[asname] = importlib.import_module(
                        module_name, package=package_name
                    )
                except ImportError as e:
                    logger.warning(f"Could not import {module_name}: {e}")
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module
            for alias in node.names:
                name = alias.name
                asname = alias.asname or name
                try:
                    if node.level > 0:  # Handle relative imports
                        package_name = get_import_path_from_path(
                            Path(
                                os.path.join(file_path, *[".."] * node.level)
                            ).resolve()
                        )
                    if (
                        package_name is not None and node.level > 0
                    ):  # Handle relative imports
                        module_rel_path = Path(
                            os.path.join(file_path, *[".."] * node.level, module_name)
                        ).resolve()
                        idx = str(module_rel_path).rfind(package_name)
                        if idx != -1:
                            module_name = str(module_rel_path)[idx:].replace(
                                os.path.sep, "."
                            )
                    try:
                        module = importlib.import_module(
                            module_name, package=package_name
                        )
                    except ModuleNotFoundError:
                        module = importlib.import_module(
                            f"{package_name}.{module_name}"
                        )
                    if name == "*":
                        scope.update(module.__dict__)
                    else:
                        scope[asname] = getattr(module, name)
                except (ImportError, AttributeError) as e:
                    logger.warning(
                        f"Could not import {module_name}: {e} while extracting imports from {file_path}"
                    )

    return scope


def extract_function_or_class_file(
    file_path: str,
    function_names: List[str],
    join_lines: bool = True,
    return_line_numbers: bool = False,
    include_signature: bool = True,
    as_list: bool = False,
    is_class: bool = False,
) -> Union[
    Dict[str, Union[str, List[str]]],
    Tuple[Dict[str, Union[str, List[str]]], Dict[str, Tuple[int, int]]],
]:
    """
    Extract the source code of a function from a file.

    :param file_path: The path to the file.
    :param function_names: The names of the functions to extract.
    :param join_lines: Whether to join the lines of the function.
    :param return_line_numbers: Whether to return the line numbers of the function.
    :param include_signature: Whether to include the function signature in the source code.
    :param as_list: Whether to return a list of function sources instead of dict (useful when there is multiple
     functions with same name).
    :param is_class: Whether to also look for class definitions
    :return: A dictionary mapping function names to their source code as a string if join_lines is True,
     otherwise as a list of strings.
    """
    with open(file_path, "r") as f:
        source = f.read()

    return extract_function_or_class_from_source(
        source,
        function_names,
        join_lines=join_lines,
        return_line_numbers=return_line_numbers,
        include_signature=include_signature,
        as_list=as_list,
        is_class=is_class,
    )


def extract_function_or_class_from_source(
    source: str,
    function_names: List[str],
    join_lines: bool = True,
    return_line_numbers: bool = False,
    include_signature: bool = True,
    as_list: bool = False,
    is_class: bool = False,
) -> Union[
    Dict[str, Union[str, List[str]]],
    Tuple[Dict[str, Union[str, List[str]]], Dict[str, Tuple[int, int]]],
]:
    """
    Extract the source code of a function from a file.

    :param source: The string containing the source code.
    :param function_names: The names of the functions to extract.
    :param join_lines: Whether to join the lines of the function.
    :param return_line_numbers: Whether to return the line numbers of the function.
    :param include_signature: Whether to include the function signature in the source code.
    :param as_list: Whether to return a list of function sources instead of dict (useful when there is multiple
     functions with same name).
    :param is_class: Whether to also look for class definitions
    :return: A dictionary mapping function names to their source code as a string if join_lines is True,
     otherwise as a list of strings.
    """
    # Parse the source code into an AST
    tree = ast.parse(source)
    function_names = make_list(function_names)
    functions_source: Dict[str, Union[str, List[str]]] = {}
    functions_source_list: List[Union[str, List[str]]] = []
    line_numbers: Dict[str, Tuple[int, int]] = {}
    line_numbers_list: List[Tuple[int, int]] = []
    if is_class:
        look_for_type = ast.ClassDef
    else:
        look_for_type = ast.FunctionDef

    for node in tree.body:
        if isinstance(node, look_for_type) and (
            node.name in function_names or len(function_names) == 0
        ):
            # Get the line numbers of the function
            lines = source.splitlines()
            func_lines = lines[node.lineno - 1 : node.end_lineno]
            if not include_signature:
                func_lines = func_lines[1:]
            if as_list:
                line_numbers_list.append((node.lineno, node.end_lineno))
            else:
                line_numbers[node.name] = (node.lineno, node.end_lineno)
            parsed_function = (
                dedent("\n".join(func_lines)) if join_lines else func_lines
            )
            if as_list:
                functions_source_list.append(parsed_function)
            else:
                functions_source[node.name] = parsed_function
            if len(function_names) > 0:
                if len(functions_source) >= len(function_names) or len(
                    functions_source_list
                ) >= len(function_names):
                    break
    if len(functions_source) < len(function_names) and len(functions_source_list) < len(
        function_names
    ):
        logger.warning(
            f"Could not find all functions: {function_names} not found, "
            f"functions not found: {set(function_names) - set(functions_source.keys())}"
        )
    if return_line_numbers:
        return functions_source if not as_list else functions_source_list, (
            line_numbers if not as_list else line_numbers_list
        )
    return functions_source if not as_list else functions_source_list


def encapsulate_user_input(
    user_input: str, func_signature: str, func_doc: Optional[str] = None
) -> str:
    """
    Encapsulate the user input string with a function definition.

    :param user_input: The user input string.
    :param func_signature: The function signature to use for encapsulation.
    :param func_doc: The function docstring to use for encapsulation.
    :return: The encapsulated user input string.
    """
    func_name = func_signature.split("(")[0].strip()
    if func_name not in user_input:
        new_user_input = func_signature + "\n    "
        if func_doc is not None:
            new_user_input += f'"""{func_doc}"""' + "\n    "
        if not any(v in user_input for v in ["return ", "yield"]):
            if "\n" not in user_input:
                new_user_input += f"return {user_input}"
            else:
                raise ValueError(
                    "User input must contain a return statement or be a single line."
                )
        else:
            for cl in user_input.split("\n"):
                sub_code_lines = cl.split("\n")
                new_user_input += "\n    ".join(sub_code_lines) + "\n    "
    else:
        new_user_input = user_input
    return new_user_input


def build_user_input_from_conclusion(conclusion: Any) -> str:
    """
    Build a user input string from the conclusion.

    :param conclusion: The conclusion to use for the callable expression.
    :return: The user input string.
    """

    # set user_input to the string representation of the conclusion
    if isinstance(conclusion, Callable):
        user_input = inspect.getsource(conclusion)
        ast_code = ast.parse(dedent(user_input))
        function_body = ast.unparse(ast_code.body[0].body)
        user_input = function_body
    elif isinstance(conclusion, set):
        user_input = (
            "{" + f"{', '.join([conclusion_to_str(t) for t in conclusion])}" + "}"
        )
    elif isinstance(conclusion, list):
        user_input = (
            "[" + f"{', '.join([conclusion_to_str(t) for t in conclusion])}" + "]"
        )
    elif isinstance(conclusion, tuple):
        user_input = (
            "(" + f"{', '.join([conclusion_to_str(t) for t in conclusion])}" + ")"
        )
    else:
        user_input = conclusion_to_str(conclusion)

    return user_input


def conclusion_to_str(conclusion_: Any) -> str:
    if isinstance(conclusion_, Enum):
        return type(conclusion_).__name__ + "." + conclusion_.name
    elif isinstance(conclusion_, type):
        return conclusion_.__name__
    else:
        return str(conclusion_)


def update_case_in_case_query(case_query: CaseQuery, conclusions: Dict[str, Any]):
    """
    Update the case with the conclusions.

    :param case_query: The case query that contains the case to update.
    :param conclusions: The conclusions to update the case with.
    """
    if not conclusions:
        return
    if len(conclusions) == 0:
        return
    if not isinstance(conclusions, dict):
        conclusions = {case_query.attribute_name: conclusions}
    if isinstance(case_query.original_case, SQLTable) or is_dataclass(
        case_query.original_case
    ):
        for conclusion_name, conclusion in conclusions.items():
            attribute = getattr(case_query.case, conclusion_name)
            if conclusion_name == case_query.attribute_name:
                attribute_type = case_query.attribute_type
            else:
                attribute_type = (
                    get_case_attribute_type(
                        case_query.original_case, conclusion_name, attribute
                    ),
                )
            if isinstance(attribute, set):
                for c in conclusion:
                    attribute.update(make_set(c))
            elif isinstance(attribute, list):
                for c in conclusion:
                    if c not in attribute:
                        attribute.append(c)
            elif any(at in {List, list} for at in attribute_type):
                attribute = [] if attribute is None else attribute
                attribute.extend(conclusion)
            elif any(at in {Set, set} for at in attribute_type):
                attribute = set() if attribute is None else attribute
                for c in conclusion:
                    attribute.update(make_set(c))
            elif (
                is_iterable(conclusion)
                and len(conclusion) == 1
                and any(at is type(list(conclusion)[0]) for at in attribute_type)
            ):
                setattr(case_query.case, conclusion_name, list(conclusion)[0])
            elif not is_iterable(conclusion) and any(
                at is type(conclusion) for at in attribute_type
            ):
                setattr(case_query.case, conclusion_name, conclusion)
            else:
                raise ValueError(
                    f"Unknown type or type mismatch for attribute {conclusion_name} with type "
                    f"{case_query.attribute_type} with conclusion "
                    f"{conclusion} of type {type(conclusion)}"
                )
    else:
        case_query.case.update(conclusions)


def is_value_conflicting(conclusion: Any, target: Any) -> bool:
    """
    :param conclusion: The conclusion to check.
    :param target: The target to compare the conclusion with.
    :return: Whether the conclusion is conflicting with the target by have different values for same type categories.
    """
    return have_common_types(conclusion, target) and not make_set(conclusion).issubset(
        make_set(target)
    )


def have_common_types(conclusion: Any, target: Any) -> bool:
    """
    :param conclusion: The conclusion to check.
    :param target: The target to compare the conclusion with.
    :return: Whether the conclusion shares some types with the target.
    """
    target_types = {type(t) for t in make_set(target)}
    conclusion_types = {type(c) for c in make_set(conclusion)}
    common_types = conclusion_types.intersection(target_types)
    return len(common_types) > 0


def calculate_precision_and_recall(
    pred_cat: Dict[str, Any], target: Dict[str, Any]
) -> Tuple[List[bool], List[bool]]:
    """
    :param pred_cat: The predicted category.
    :param target: The target category.
    :return: The precision and recall of the classifier.
    """
    recall = []
    precision = []
    for pred_key, pred_value in pred_cat.items():
        if pred_key not in target:
            continue
        precision.extend(
            [v in make_set(target[pred_key]) for v in make_set(pred_value)]
        )
    for target_key, target_value in target.items():
        if target_key not in pred_cat:
            recall.append(False)
            continue
        recall.extend(
            [v in make_set(pred_cat[target_key]) for v in make_set(target_value)]
        )
    return precision, recall


def ask_llm(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "codellama:7b-instruct",  # or "phi"
                "prompt": prompt,
                "stream": False,
            },
        )
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        return f"❌ Local LLM error: {e}"


def get_case_attribute_type(
    original_case: Any, attribute_name: str, known_value: Optional[Any] = None
) -> Type:
    """
    :param original_case: The case to get the attribute from.
    :param attribute_name: The name of the attribute.
    :param known_value: A known value of the attribute.
    :return: The type of the attribute.
    """
    if known_value is not None:
        return type(known_value)
    elif hasattr(original_case, attribute_name):
        hint, origin, args = get_hint_for_attribute(attribute_name, original_case)
        if origin is not None:
            origin = typing_to_python_type(origin)
        if origin == Union:
            if len(args) == 2:
                if args[1] is type(None):
                    return typing_to_python_type(args[0])
                elif args[0] is type(None):
                    return typing_to_python_type(args[1])
            elif len(args) == 1:
                return typing_to_python_type(args[0])
            else:
                raise ValueError(
                    f"Union with more than 2 types is not supported: {args}"
                )
        elif origin is not None:
            return origin
        if hint is not None:
            return typing_to_python_type(hint)


def conclusion_to_json(conclusion):
    if is_iterable(conclusion):
        conclusions = {"_type": get_full_class_name(type(conclusion)), "value": []}
        for c in conclusion:
            conclusions["value"].append(conclusion_to_json(c))
    elif hasattr(conclusion, "to_json"):
        conclusions = conclusion.to_json()
    else:
        conclusions = {
            "_type": get_full_class_name(type(conclusion)),
            "value": conclusion,
        }
    return conclusions


def contains_return_statement(source: str) -> bool:
    """
    :param source: The source code to check.
    :return: True if the source code contains a return statement, False otherwise.
    """
    try:
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Return):
                return True
        return False
    except SyntaxError:
        return False


def get_names_used(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def extract_dependencies(code_lines):
    full_code = "\n".join(code_lines)
    tree = ast.parse(full_code)
    final_stmt = tree.body[-1]

    if not isinstance(final_stmt, ast.Return):
        raise ValueError("Last line is not a return statement")
    if final_stmt.value is None:
        raise ValueError("Return statement has no value")
    needed = get_names_used(final_stmt.value)
    required_lines = []
    line_map = {id(node): i for i, node in enumerate(tree.body)}

    def handle_stmt(stmt, needed):
        keep = False
        if isinstance(stmt, ast.Assign):
            targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            if any(t in needed for t in targets):
                needed.update(get_names_used(stmt.value))
                keep = True
        elif isinstance(stmt, ast.AugAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id in needed:
                needed.update(get_names_used(stmt.value))
                keep = True
        elif isinstance(stmt, ast.FunctionDef):
            if stmt.name in needed:
                for n in ast.walk(stmt):
                    if isinstance(n, ast.Name):
                        needed.add(n.id)
                keep = True
        elif isinstance(stmt, (ast.For, ast.While, ast.If)):
            # Check if any of the body statements interact with needed variables
            for substmt in stmt.body + getattr(stmt, "orelse", []):
                if handle_stmt(substmt, needed):
                    keep = True
            # Also check the condition (test or iter)
            if isinstance(stmt, ast.For):
                if isinstance(stmt.target, ast.Name) and stmt.target.id in needed:
                    keep = True
                needed.update(get_names_used(stmt.iter))
            elif isinstance(stmt, ast.If) or isinstance(stmt, ast.While):
                needed.update(get_names_used(stmt.test))

        return keep

    for stmt in reversed(tree.body[:-1]):
        if handle_stmt(stmt, needed):
            # check if the statement is a function, if so then all its lines not just the line in line_map are needed.
            if isinstance(stmt, ast.FunctionDef):
                start_code_line = line_map[id(stmt)]
                end_code_line = start_code_line + stmt.end_lineno
                required_lines.extend(code_lines[start_code_line:end_code_line])
            else:
                required_lines.insert(0, code_lines[line_map[id(stmt)]])

    required_lines.append(code_lines[-1])  # Always include return
    return required_lines


def serialize_dataclass(obj: Any, seen=None) -> Any:
    """
    Recursively serialize a dataclass to a dictionary. If the dataclass contains any nested dataclasses, they will be
    serialized as well. If the object is not a dataclass, it will be returned as is.

    :param obj: The dataclass to serialize.
    :return: The serialized dataclass as a dictionary or the object itself if it is not a dataclass.
    """
    if seen is None:
        seen = {}

    obj_id = id(obj)
    if obj_id in seen:
        return {"$ref": seen[obj_id]}

    if is_dataclass(obj):
        uid = str(uuid.uuid4())
        seen[obj_id] = uid
        result = {
            "$id": uid,
            "__dataclass__": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
            "fields": {},
        }
        for f in fields(obj):
            value = getattr(obj, f.name)
            result["fields"][f.name] = serialize_dataclass(value, seen)
        return result
    else:
        return SubclassJSONSerializer.to_json_static(obj, seen)


def deserialize_dataclass(data: Any, refs: Optional[Dict[str, Any]] = None) -> Any:
    refs = {} if refs is None else refs
    preloaded = preload_serialized_objects(data, refs)
    return resolve_refs(preloaded, refs)


def preload_serialized_objects(data: Any, refs: Dict[str, Any] = None) -> Any:
    """
    Recursively deserialize a dataclass from a dictionary, if the dictionary contains a key "__dataclass__" (Most likely
    created by the serialize_dataclass function), it will be treated as a dataclass and deserialized accordingly,
    otherwise it will be returned as is.

    :param data: The dictionary to deserialize.
    :return: The deserialized dataclass.
    """
    if refs is None:
        refs = {}

    if isinstance(data, dict):

        if "$ref" in data:
            ref_id = data["$ref"]
            if ref_id not in refs:
                return {"$ref": data["$ref"]}
            return refs[ref_id]

        elif "$id" in data and "__dataclass__" in data and "fields" in data:
            cls_path = data["__dataclass__"]
            module_name, class_name = cls_path.rsplit(".", 1)
            cls = getattr(importlib.import_module(module_name), class_name)

            dummy_instance = cls.__new__(cls)  # Don't call __init__ yet
            refs[data["$id"]] = dummy_instance

            for f in fields(cls):
                raw_value = data["fields"].get(f.name)
                value = preload_serialized_objects(raw_value, refs)
                setattr(dummy_instance, f.name, value)

            return dummy_instance

        else:
            return {k: preload_serialized_objects(v, refs) for k, v in data.items()}

    elif isinstance(data, list):
        return [preload_serialized_objects(item, refs) for item in data]
    elif isinstance(data, dict):
        return {k: preload_serialized_objects(v, refs) for k, v in data.items()}

    return data  # Primitive


def resolve_refs(obj, refs, seen=None):
    if seen is None:
        seen = {}

    obj_id = id(obj)
    if obj_id in seen:
        return seen[obj_id]

    # Resolve if dict with $ref
    if isinstance(obj, dict) and "$ref" in obj:
        ref_id = obj["$ref"]
        if ref_id not in refs:
            raise KeyError(f"$ref to unknown ID: {ref_id}")
        return refs[ref_id]

    elif is_dataclass(obj):
        seen[obj_id] = obj  # Mark before diving deeper
        for f in fields(obj):
            val = getattr(obj, f.name)
            resolved = resolve_refs(val, refs, seen)
            setattr(obj, f.name, resolved)
        return obj

    elif isinstance(obj, list):
        resolved_list = []
        seen[obj_id] = resolved_list
        for item in obj:
            resolved_list.append(resolve_refs(item, refs, seen))
        return resolved_list

    elif isinstance(obj, dict):
        resolved_dict = {}
        seen[obj_id] = resolved_dict
        for k, v in obj.items():
            resolved_dict[k] = resolve_refs(v, refs, seen)
        return resolved_dict

    return obj  # Primitive


def typing_to_python_type(typing_hint: Type) -> Type:
    """
    Convert a typing hint to a python type.

    :param typing_hint: The typing hint to convert.
    :return: The python type.
    """
    if typing_hint in [list, List]:
        return list
    elif typing_hint in [tuple, Tuple]:
        return tuple
    elif typing_hint in [set, Set]:
        return set
    elif typing_hint in [dict, Dict]:
        return dict
    elif typing_hint in [type, Type]:
        return type
    else:
        return typing_hint


def capture_variable_assignment(code: str, variable_name: str) -> Optional[str]:
    """
    Capture the assignment of a variable in the given code.

    :param code: The code to analyze.
    :param variable_name: The name of the variable to capture.
    :return: The assignment statement or None if not found.
    """
    tree = ast.parse(code)
    assignment = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == variable_name:
                    # now extract the right side of the assignment
                    assignment = ast.get_source_segment(code, node.value)
                    break
        if assignment is not None:
            break
    return assignment


def get_func_rdr_model_path(
    func: Callable, model_dir: str, include_file_name: bool = False
) -> str:
    """
    :param func: The function to get the model path for.
    :param model_dir: The directory to save the model to.
    :param include_file_name: Whether to include the file name in the model name.
    :return: The path to the model file.
    """
    return os.path.join(
        model_dir,
        get_func_rdr_model_name(func, include_file_name=include_file_name),
        f"{get_func_rdr_model_name(func)}_rdr.py",
    )


def get_func_rdr_model_name(func: Callable, include_file_name: bool = False) -> str:
    """
    :param func: The function to get the model name for.
    :param include_file_name: Whether to include the file name in the model name.
    :return: The name of the model.
    """
    func_name = get_method_name(func)
    func_class_name = get_method_class_name_if_exists(func)
    if include_file_name:
        func_file_name = get_method_file_name(func).split(os.sep)[-1].split(".")[0]
        model_name = func_file_name + "_"
    else:
        model_name = ""
    model_name += f"{func_class_name}_" if func_class_name else ""
    model_name += f"{func_name}"
    return str_to_snake_case(model_name)


def stringify_hint(tp):
    """Recursively convert a type hint to a string."""
    if isinstance(tp, str):
        return tp

    # Handle ForwardRef (string annotations not yet evaluated)
    if isinstance(tp, ForwardRef):
        return tp.__forward_arg__

    # Handle typing generics like List[int], Dict[str, List[int]], etc.
    origin = get_origin(tp)
    args = get_args(tp)

    if origin is not None:
        origin_str = getattr(origin, "__name__", str(origin)).capitalize()
        args_str = ", ".join(stringify_hint(arg) for arg in args)
        return f"{origin_str}[{args_str}]"

    # Handle built-in types like int, str, etc.
    if isinstance(tp, type):
        if tp.__module__ == "builtins":
            return tp.__name__
        return f"{tp.__qualname__}"

    return str(tp)


def is_builtin_type(tp):
    return isinstance(tp, type) and tp.__module__ == "builtins"


def is_typing_type(tp):
    return tp.__module__ == "typing"


origin_type_to_hint = {
    list: List,
    set: Set,
    dict: Dict,
    tuple: Tuple,
}


def get_file_that_ends_with(directory_path: str, suffix: str) -> Optional[str]:
    """
    Get the file that ends with the given suffix in the model directory.

    :param directory_path: The path to the directory where the file is located.
    :param suffix: The suffix to search for.
    :return: The path to the file that ends with the given suffix, or None if not found.
    """
    files = [f for f in os.listdir(directory_path) if f.endswith(suffix)]
    if files:
        return files[0]
    return None


def get_function_return_type(func: Callable) -> Union[Type, None, Tuple[Type, ...]]:
    """
    Get the return type of a function.

    :param func: The function to get the return type for.
    :return: The return type of the function, or None if not specified.
    """
    sig = inspect.signature(func)
    if sig.return_annotation == inspect.Signature.empty:
        return None
    type_hint = sig.return_annotation
    return get_type_from_type_hint(type_hint)


def get_type_from_type_hint(type_hint: Type) -> Union[Type, Tuple[Type, ...]]:
    origin = get_origin(type_hint)
    args = get_args(type_hint)
    if origin not in [list, set, None, Union, Iterator]:
        raise TypeError(
            f"{origin} is not a handled return type for type hint {type_hint}"
        )
    if origin is None:
        return typing_to_python_type(type_hint)
    if args is None or len(args) == 0:
        return typing_to_python_type(type_hint)
    return args


def extract_types(tp, seen: Set = None) -> Set[type]:
    """Recursively extract all base types from a type hint."""
    if seen is None:
        seen = set()

    if tp in seen or isinstance(tp, str):
        return seen

    # seen.add(tp)

    if isinstance(tp, ForwardRef):
        # Can't resolve until evaluated
        return seen

    origin = get_origin(tp)
    args = get_args(tp)

    if origin:
        if origin in origin_type_to_hint:
            seen.add(origin_type_to_hint[origin])
        else:
            seen.add(origin)
        for arg in args:
            extract_types(arg, seen)

    elif isinstance(tp, type):
        seen.add(tp)

    return seen


def get_types_to_import_from_func_type_hints(func: Callable) -> Set[Type]:
    """
    Extract importable types from a function's annotations.

    :param func: The function to extract type hints from.
    """
    hints = get_type_hints(func)

    sig = inspect.signature(func)
    all_hints = list(hints.values())
    if sig.return_annotation != inspect.Signature.empty:
        all_hints.append(sig.return_annotation)

    for param in sig.parameters.values():
        if param.annotation != inspect.Parameter.empty:
            all_hints.append(param.annotation)

    return get_types_to_import_from_type_hints(all_hints)


def get_types_to_import_from_type_hints(hints: List[Type]) -> Set[Type]:
    """
    Extract importable types from a list of type hints.

    :param hints: A list of type hints to extract types from.
    :return: A set of types that need to be imported.
    """
    seen_types = set()
    for hint in hints:
        extract_types(hint, seen_types)

    # Filter out built-in and internal types
    to_import = set()
    for tp in seen_types:
        if isinstance(tp, ForwardRef) or isinstance(tp, str):
            continue
        if not is_builtin_type(tp):
            to_import.add(tp)

    return to_import


def get_import_path_from_path(path: str) -> Optional[str]:
    """
    Convert a file system path to a Python import path.

    :param path: The file system path to convert.
    :return: The Python import path.
    """
    package_name = os.path.abspath(path)
    packages = package_name.split(os.path.sep)
    # formated_package_name = package_name.replace('/', '.')
    parent_package_idx = 0
    # packages = formated_package_name.split('.')
    for i in range(len(packages)):
        if i == 0:
            current_path = package_name
        else:
            current_path = "/" + "/".join(packages[:-i])
        if os.path.exists(os.path.join(current_path, "__init__.py")):
            parent_package_idx -= 1
        else:
            break
    package_name = (
        ".".join(packages[parent_package_idx:]) if parent_package_idx < 0 else None
    )
    return package_name


def get_class_file_path(cls):
    """
    Get the file path of a class.
    """
    return os.path.abspath(inspect.getfile(cls))


def get_function_import_data(func: Callable) -> Tuple[str, str]:
    """
    Get the import path of a function.

    :param func: The function to get the import path for.
    :return: The import path of the function.
    """
    func_name = get_method_name(func)
    func_class_name = get_method_class_name_if_exists(func)
    func_file_path = get_method_file_name(func)
    func_file_name = func_file_path.split("/")[-1].split(".")[
        0
    ]  # Get the file name without extension
    func_import_path = get_import_path_from_path(dirname(func_file_path))
    func_import_path = (
        f"{func_import_path}.{func_file_name}" if func_import_path else func_file_name
    )
    if func_class_name and func_class_name != func_name:
        func_import_name = func_class_name
    else:
        func_import_name = func_name
    return func_import_path, func_import_name


def get_function_representation(func: Callable) -> str:
    """
    Get a string representation of a function, including its module and class if applicable.

    :param func: The function to represent.
    :return: A string representation of the function.
    """
    func_name = get_method_name(func)
    func_class_name = get_method_class_name_if_exists(func)
    if func_class_name and func_class_name != func_name:
        return f"{func_class_name}.{func_name}"
    return func_name


def get_relative_import(
    target_file_path,
    imported_module_path: Optional[str] = None,
    module: Optional[str] = None,
    package_name: Optional[str] = None,
) -> str:
    """
    Get a relative import path from the target file to the imported module.

    :param target_file_path: The file path of the target file.
    :param imported_module_path: The file path of the module being imported.
    :param module: The module name, if available.
    :param package_name: The name of the root package where the module is located.
    :return: A relative import path as a string.
    """
    # Convert to absolute paths
    if module is not None:
        imported_module_path = sys.modules[module].__file__
    if imported_module_path is None:
        raise ValueError("Either imported_module_path or module must be provided")
    target_path = Path(target_file_path).resolve()
    imported_file_name = Path(imported_module_path).name
    target_file_name = Path(target_file_path).name
    if package_name is not None:
        target_path = Path(
            get_path_starting_from_latest_encounter_of(
                str(target_path), package_name, [target_file_name]
            )
        )
    imported_path = Path(imported_module_path).resolve()
    if package_name is not None:
        imported_path = Path(
            get_path_starting_from_latest_encounter_of(
                str(imported_path), package_name, [imported_file_name]
            )
        )

    # Compute relative path from target to imported module
    rel_path = os.path.relpath(imported_path.parent, target_path.parent)

    # Convert path to Python import format
    rel_parts = [part.replace("..", ".") for part in Path(rel_path).parts]
    rel_parts = rel_parts if rel_parts else [""]
    dot_parts = [part for part in rel_parts if part == "."]
    non_dot_parts = [part for part in rel_parts if part != "."] + [imported_path.stem]

    # Join the parts
    joined_parts = "." + "".join(dot_parts) + ".".join(non_dot_parts)

    return joined_parts


def get_path_starting_from_latest_encounter_of(
    path: str, package_name: str, should_contain: List[str]
) -> str:
    """
    Get the path starting from the package name.

    :param path: The full path to the file.
    :param package_name: The name of the package to start from.
    :param should_contain: The names of the files or directorys to look for.
    :return: The path starting from the package name that contains all the names in should_contain, otherwise raise an error.
    :raise ValueError: If the path does not contain all the names in should_contain.
    """
    path_parts = path.split(os.path.sep)
    if package_name not in path_parts:
        raise ValueError(f"Could not find {package_name} in {path}")
    idx = path_parts.index(package_name)
    prev_idx = idx
    while all(sc in path_parts[idx:] for sc in should_contain):
        prev_idx = idx
        try:
            idx = path_parts.index(package_name, idx + 1)
        except ValueError:
            break
    if all(sc in path_parts[idx:] for sc in should_contain):
        path_parts = path_parts[prev_idx:]
        return os.path.join(*path_parts)
    else:
        raise ValueError(f"Could not find {should_contain} in {path}")


def get_imports_from_types(
    type_objs: Iterable[Type],
    target_file_path: Optional[str] = None,
    package_name: Optional[str] = None,
    exclueded_names: Optional[List[str]] = None,
    excluded_modules: Optional[List[str]] = None,
) -> List[str]:
    """
    Format import lines from type objects.

    :param type_objs: A list of type objects to format.
    :param target_file_path: The file path to which the imports should be relative.
    :param package_name: The name of the package to use for relative imports.
    :param exclueded_names: A list of names to exclude from the imports.
    :param excluded_modules: A list of modules to exclude from the imports.
    :return: A list of formatted import lines.
    """
    excluded_modules = [] if excluded_modules is None else excluded_modules
    exclueded_names = [] if exclueded_names is None else exclueded_names
    module_to_types = defaultdict(list)
    for tp in type_objs:
        try:
            if isinstance(tp, type) or is_typing_type(tp):
                module = tp.__module__
                name = tp.__qualname__
            elif callable(tp):
                module, name = get_function_import_data(tp)
            elif hasattr(type(tp), "__module__"):
                module = type(tp).__module__
                name = type(tp).__qualname__
            else:
                continue
            if name == "NoneType":
                module = "types"
            if (
                module is None
                or module == "builtins"
                or module.startswith("_")
                or module in sys.builtin_module_names
                or module in excluded_modules
                or "<" in module
                or name in exclueded_names
            ):
                continue
            if module == "typing":
                module = "typing_extensions"
            module_to_types[module].append(name)
        except AttributeError:
            continue

    lines = []
    stem_imports = []
    for module, names in module_to_types.items():
        filtered_names = set()
        for name in set(names):
            if "." in name:
                stem = ".".join(name.split(".")[1:])
                name_to_import = name.split(".")[0]
                filtered_names.add(name_to_import)
                stem_imports.append(f"{stem} = {name_to_import}.{stem}")
            else:
                filtered_names.add(name)
        joined = ", ".join(sorted(set(filtered_names)))
        import_path = module
        if (
            (target_file_path is not None)
            and (package_name is not None)
            and (package_name in module)
        ):
            import_path = get_relative_import(
                target_file_path, module=module, package_name=package_name
            )
        lines.append(f"from {import_path} import {joined}")
    lines.extend(stem_imports)
    return lines


def get_method_args_as_dict(method: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    Get the arguments of a method as a dictionary.

    :param method: The method to get the arguments from.
    :param args: The positional arguments.
    :param kwargs: The keyword arguments.
    :return: A dictionary of the arguments.
    """
    func_arg_names = method.__code__.co_varnames
    func_arg_names = list(
        map(
            lambda arg_name: (
                f"{arg_name}_" if arg_name in ["self", "cls"] else arg_name
            ),
            func_arg_names,
        )
    )
    func_arg_values = args + tuple(kwargs.values())
    return dict(zip(func_arg_names, func_arg_values))


def get_method_name(method: Callable) -> str:
    """
    Get the name of a method.

    :param method: The method to get the name of.
    :return: The name of the method.
    """
    return method.__name__ if hasattr(method, "__name__") else str(method)


def get_method_class_name_if_exists(method: Callable) -> Optional[str]:
    """
    Get the class name of a method if it has one.

    :param method: The method to get the class name of.
    :return: The class name of the method.
    """
    if hasattr(method, "__self__"):
        if hasattr(method.__self__, "__name__"):
            return method.__self__.__name__
        elif hasattr(method.__self__, "__class__"):
            return method.__self__.__class__.__name__
    return (
        method.__qualname__.split(".")[0]
        if hasattr(method, "__qualname__") and "." in method.__qualname__
        else None
    )


def get_method_class_if_exists(method: Callable, *args) -> Optional[Type]:
    """
    Get the class of a method if it has one.

    :param method: The method to get the class of.
    :return: The class of the method, if it exists otherwise None.
    """
    if hasattr(method, "__self__"):
        if hasattr(method.__self__, "__class__"):
            return method.__self__.__class__
    elif method.__code__.co_varnames:
        if method.__code__.co_varnames[0] == "self":
            return args[0].__class__
        elif method.__code__.co_varnames[0] == "cls":
            return args[0]
    return None


def get_method_file_name(method: Callable) -> str:
    """
    Get the file name of a method.

    :param method: The method to get the file name of.
    :return: The file name of the method.
    """
    return method.__code__.co_filename


def flatten_list(a: List):
    a_flattened = []
    for c in a:
        if is_iterable(c):
            a_flattened.extend(list(c))
        else:
            a_flattened.append(c)
    return a_flattened


def make_list(value: Any) -> List:
    """
    Make a list from a value.

    :param value: The value to make a list from.
    """
    return list(value) if is_iterable(value) else [value]


def is_iterable(obj: Any) -> bool:
    """
    Check if an object is iterable.

    :param obj: The object to check.
    """
    return hasattr(obj, "__iter__") and not isinstance(
        obj, (str, type, bytes, bytearray)
    )


def get_type_from_string(type_path: str):
    """
    Get a type from a string describing its path using the format "module_path.ClassName".

    :param type_path: The path to the type.
    """
    module_path, class_name = type_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        module_path_parts = module_path.split(".")
        idx = -1
        while True:
            try:
                module = importlib.import_module(".".join(module_path_parts[:idx]))
                break
            except ModuleNotFoundError:
                idx -= 1
                if abs(idx) > len(module_path_parts):
                    raise
    if module == builtins and class_name == "NoneType":
        return type(None)
    return getattr(module, class_name)


def get_full_class_name(cls):
    """
    Returns the full name of a class, including the module name.
    Copied from: https://github.com/tomsch420/random-events/blob/master/src/random_events/utils.py#L6C1-L21C101

    :param cls: The class.
    :return: The full name of the class
    """
    return cls.__module__ + "." + cls.__name__


def recursive_subclasses(cls):
    """
    Copied from: https://github.com/tomsch420/random-events/blob/master/src/random_events/utils.py#L6C1-L21C101

    :param cls: The class.
    :return: A list of the classes subclasses.
    """
    return cls.__subclasses__() + [
        g for s in cls.__subclasses__() for g in recursive_subclasses(s)
    ]


class SubclassJSONSerializer:
    """
    Originally from: https://github.com/tomsch420/random-events/blob/master/src/random_events/utils.py#L6C1-L21C101
    Class for automatic (de)serialization of subclasses.
    Classes that inherit from this class can be serialized and deserialized automatically by calling this classes
    'from_json' method.
    """

    data_class_refs = {}

    def to_json_file(self, filename: str):
        """
        Save the object to a json file.
        """
        data = self.to_json()
        # save the json to a file
        if not filename.endswith(".json"):
            filename += ".json"
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        return data

    @staticmethod
    def to_json_static(obj, seen=None) -> Any:
        if isinstance(obj, SubclassJSONSerializer):
            return {"_type": get_full_class_name(obj.__class__), **obj._to_json()}
        elif isinstance(obj, type):
            return {"_type": get_full_class_name(obj)}
        elif is_dataclass(obj):
            return serialize_dataclass(obj, seen)
        elif isinstance(obj, list):
            return [SubclassJSONSerializer.to_json_static(v, seen) for v in obj]
        elif isinstance(obj, dict):
            serialized_dict = {}
            for k, v in obj.items():
                if not isinstance(k, (str, int, bool, float, type(None))):
                    continue
                serialized_dict[k] = SubclassJSONSerializer.to_json_static(v, seen)
            return serialized_dict
        else:
            try:
                json.dumps(obj)  # Check if the object is JSON serializable
                return obj
            except TypeError:
                return None

    def to_json(self) -> Dict[str, Any]:
        return self.to_json_static(self)

    def _to_json(self) -> Dict[str, Any]:
        """
        Create a json dict from the object.
        """
        raise NotImplementedError()

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        """
        Create a variable from a json dict.
        This method is called from the from_json method after the correct subclass is determined and should be
        overwritten by the respective subclass.

        :param data: The json dict
        :return: The deserialized object
        """
        raise NotImplementedError()

    @classmethod
    def from_json_file(cls, filename: str) -> Any:
        """
        Create an instance of the subclass from the data in the given json file.

        :param filename: The filename of the json file.
        """
        if not filename.endswith(".json"):
            filename += ".json"
        with open(filename, "r") as f:
            rdr_json = json.load(f)
        deserialized_obj = cls.from_json(rdr_json)
        cls.data_class_refs.clear()
        return deserialized_obj

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> Self:
        """
        Create the correct instanceof the subclass from a json dict.

        :param data: The json dict
        :return: The correct instance of the subclass
        """
        if data is None:
            return None
        if isinstance(data, list):
            # if the data is a list, deserialize it
            return [cls.from_json(d) for d in data]
        elif isinstance(data, dict):
            if "__dataclass__" in data:
                # if the data is a dataclass, deserialize it
                return deserialize_dataclass(data, cls.data_class_refs)
            elif "_type" not in data:
                return {k: cls.from_json(v) for k, v in data.items()}
        elif not isinstance(data, dict):
            return data

        # check if type module is builtins
        data_type = get_type_from_string(data["_type"])
        if len(data) == 1:
            return data_type
        if data_type == NoneType:
            return None
        if data_type.__module__ == "builtins":
            if is_iterable(data["value"]) and not isinstance(data["value"], dict):
                return data_type([cls.from_json(d) for d in data["value"]])
            return data_type(data["value"])
        if get_full_class_name(cls) == data["_type"]:
            data.pop("_type")
            return cls._from_json(data)
        try:
            module = importlib.import_module(data_type.__module__)
            return getattr(module, data_type.__qualname__)._from_json(data)
        except (ModuleNotFoundError, AttributeError):
            for subclass in recursive_subclasses(SubclassJSONSerializer):
                if get_full_class_name(subclass) == data["_type"]:
                    # subclass_data = deepcopy(data)
                    subclass_data = data
                    subclass_data.pop("_type")
                    return subclass._from_json(subclass_data)

        raise ValueError("Unknown type {}".format(data["_type"]))


def _pickle_thread(thread_obj) -> Any:
    """Return a plain object with user-defined attributes but no thread behavior."""

    class DummyThread:
        pass

    dummy = DummyThread()
    # Copy only non-thread-related attributes
    for attr, value in thread_obj.__dict__.items():
        print(attr)
        if not attr.startswith("_"):  # Skip internal Thread attributes
            setattr(dummy, attr, value)
    return dummy


copyreg.pickle(threading.Thread, _pickle_thread)


def copy_case(case: Union[Case, SQLTable]) -> Union[Case, SQLTable, Any]:
    """
    Copy a case.

    :param case: The case to copy.
    :return: The copied case.
    """
    if isinstance(case, SQLTable):
        return copy_orm_instance_with_relationships(case)
    else:
        # copy the case recursively for 1 level
        case_copy = copy(case)
        if isinstance(case, (dict, UserDict)):
            return case_copy
        for attr in dir(case):
            if attr.startswith("_") or callable(getattr(case, attr)):
                continue
            attr_value = getattr(case, attr)
            if is_iterable(attr_value):
                try:
                    setattr(case_copy, attr, copy(attr_value))
                except AttributeError as e:
                    # if the attribute is not settable, just skip it
                    pass
        return case_copy


def copy_orm_instance(instance: SQLTable) -> SQLTable:
    """
    Copy an ORM instance by expunging it from the session then deep copying it and adding it back to the session. This
    is useful when you want to copy an instance and make changes to it without affecting the original instance.

    :param instance: The instance to copy.
    :return: The copied instance.
    """
    try:
        session: Session = sql_inspect(instance).session
    except NoInspectionAvailable:
        session = None
    if session is not None:
        session.expunge(instance)
        new_instance = deepcopy(instance)
        session.add(instance)
    else:
        try:
            new_instance = deepcopy(instance)
        except Exception as e:
            logger.debug(e)
            new_instance = instance
    return new_instance


def copy_orm_instance_with_relationships(instance: SQLTable) -> SQLTable:
    """
    Copy an ORM instance with its relationships (i.e. its foreign keys).

    :param instance: The instance to copy.
    :return: The copied instance.
    """
    instance_cp = copy_orm_instance(instance)
    for rel in class_mapper(instance.__class__).relationships:
        related_obj = getattr(instance, rel.key)
        related_obj_cp = copy_orm_instance(related_obj)
        if related_obj is not None:
            try:
                setattr(instance_cp, rel.key, related_obj_cp)
            except Exception as e:
                logger.debug(e)
    return instance_cp


def get_value_type_from_type_hint(attr_name: str, obj: Any) -> Type:
    """
    Get the value type from the type hint of an object attribute.

    :param attr_name: The name of the attribute.
    :param obj: The object to get the attributes from.
    """
    # check first if obj is a function object
    if hasattr(obj, "__code__"):
        func_type_hints = get_type_hints(obj)
        if attr_name in func_type_hints:
            hint = func_type_hints[attr_name]
            origin = get_origin(hint)
            args = get_args(hint)
        else:
            raise ValueError(f"Unknown type hint: {attr_name}")
    else:
        hint, origin, args = get_hint_for_attribute(attr_name, obj)
    if not origin and not hint:
        if hasattr(obj, attr_name):
            attr_value = getattr(obj, attr_name)
            if attr_value is not None:
                return type(attr_value)
        raise ValueError(
            f"Couldn't get type for Attribute {attr_name}, please provide a type hint"
        )
    if origin in [list, set, tuple, type, dict]:
        attr_value_type = args[0]
    elif hint:
        attr_value_type = hint
    else:
        raise ValueError(f"Attribute {attr_name} has unsupported type {hint}.")
    return attr_value_type


def get_hint_for_attribute(
    attr_name: str, obj: Any
) -> Tuple[Optional[Type], Optional[Type], Tuple[Type]]:
    """
    Get the type hint for an attribute of an object.

    :param attr_name: The name of the attribute.
    :param obj: The object to get the attribute from.
    :return: The type hint of the attribute.
    """
    if attr_name is None or not hasattr(obj.__class__, attr_name):
        return None, None, ()
    class_attr = getattr(obj.__class__, attr_name)
    if isinstance(class_attr, property):
        if not class_attr.fget:
            raise ValueError(f"Attribute {attr_name} has no getter.")
        hint = get_type_hints(class_attr.fget)["return"]
    else:
        try:
            hint = get_type_hints(obj.__class__)[attr_name]
        except KeyError:
            hint = type(class_attr)
    origin, args = get_origin_and_args_from_type_hint(hint)
    return origin, origin, args


def get_origin_and_args_from_type_hint(
    type_hint: Type,
) -> Tuple[Optional[Type], Tuple[Type]]:
    """
    Get the origin and arguments from a type hint.W

    :param type_hint: The type hint to get the origin and arguments from.
    :return: The origin and arguments of the type hint.
    """
    origin = get_origin(type_hint)
    args = get_args(type_hint)
    if origin is Mapped:
        return get_origin(args[0]), get_args(args[0])
    else:
        return origin, args


def table_rows_as_str(row_dicts: List[Dict[str, Any]], columns_per_row: int = 20):
    """
    Print a table row.

    :param row_dicts: The rows to print.
    :param columns_per_row: The maximum number of columns per row.
    """
    max_line_sze = 100
    all_row_dicts_items = [list(row_dict.items()) for row_dict in row_dicts]
    # make items a list of n rows such that each row has a max size of 4
    all_items = [
        all_items[i : i + columns_per_row]
        for all_items in all_row_dicts_items
        for i in range(0, len(all_items), columns_per_row)
    ]
    keys = [list(map(lambda i: i[0], row)) for row in all_items]
    values = [list(map(lambda i: i[1], row)) for row in all_items]
    zipped_keys = list(zip(*keys))
    zipped_values = list(zip(*values))
    keys_values = [
        list(zip(zipped_keys[i], zipped_values[i])) for i in range(len(zipped_keys))
    ]
    keys_values = [list(r[0]) + list(r[1]) if len(r) > 1 else r[0] for r in keys_values]
    all_table_rows = []
    row_values = [
        list(map(lambda v: str(v) if v is not None else "", row)) for row in keys_values
    ]
    row_values = [
        list(
            map(lambda v: v[:max_line_sze] + "..." if len(v) > max_line_sze else v, row)
        )
        for row in row_values
    ]
    row_values = [
        list(map(lambda v: v.lower() if v in ["True", "False"] else v, row))
        for row in row_values
    ]
    # Step 1: Get terminal size
    terminal_width = shutil.get_terminal_size((80, 20)).columns
    # Step 2: Dynamically calculate max width per column (simple approximation)
    max_col_width = terminal_width // len(row_values[0])
    table = tabulate(
        row_values, tablefmt="simple_grid", maxcolwidths=max_col_width
    )  # [max_line_sze] * 2)
    all_table_rows.append(table)
    return "\n".join(all_table_rows)


def row_to_dict(obj):
    return {
        col.name: getattr(obj, col.name)
        for col in obj.__table__.columns
        if not col.primary_key and not col.foreign_keys
    }


def dataclass_to_dict(obj):
    """
    Convert a dataclass to a dictionary.

    :param obj: The dataclass to convert.
    :return: The dictionary representation of the dataclass.
    """
    if is_dataclass(obj):
        return {
            f.name: getattr(obj, f.name)
            for f in fields(obj)
            if not f.name.startswith("_")
        }
    else:
        raise ValueError(f"Object {obj} is not a dataclass.")


def get_attribute_name(
    obj: Any,
    attribute: Optional[Any] = None,
    attribute_type: Optional[Type] = None,
    possible_value: Optional[Any] = None,
) -> Optional[str]:
    """
    Get the name of an attribute from an object. The attribute can be given as a value, a type or a target value.
    And this method will try to find the attribute name using the given information.

    :param obj: The object to get the attribute name from.
    :param attribute: The attribute to get the name of.
    :param attribute_type: The type of the attribute to get the name of.
    :param possible_value: A possible value of the attribute to get the name of.
    :return: The name of the attribute.
    """
    attribute_name: Optional[str] = None
    if attribute_name is None and attribute is not None:
        attribute_name = get_attribute_name_from_value(obj, attribute)
    if attribute_name is None and attribute_type is not None:
        attribute_name = get_attribute_by_type(obj, attribute_type)[0]
    if attribute_name is None and possible_value is not None:
        attribute_name = get_attribute_by_type(obj, type(possible_value))[0]
    return attribute_name


def get_attribute_by_type(
    obj: Any, prop_type: Type
) -> Tuple[Optional[str], Optional[Any]]:
    """
    Get a property from an object by type.

    :param obj: The object to get the property from.
    :param prop_type: The type of the property.
    """
    for name in dir(obj):
        if name.startswith("_") or callable(getattr(obj, name)):
            continue
        if isinstance(getattr(obj, name), (MetaData, registry)):
            continue
        prop_value = getattr(obj, name)
        if isinstance(prop_value, prop_type):
            return name, prop_value
        if hasattr(prop_value, "__iter__") and not isinstance(prop_value, str):
            if len(prop_value) > 0 and any(
                isinstance(v, prop_type) for v in prop_value
            ):
                return name, prop_value
            else:
                # get args of type hint
                hint, origin, args = get_hint_for_attribute(name, obj)
                if origin in [list, set, tuple, dict, List, Set, Tuple, Dict]:
                    if prop_type is args[0]:
                        return name, prop_value
        else:
            # get the type hint of the attribute
            hint, origin, args = get_hint_for_attribute(name, obj)
            if hint is prop_type:
                return name, prop_value
            elif origin in [list, set, tuple, dict, List, Set, Tuple, Dict]:
                if prop_type is args[0]:
                    return name, prop_value
    return None, None


def get_attribute_name_from_value(obj: Any, attribute_value: Any) -> Optional[str]:
    """
    Get the name of an attribute from an object.

    :param obj: The object to get the attribute name from.
    :param attribute_value: The attribute value to get the name of.
    """
    for name in dir(obj):
        if name.startswith("_") or callable(getattr(obj, name)):
            continue
        prop_value = getattr(obj, name)
        if prop_value is attribute_value:
            return name


def get_attribute_values_transitively(obj: Any, attribute: Any) -> Any:
    """
    Get an attribute from a python object, if it is iterable, get the attribute values from all elements and unpack them
    into a list.

    :param obj: The object to get the sub attribute from.
    :param attribute: The  attribute to get.
    """
    if hasattr(obj, "__iter__") and not isinstance(obj, str):
        if isinstance(obj, (dict, UserDict)):
            all_values = [
                get_attribute_values_transitively(v, attribute)
                for v in obj.values()
                if not isinstance(v, (str, type)) and hasattr(v, attribute)
            ]
        else:
            all_values = [
                get_attribute_values_transitively(a, attribute)
                for a in obj
                if not isinstance(a, (str, type)) and hasattr(a, attribute)
            ]
        if can_be_a_set(all_values):
            return set().union(*all_values)
        else:
            return set(all_values)
    return getattr(obj, attribute)


def can_be_a_set(value: Any) -> bool:
    """
    Check if a value can be a set.

    :param value: The value to check.
    """
    if hasattr(value, "__iter__") and not isinstance(value, str):
        if len(value) > 0 and any(
            hasattr(v, "__iter__") and not isinstance(v, str) for v in value
        ):
            return False
        else:
            return True
    else:
        return False


def get_all_subclasses(cls: Type) -> Dict[str, Type]:
    """
    Get all subclasses of a class recursively.

    :param cls: The class to get the subclasses of.
    :return: A dictionary of all subclasses.
    """
    all_subclasses: Dict[str, Type] = {}
    for sub_cls in cls.__subclasses__():
        all_subclasses[sub_cls.__name__.lower()] = sub_cls
        all_subclasses.update(get_all_subclasses(sub_cls))
    return all_subclasses


def make_tuple(value: Any) -> Any:
    """
    Make a tuple from a value.
    """
    return tuple(value) if is_iterable(value) else (value,)


def make_set(value: Any) -> Set:
    """
    Make a set from a value.

    :param value: The value to make a set from.
    """
    return set(value) if is_iterable(value) else {value}


def make_value_or_raise_error(value: Any) -> Any:
    """
    Make a value or raise an error if the value is not a single value.

    :param value: The value to check.
    """
    if hasattr(value, "__iter__") and not isinstance(value, str):
        if hasattr(value, "__len__") and len(value) == 1:
            return list(value)[0]
        else:
            raise ValueError(f"Expected a single value, got {value}")
    return value


def tree_to_graph(root_node: Node) -> nx.DiGraph:
    """
    Convert anytree to a networkx graph.

    :param root_node: The root node of the tree.
    :return: A networkx graph.
    """
    graph = nx.DiGraph()
    unique_node_names = get_unique_node_names_func(root_node)

    def add_edges(node):
        if unique_node_names(node) not in graph.nodes:
            graph.add_node(unique_node_names(node))
        for child in node.children:
            if unique_node_names(child) not in graph.nodes:
                graph.add_node(unique_node_names(child))
            graph.add_edge(
                unique_node_names(node), unique_node_names(child), weight=child.weight
            )
            add_edges(child)

    add_edges(root_node)
    return graph


def get_unique_node_names_func(root_node) -> Callable[[Node], str]:
    nodes = [root_node]

    def get_all_nodes(node):
        for c in node.children:
            nodes.append(c)
            get_all_nodes(c)

    get_all_nodes(root_node)

    def nodenamefunc(node: Node):
        """
        Set the node name for the dot exporter.
        """
        similar_nodes = [n for n in nodes if n.name == node.name]
        node_idx = similar_nodes.index(node)
        return node.name if node_idx == 0 else f"{node.name}_{node_idx}"

    return nodenamefunc


def edge_attr_setter(parent, child):
    """
    Set the edge attributes for the dot exporter.
    """
    if child and hasattr(child, "weight") and child.weight is not None:
        return f'style="bold", label=" {child.weight}"'
    return ""


_RE_ESC = re.compile(r'["\\]')


class FilteredDotExporter(object):

    def __init__(
        self,
        node,
        include_nodes=None,
        graph="digraph",
        name="tree",
        options=None,
        indent=4,
        nodenamefunc=None,
        nodeattrfunc=None,
        edgeattrfunc=None,
        edgetypefunc=None,
        maxlevel=None,
        use_legend: bool = True,
    ):
        """
        Dot Language Exporter.

        Args:
            node (Node): start node.

        Keyword Args:
            graph: DOT graph type.

            name: DOT graph name.

            options: list of options added to the graph.

            indent (int): number of spaces for indent.

            nodenamefunc: Function to extract node name from `node` object.
                          The function shall accept one `node` object as
                          argument and return the name of it.

            nodeattrfunc: Function to decorate a node with attributes.
                          The function shall accept one `node` object as
                          argument and return the attributes.

            edgeattrfunc: Function to decorate a edge with attributes.
                          The function shall accept two `node` objects as
                          argument. The first the node and the second the child
                          and return the attributes.

            edgetypefunc: Function to which gives the edge type.
                          The function shall accept two `node` objects as
                          argument. The first the node and the second the child
                          and return the edge (i.e. '->').

            maxlevel (int): Limit export to this number of levels.

        >>> from anytree import Node
        >>> root = Node("root")
        >>> s0 = Node("sub0", parent=root, edge=2)
        >>> s0b = Node("sub0B", parent=s0, foo=4, edge=109)
        >>> s0a = Node("sub0A", parent=s0, edge="")
        >>> s1 = Node("sub1", parent=root, edge="")
        >>> s1a = Node("sub1A", parent=s1, edge=7)
        >>> s1b = Node("sub1B", parent=s1, edge=8)
        >>> s1c = Node("sub1C", parent=s1, edge=22)
        >>> s1ca = Node("sub1Ca", parent=s1c, edge=42)

        .. note:: If the node names are not unqiue, see :any:`UniqueDotExporter`.

        A directed graph:

        >>> from anytree.exporter import DotExporter
        >>> for line in DotExporter(root):
        ...     print(line)
        digraph tree {
            "root";
            "sub0";
            "sub0B";
            "sub0A";
            "sub1";
            "sub1A";
            "sub1B";
            "sub1C";
            "sub1Ca";
            "root" -> "sub0";
            "root" -> "sub1";
            "sub0" -> "sub0B";
            "sub0" -> "sub0A";
            "sub1" -> "sub1A";
            "sub1" -> "sub1B";
            "sub1" -> "sub1C";
            "sub1C" -> "sub1Ca";
        }

        The resulting graph:

        .. image:: ../static/dotexporter0.png

        An undirected graph:

        >>> def nodenamefunc(node):
        ...     return '%s:%s' % (node.name, node.depth)
        >>> def edgeattrfunc(node, child):
        ...     return 'label="%s:%s"' % (node.name, child.name)
        >>> def edgetypefunc(node, child):
        ...     return '--'
                >>> from anytree.exporter import DotExporter
        >>> for line in DotExporter(root, graph="graph",
        ...                             nodenamefunc=nodenamefunc,
        ...                             nodeattrfunc=lambda node: "shape=box",
        ...                             edgeattrfunc=edgeattrfunc,
        ...                             edgetypefunc=edgetypefunc):
        ...     print(line)
        graph tree {
            "root:0" [shape=box];
            "sub0:1" [shape=box];
            "sub0B:2" [shape=box];
            "sub0A:2" [shape=box];
            "sub1:1" [shape=box];
            "sub1A:2" [shape=box];
            "sub1B:2" [shape=box];
            "sub1C:2" [shape=box];
            "sub1Ca:3" [shape=box];
            "root:0" -- "sub0:1" [label="root:sub0"];
            "root:0" -- "sub1:1" [label="root:sub1"];
            "sub0:1" -- "sub0B:2" [label="sub0:sub0B"];
            "sub0:1" -- "sub0A:2" [label="sub0:sub0A"];
            "sub1:1" -- "sub1A:2" [label="sub1:sub1A"];
            "sub1:1" -- "sub1B:2" [label="sub1:sub1B"];
            "sub1:1" -- "sub1C:2" [label="sub1:sub1C"];
            "sub1C:2" -- "sub1Ca:3" [label="sub1C:sub1Ca"];
        }

        The resulting graph:

        .. image:: ../static/dotexporter1.png

        To export custom node implementations or :any:`AnyNode`, please provide a proper `nodenamefunc`:

        >>> from anytree import AnyNode
        >>> root = AnyNode(id="root")
        >>> s0 = AnyNode(id="sub0", parent=root)
        >>> s0b = AnyNode(id="s0b", parent=s0)
        >>> s0a = AnyNode(id="s0a", parent=s0)

        >>> from anytree.exporter import DotExporter
        >>> for line in DotExporter(root, nodenamefunc=lambda n: n.id):
        ...     print(line)
        digraph tree {
            "root";
            "sub0";
            "s0b";
            "s0a";
            "root" -> "sub0";
            "sub0" -> "s0b";
            "sub0" -> "s0a";
        }
        """
        self.node = node
        self.graph = graph
        self.name = name
        self.options = options
        self.indent = indent
        self.nodenamefunc = nodenamefunc
        self.nodeattrfunc = nodeattrfunc
        self.edgeattrfunc = edgeattrfunc
        self.edgetypefunc = edgetypefunc
        self.maxlevel = maxlevel
        self.include_nodes = include_nodes
        node_name_func = get_unique_node_names_func(node)
        self.include_node_names = (
            [node_name_func(n) for n in self.include_nodes] if include_nodes else None
        )
        self.use_legend: bool = use_legend

    def __iter__(self):
        # prepare
        indent = " " * self.indent
        nodenamefunc = self.nodenamefunc or self._default_nodenamefunc
        nodeattrfunc = self.nodeattrfunc or self._default_nodeattrfunc
        edgeattrfunc = self.edgeattrfunc or self._default_edgeattrfunc
        edgetypefunc = self.edgetypefunc or self._default_edgetypefunc
        return self.__iter(
            indent, nodenamefunc, nodeattrfunc, edgeattrfunc, edgetypefunc
        )

    @staticmethod
    def _default_nodenamefunc(node):
        return node.name

    @staticmethod
    def _default_nodeattrfunc(node):
        return None

    @staticmethod
    def _default_edgeattrfunc(node, child):
        return None

    @staticmethod
    def _default_edgetypefunc(node, child):
        return "->"

    def __iter(self, indent, nodenamefunc, nodeattrfunc, edgeattrfunc, edgetypefunc):
        yield "{self.graph} {self.name} {{".format(self=self)
        for option in self.__iter_options(indent):
            yield option
        for node in self.__iter_nodes(indent, nodenamefunc, nodeattrfunc):
            yield node
        for edge in self.__iter_edges(indent, nodenamefunc, edgeattrfunc, edgetypefunc):
            yield edge
        if self.use_legend:
            legend_dot_graph = """
// Color legend as a subgraph
subgraph cluster_legend {
    label = "Legend";
    style = dashed;
    color = gray;

    legend_green [label="Fired->Query Related Value", shape=box, style=filled, fillcolor=green, fontcolor=black, size=0.5];
    legend_yellow [label="Fired->Some Value", shape=box, style=filled, fillcolor=yellow, fontcolor=black, size=0.5];
    legend_orange [label="Fired->Empty Value", shape=box, style=filled, fillcolor=orange, fontcolor=black, size=0.5];
    legend_red [label="Evaluated->Not Fired", shape=box, style=filled, fillcolor=red, fontcolor=black, size=0.5];
    legend_white [label="Not Evaluated",  shape=box, style=filled, fillcolor=white, fontcolor=black, size=0.5];

    // Invisible edges to arrange legend vertically
    legend_white -> legend_red -> legend_orange -> legend_yellow -> legend_green [style=invis];
}"""
            for line in legend_dot_graph.splitlines():
                yield "%s" % (line.strip())
        yield "}"

    def __iter_options(self, indent):
        options = self.options
        if options:
            for option in options:
                yield "%s%s" % (indent, option)

    def __iter_nodes(self, indent, nodenamefunc, nodeattrfunc):
        for node in PreOrderIter(self.node, maxlevel=self.maxlevel):
            nodename = nodenamefunc(node)
            if (
                self.include_nodes is not None
                and nodename not in self.include_node_names
            ):
                continue
            nodeattr = nodeattrfunc(node)
            nodeattr = " [%s]" % nodeattr if nodeattr is not None else ""
            yield '%s"%s"%s;' % (indent, FilteredDotExporter.esc(nodename), nodeattr)

    def __iter_edges(self, indent, nodenamefunc, edgeattrfunc, edgetypefunc):
        maxlevel = self.maxlevel - 1 if self.maxlevel else None
        for node in PreOrderIter(self.node, maxlevel=maxlevel):
            nodename = nodenamefunc(node)
            if (
                self.include_nodes is not None
                and nodename not in self.include_node_names
            ):
                continue
            for child in node.children:
                childname = nodenamefunc(child)
                if (
                    self.include_nodes is not None
                    and childname not in self.include_node_names
                ):
                    continue
                edgeattr = edgeattrfunc(node, child)
                edgetype = edgetypefunc(node, child)
                edgeattr = " [%s]" % edgeattr if edgeattr is not None else ""
                yield '%s"%s" %s "%s"%s;' % (
                    indent,
                    FilteredDotExporter.esc(nodename),
                    edgetype,
                    FilteredDotExporter.esc(childname),
                    edgeattr,
                )

    def to_dotfile(self, filename):
        """
        Write graph to `filename`.

        >>> from anytree import Node
        >>> root = Node("root")
        >>> s0 = Node("sub0", parent=root)
        >>> s0b = Node("sub0B", parent=s0)
        >>> s0a = Node("sub0A", parent=s0)
        >>> s1 = Node("sub1", parent=root)
        >>> s1a = Node("sub1A", parent=s1)
        >>> s1b = Node("sub1B", parent=s1)
        >>> s1c = Node("sub1C", parent=s1)
        >>> s1ca = Node("sub1Ca", parent=s1c)

        >>> from anytree.exporter import DotExporter
        >>> DotExporter(root).to_dotfile("tree.dot")

        The generated file should be handed over to the `dot` tool from the
        http://www.graphviz.org/ package::

            $ dot tree.dot -T png -o tree.png
        """
        with codecs.open(filename, "w", "utf-8") as file:
            for line in self:
                file.write("%s\n" % line)

    def to_picture(self, filename):
        """
        Write graph to a temporary file and invoke `dot`.

        The output file type is automatically detected from the file suffix.

        *`graphviz` needs to be installed, before usage of this method.*
        """
        fileformat = os.path.splitext(filename)[1][1:]
        with NamedTemporaryFile("wb", delete=False) as dotfile:
            dotfilename = dotfile.name
            for line in self:
                dotfile.write(("%s\n" % line).encode("utf-8"))
            dotfile.flush()
            cmd = ["dot", dotfilename, "-T", fileformat, "-o", filename]
            check_call(cmd)
        try:
            os.remove(dotfilename)
        except Exception:  # pragma: no cover
            msg = "Could not remove temporary file %s" % dotfilename
            logger.warning(msg)

    def to_source(self) -> Source:
        """
        Return the source code of the graph as a Source object.
        """
        return Source("\n".join(self), filename=self.name)

    @staticmethod
    def esc(value):
        """Escape Strings."""
        return _RE_ESC.sub(lambda m: r"\%s" % m.group(0), six.text_type(value))


def render_tree(
    root: Node,
    use_dot_exporter: bool = False,
    filename: str = "scrdr",
    only_nodes: List[Node] = None,
    show_in_console: bool = False,
    color_map: Optional[Callable[[Node], str]] = None,
    view: bool = False,
    use_legend: bool = True,
) -> None:
    """
    Render the tree using the console and optionally export it to a dot file.

    :param root: The root node of the tree.
    :param use_dot_exporter: Whether to export the tree to a dot file.
    :param filename: The name of the file to export the tree to.
    :param only_nodes: A list of nodes to include in the dot export.
    :param show_in_console: Whether to print the tree to the console.
    :param color_map: A function that returns a color for certain nodes.
    :param view: Whether to view the dot file in a viewer.
    :param use_legend: Whether to show the legend or not.
    """
    if not root:
        logger.warning("No rules to render")
        return
    if show_in_console:
        for pre, _, node in RenderTree(root):
            if only_nodes is not None and node not in only_nodes:
                continue
            print(
                f"{pre}{node.weight if hasattr(node, 'weight') and node.weight else ''} {node.__str__()}"
            )
    if use_dot_exporter:
        unique_node_names = get_unique_node_names_func(root)

        de = FilteredDotExporter(
            root,
            include_nodes=only_nodes,
            nodenamefunc=unique_node_names,
            edgeattrfunc=edge_attr_setter,
            nodeattrfunc=lambda node: f"style=filled,"
            f' fillcolor={color_map(node) if color_map else getattr(node, "color", "white")}',
            use_legend=use_legend,
        )
        if view:
            de.to_source().view()
        else:
            filename = filename or "rule_tree"
            de.to_dotfile(f"{filename}{'.dot'}")
            try:
                de.to_picture(f"{filename}{'.svg'}")
            except FileNotFoundError as e:
                logger.warning(f"{e}")


def draw_tree(root: Node, fig: Figure):
    """
    Draw the tree using matplotlib and networkx.
    """
    # if matplotlib.get_backend().lower() not in ['qt5agg', 'qt4agg', 'qt6agg']:
    #     matplotlib.use("Qt6Agg")  # or "Qt6Agg", depending on availability

    if root is None:
        return
    fig.clf()
    graph = tree_to_graph(root)
    fig_sz_x = 13
    fig_sz_y = 10
    fig.set_size_inches(fig_sz_x, fig_sz_y)
    pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog="dot")
    # scale down pos
    max_pos_x = max([v[0] for v in pos.values()])
    max_pos_y = max([v[1] for v in pos.values()])
    pos = {
        k: (v[0] * fig_sz_x / max_pos_x, v[1] * fig_sz_y / max_pos_y)
        for k, v in pos.items()
    }
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        node_size=1000,
        ax=fig.gca(),
        node_shape="o",
        font_size=8,
    )
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=nx.get_edge_attributes(graph, "weight"),
        ax=fig.gca(),
        rotate=False,
        clip_on=False,
    )
    plt.pause(0.1)


def encapsulate_code_lines_into_a_function(
    code_lines: List[str],
    function_name: str,
    function_signature: str,
    func_doc: str,
    case_query: CaseQuery,
) -> str:
    """
    Encapsulate the given code lines into a function with the specified name, signature, and docstring.

    :param code_lines: The lines of code to include in the user input.
    :param function_name: The name of the function to include in the user input.
    :param function_signature: The function signature to include in the user input.
    :param func_doc: The function docstring to include in the user input.
    :param case_query: The case query object.
    """
    code = "\n".join(code_lines)
    code = encapsulate_user_input(code, function_signature, func_doc)
    if case_query.is_function:
        args = "**case"
    else:
        args = "case"
    if f"return {function_name}({args})" not in code:
        code = code.strip() + f"\nreturn {function_name}({args})"
    return code


def get_method_object_from_pytest_request(request) -> Callable:
    test_module = request.module.__name__  # e.g., "test_my_module"
    test_class = request.cls.__name__ if request.cls else None  # if inside a class
    test_name = request.node.name
    func = importlib.import_module(test_module)
    if test_class:
        func = getattr(getattr(func, test_class), test_name)
    else:
        func = getattr(func, test_name)
    return func
