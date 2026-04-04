from __future__ import annotations

import ast
import logging
import os
from _ast import AST
from copy import copy
from enum import Enum
from types import GeneratorType

from typing_extensions import Type, Optional, Any, List, Union, Tuple, Dict, Set, get_origin

from krrood.ripple_down_rules.datastructures.case import create_case, Case
from krrood.ripple_down_rules.utils import SubclassJSONSerializer, get_full_class_name, get_type_from_string, conclusion_to_json, is_iterable, \
    build_user_input_from_conclusion, encapsulate_user_input, extract_function_or_class_file, \
    are_results_subclass_of_types, \
    make_list, get_imports_from_scope, get_type_from_type_hint


class VariableVisitor(ast.NodeVisitor):
    """
    A visitor to extract all variables and comparisons from a python expression represented as an AST tree.
    """
    compares: List[Tuple[Union[ast.Name, ast.Call], ast.cmpop, Union[ast.Name, ast.Call]]]
    variables: Set[str]
    all: List[ast.BoolOp]

    def __init__(self):
        self.variables = set()
        self.attributes: Dict[ast.Name, ast.Attribute] = {}
        self.types = set()
        self.callables = set()
        self.compares = list()
        self.binary_ops = list()
        self.all = list()

    def visit_Constant(self, node):
        self.all.append(node)
        self.types.add(node)
        self.generic_visit(node)

    def visit_Call(self, node):
        self.all.append(node)
        self.callables.add(node)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        self.all.append(node)
        self.attributes[node.value] = node
        self.generic_visit(node)

    def visit_BinOp(self, node):
        self.binary_ops.append(node)
        self.all.append(node)
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        self.all.append(node)
        self.generic_visit(node)

    def visit_Compare(self, node):
        self.all.append(node)
        self.compares.append([node.left, node.ops[0], node.comparators[0]])
        self.generic_visit(node)

    def visit_Name(self, node):
        if f"__{node.id}__" not in dir(__builtins__) and node not in self.attributes:
            self.variables.add(node.id)
        self.generic_visit(node)


def get_used_scope(code_str, scope):
    # Parse the code into an AST
    mode = 'exec' if code_str.startswith('def') else 'eval'
    tree = ast.parse(code_str, mode=mode)

    # Walk the AST to collect used variable names
    class NameCollector(ast.NodeVisitor):
        def __init__(self):
            self.names = set()

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):  # We care only about variables being read
                self.names.add(node.id)

    collector = NameCollector()
    collector.visit(tree)

    # Filter the scope to include only used names
    used_scope = {k: scope[k] for k in collector.names if k in scope}
    return used_scope


class CallableExpression(SubclassJSONSerializer):
    """
    A callable that is constructed from a string statement written by an expert.
    """
    encapsulating_function_name: str = "_get_value"

    def __init__(self, user_input: Optional[str] = None,
                 conclusion_type: Optional[Tuple[Type, ...]] = None,
                 expression_tree: Optional[AST] = None,
                 scope: Optional[Dict[str, Any]] = None,
                 conclusion: Optional[Any] = None,
                 mutually_exclusive: bool = True):
        """
        Create a callable expression.

        :param user_input: The input given by the expert.
        :param conclusion_type: The type of the output of the callable.
        :param expression_tree: The AST tree parsed from the user input.
        :param scope: The scope to use for the callable expression.
        :param conclusion: The conclusion to use for the callable expression.
        :param mutually_exclusive: If True, the conclusion is mutually exclusive, i.e. the callable expression can only
            return one conclusion. If False, the callable expression can return multiple conclusions.
        """
        if user_input is None and conclusion is None:
            raise ValueError("Either user_input or conclusion must be provided.")
        if user_input is None:
            user_input = build_user_input_from_conclusion(conclusion)
        self.conclusion: Optional[Any] = conclusion
        if "def " in user_input:
            self.user_defined_name = user_input.split('(')[0].replace('def ', '')
        else:
            self.user_defined_name = user_input
        if f"def {self.encapsulating_function_name}" not in user_input:
            user_input = encapsulate_user_input(user_input, self.get_encapsulating_function())
        self._user_input: str = user_input
        if conclusion_type is not None:
            if is_iterable(conclusion_type):
                conclusion_type = tuple(conclusion_type)
            else:
                conclusion_type = (conclusion_type,)
        self.conclusion_type = conclusion_type
        self.expected_types: Set[Type] = set(conclusion_type) if conclusion_type is not None else set()
        if list in self.expected_types:
            self.expected_types.remove(list)
        if set in self.expected_types:
            self.expected_types.add(set)
        self.scope: Optional[Dict[str, Any]] = scope if scope is not None else {}
        self.scope = get_used_scope(self.user_input, self.scope)
        self.expression_tree: AST = expression_tree if expression_tree else parse_string_to_expression(self.user_input)
        self.code = compile_expression_to_code(self.expression_tree)
        self.visitor = VariableVisitor()
        self.visitor.visit(self.expression_tree)
        self.mutually_exclusive: bool = mutually_exclusive

    @classmethod
    def get_encapsulating_function(cls, postfix: str = '') -> str:
        """
        Get the encapsulating function that is used to wrap the user input.
        """
        return f"def {cls.encapsulating_function_name}{postfix}(case):"

    def __call__(self, case: Any, **kwargs) -> Any:
        try:
            if self.user_input is not None:
                if not isinstance(case, Case):
                    case = create_case(case, max_recursion_idx=3)
                scope = {'case': case, **self.scope}
                output = eval(self.code, scope)
                if output is None:
                    output = scope['_get_value'](case)
                if self.conclusion_type is not None:
                    if self.mutually_exclusive and issubclass(type(output), (list, set)):
                        raise ValueError(f"Mutually exclusive types cannot be lists or sets, got {type(output)}")
                    output_types = {type(o) for o in make_list(output)}
                    if not are_results_subclass_of_types(output_types, self.expected_types):
                        raise ValueError(f"Not all result types {output_types} are subclasses of expected types"
                                         f" {self.conclusion_type}")
                return output
            elif self.conclusion is not None:
                return self.conclusion
            else:
                raise ValueError("Either user_input or conclusion must be provided.")
        except Exception as e:
            raise ValueError(f"Error during evaluation: {e}, user_input: {self.user_input}")

    def combine_with(self, other: 'CallableExpression') -> 'CallableExpression':
        """
        Combine this callable expression with another callable expression using the 'and' operator.
        """
        cond1_user_input = self.user_input.replace(self.get_encapsulating_function(), "def _cond1(case):")
        cond2_user_input = other.user_input.replace(self.get_encapsulating_function(), "def _cond2(case):")
        new_user_input = (f"{cond1_user_input}\n"
                          f"{cond2_user_input}\n"
                          f"return _cond1(case) and _cond2(case)")
        return CallableExpression(new_user_input, conclusion_type=self.conclusion_type,
         mutually_exclusive=self.mutually_exclusive)

    def update_user_input_from_file(self, file_path: str, function_name: str):
        """
        Update the user input from a file.
        """
        new_function_body = extract_function_or_class_file(file_path, [function_name])[function_name]
        if new_function_body is None:
            return
        self.user_input = self.get_encapsulating_function() + '\n' + new_function_body

    def write_to_python_file(self, file_path: str, append: bool = False):
        """
        Write the callable expression to a python file.

        :param file_path: The path to the file where the callable expression will be written.
        :param append: If True, the callable expression will be appended to the file. If False,
         the file will be overwritten.
        """
        imports = '\n'.join(get_imports_from_scope(self.scope))
        if append and os.path.exists(file_path):
            with open(file_path, 'a') as f:
                f.write('\n\n\n' + imports + '\n\n\n')
                f.write(self.user_input)
        else:
            with open(file_path, 'w') as f:
                f.write(imports + '\n\n\n')
                f.write(self.user_input)

    @property
    def user_input(self):
        """
        Get the user input.
        """
        return self._user_input

    @user_input.setter
    def user_input(self, value: str):
        """
        Set the user input.
        """
        if value is not None:
            if "def " in value:
                self.user_defined_name = value.split('(')[0].replace('def ', '')
            else:
                self.user_defined_name = value
            self._user_input = encapsulate_user_input(value, self.get_encapsulating_function())
            self.scope = get_used_scope(self.user_input, self.scope)
            self.expression_tree = parse_string_to_expression(self.user_input)
            self.code = compile_expression_to_code(self.expression_tree)
            self.visitor = VariableVisitor()
            self.visitor.visit(self.expression_tree)

    def __eq__(self, other):
        """
        Check if two callable expressions are equal.
        """
        if not isinstance(other, CallableExpression):
            return False
        return self.user_input == other.user_input and self.conclusion == other.conclusion

    def __hash__(self):
        """
        Hash the callable expression.
        """
        conclusion_hash = self.conclusion if not isinstance(self.conclusion, set) else frozenset(self.conclusion)
        return hash((self.user_input, conclusion_hash))

    def __str__(self):
        """
        Return the user string where each compare is written in a line using compare column offset start and end.
        """
        if self.user_input is None:
            return str(self.conclusion)
        binary_ops = sorted(self.visitor.binary_ops, key=lambda x: x.end_col_offset)
        binary_ops_indices = [b.end_col_offset for b in binary_ops]
        all_binary_ops = []
        prev_e = 0
        for i, e in enumerate(binary_ops_indices):
            if i == 0:
                all_binary_ops.append(self.user_input[:e])
            else:
                all_binary_ops.append(self.user_input[prev_e:e])
            prev_e = e
        return "\n".join(all_binary_ops) if len(all_binary_ops) > 0 else self.user_input

    def _to_json(self) -> Dict[str, Any]:
        return {"user_input": self.user_input,
                "conclusion_type": [get_full_class_name(t) for t in self.conclusion_type]
                if self.conclusion_type is not None else None,
                "scope": {k: get_full_class_name(v) for k, v in self.scope.items()
                          if hasattr(v, '__module__') and hasattr(v, '__name__')
                          and v.__module__ is not None and v.__name__ is not None},
                "conclusion": conclusion_to_json(self.conclusion),
                "mutually_exclusive": self.mutually_exclusive,
                }

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> CallableExpression:
        scope = {}
        for k, v in data['scope'].items():
            try:
                scope[k] = get_type_from_string(v)
            except ModuleNotFoundError:
                pass
        return cls(user_input=data["user_input"],
                   conclusion_type=tuple(get_type_from_string(t) for t in data["conclusion_type"])
                   if data["conclusion_type"] else None,
                   scope=scope,
                   conclusion=SubclassJSONSerializer.from_json(data["conclusion"]),
                   mutually_exclusive=data["mutually_exclusive"])


def compile_expression_to_code(expression_tree: AST) -> Any:
    """
    Compile an expression tree that was parsed from string into code that can be executed using 'eval(code)'

    :param expression_tree: The parsed expression tree.
    :return: The code that was compiled from the expression tree.
    """
    mode = 'exec' if isinstance(expression_tree, ast.Module) else 'eval'
    return compile(expression_tree, filename="<string>", mode=mode)


def parse_string_to_expression(expression_str: str) -> AST:
    """
    Parse a string statement into an AST expression.

    :param expression_str: The string which will be parsed.
    :return: The parsed expression.
    """
    if not expression_str.startswith(f"def {CallableExpression.encapsulating_function_name}"):
        expression_str = encapsulate_user_input(expression_str, CallableExpression.get_encapsulating_function())
    mode = 'exec' if expression_str.startswith('def') else 'eval'
    tree = ast.parse(expression_str, mode=mode)
    logging.debug(f"AST parsed successfully: {ast.dump(tree)}")
    return tree
