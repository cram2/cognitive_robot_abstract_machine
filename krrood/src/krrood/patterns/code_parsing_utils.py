import ast
import inspect
from functools import lru_cache
from textwrap import dedent
from typing import Optional

from typing_extensions import Type

from krrood.entity_query_language.exceptions import WrongPropertyReturnStatementImplementation, \
    NoReturnStatementInProperty


@lru_cache
def get_accessed_attribute_name_in_return_statement_of_property(property_object: property, clazz: Optional[Type] = None) -> str:
    """
    :param property_object: The property to extract the accessed attribute name from.
    :param clazz: The clazz to extract the accessed attribute name from.
    :return: The accessed attribute name.
    """
    subject_property_code = inspect.getsource(property_object.fget)
    # extract the return statement, it should match `self.{attribute_name}`
    ast_tree = ast.parse(dedent(subject_property_code))
    for node in ast.walk(ast_tree):
        if isinstance(node, ast.Return):
            statement: ast.Attribute = node.value
            if not isinstance(statement, ast.Attribute):
                raise WrongPropertyReturnStatementImplementation(clazz=clazz,
                                                                 property_object=property_object,
                                                                 reason="The return statement is not an attribute access")
            if not isinstance(statement.value, ast.Name) or statement.value.id != 'self':
                raise WrongPropertyReturnStatementImplementation(clazz=clazz,
                                                                 property_object=property_object,
                                                                 reason="The return statement is not accesing the attribute from `self`")
            return statement.attr
    raise NoReturnStatementInProperty(clazz=clazz, property_object=property_object)
