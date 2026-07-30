from __future__ import annotations

import importlib.resources
import inspect
from dataclasses import dataclass, field
from functools import cached_property

from typing_extensions import Callable, Dict, Optional, ClassVar, Type, Any, List, Self

from krrood.class_diagrams.utils import get_type_hints_of_object
from krrood.code_generation import templates
from krrood.code_generation.enums import PythonBuiltinParameterNames
from krrood.code_generation.generator import CodeGenerator
from krrood.code_generation.imports import (
    generate_import_statement_for_callable,
    get_imports_from_types,
    validate_annotations,
)
from krrood.code_generation.naming import to_camel_case
from krrood.code_generation.type_hints import (
    stringify_type_hint,
    get_types_to_import_from_function_type_hints,
)


@dataclass
class FunctionCase:
    """
    Base class for function cases (dataclass representations of a function).
    """

    function: ClassVar[Callable]
    """
    The original (unwrapped) function object.
    """

    _output: Any = field(init=False)
    """
    The output of the function. The return type should be specified in the subclass.
    """


@dataclass
class FunctionParameter:
    """
    A single parameter of a function.
    """

    name: str
    """
    The name of the parameter.
    """
    type_hint: Any
    """
    The type hint of the parameter.
    """

    @cached_property
    def string_type_hint(self) -> str:
        """
        :return: The string representation of the type hint.
        """
        return stringify_type_hint(self.type_hint)


@dataclass
class FunctionMetaData:
    """
    Metadata about a function.
    """

    parameter_data: List[FunctionParameter] = field(default_factory=list)
    """
    The data of the parameters of the function.
    """
    return_type_hint: Any = None
    """
    The return type hint of the function.
    """

    @classmethod
    def from_function(cls, function: Callable) -> Self:
        """
        :param function: The function to extract metadata from.
        :return: A FunctionMetaData instance containing the given function's metadata.
        """
        try:
            type_hints: Dict[str, object] = get_type_hints_of_object(function)
        except NameError:
            type_hints = {}

        # Build field data for the Jinja2 template.
        signature = inspect.signature(function)
        parameters = [
            FunctionParameter(
                name=parameter_name,
                type_hint=type_hints.get(parameter_name, parameter.annotation),
            )
            for parameter_name, parameter in signature.parameters.items()
            if parameter_name not in PythonBuiltinParameterNames
        ]
        return_annotation_string = type_hints.get("return", signature.return_annotation)
        return cls(parameter_data=parameters, return_type_hint=return_annotation_string)

    @cached_property
    def string_return_type_hint(self) -> str:
        """
        :return: The string representation of the return type hint.
        """
        return stringify_type_hint(self.return_type_hint)


@dataclass
class FunctionCaseGenerator:
    """
    Generates a dataclass FunctionCase subclass source for a given function.
    """

    base_class: Type[FunctionCase] = FunctionCase
    """
    The base class to inherit from.  Must be a subclass of ``FunctionCase``.
    """

    code_generator: CodeGenerator = field(init=False)
    """
    Jinja code generator bound to this package's templates directory.
    """

    template_file_name: ClassVar[str] = "function_case.py.jinja"
    """
    The name of the template file to use for generating FunctionCase subclasses.
    """

    def __post_init__(self):
        templates_path = importlib.resources.files(templates)
        self.code_generator = CodeGenerator(template_directory=str(templates_path))

    def generate(self, function: Callable, class_name: Optional[str] = None) -> str:
        """
        Generate Python source for a ``@dataclass`` subclass of ``FunctionCase``.

        The generated class has:

        - ``function: ClassVar[Callable] = <access_expression>`` — bound to the
          decorated callable via a module-level import (wrapped in try/except so
          the source can also be executed in isolated test namespaces).
        - One field per annotated parameter (``self`` / ``cls`` excluded).
        - ``_output: <return_annotation>`` — the attribute the RDR will predict.

        :param function: The callable to generate a case type for.
        :param class_name: Override for the generated class name.  When ``None``
            the name is derived from ``function.__name__`` via :func:`to_camel_case`.
        :raises FunctionMissingAnnotationsError: If any required annotation is absent.
        :returns: A Python source string that can be written to a ``.py`` file.
        """
        validate_annotations(function)

        if class_name is None:
            class_name = to_camel_case(function.__name__)
        callable_import = generate_import_statement_for_callable(function)

        # Collect custom types referenced by annotations.
        types_to_import = get_types_to_import_from_function_type_hints(function)

        # Generate type-import lines using the centralized import generator.
        type_import_lines = get_imports_from_types(list(types_to_import))

        function_meta_data = FunctionMetaData.from_function(function)

        function_attribute_setter_line = ""
        if callable_import and callable_import.access_expression:
            function_attribute_setter_line = f"{class_name}.function = {callable_import.access_expression}"

        return self.code_generator.render(
            self.template_file_name,
            class_name=class_name,
            base_class_name=self.base_class.__name__,
            base_class_module=self.base_class.__module__,
            function_name=function.__name__,
            import_line=callable_import.import_line,
            access_expression=callable_import.access_expression,
            type_imports=type_import_lines,
            function_meta_data=function_meta_data,
            function_attribute_setter_line=function_attribute_setter_line
        )
