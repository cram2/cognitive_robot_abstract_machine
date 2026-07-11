"""Generation of ``@dataclass`` ``FunctionCase`` subclasses for callables."""

from __future__ import annotations

import importlib.resources
import inspect
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from krrood.class_diagrams.utils import get_type_hints_of_object
from krrood.code_generation.generator import CodeGenerator
from krrood.code_generation.imports import (
    generate_import_statement_for_callable,
    get_imports_from_types,
    validate_annotations,
)
from krrood.code_generation.naming import to_camel_case
from krrood.code_generation.type_hints import stringify_type_hint

# %%
# FunctionCase dataclass generation


@dataclass
class FunctionCaseGenerator:
    """Emits @dataclass FunctionCase subclass source for a callable."""

    base_class_fully_qualified_name: str = (
        "krrood.entity_query_language.rdr.function_case.FunctionCase"
    )
    """Fully-qualified name of the base class the emitted dataclass inherits from."""

    code_generator: CodeGenerator = field(init=False)
    """Renderer bound to this package's templates directory."""

    def __post_init__(self):
        templates = importlib.resources.files("krrood.code_generation") / "templates"
        self.code_generator = CodeGenerator(template_directory=str(templates))

    def generate(self, function: Callable, class_name: Optional[str] = None) -> str:
        """Emit Python source for a ``@dataclass`` subclass of ``FunctionCase``.

        The emitted class has:

        - ``function: ClassVar[Callable] = <access_expression>`` — bound to the
          decorated callable via a module-level import (wrapped in try/except so
          the source can also be exec'd in isolated test namespaces).
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

        base_module, base_class_name = self.base_class_fully_qualified_name.rsplit(
            ".", 1
        )

        # Resolve string annotations (produced by
        # ``from __future__ import annotations`` in the caller's module) to
        # actual type objects before formatting.
        try:
            type_hints: Dict[str, object] = get_type_hints_of_object(function)
        except NameError:
            type_hints = {}

        # Collect custom types referenced by annotations.
        signature = inspect.signature(function)
        referenced_types: Dict[str, type] = {}
        for parameter_name, parameter in signature.parameters.items():
            if parameter_name in ("self", "cls"):
                continue
            annotation_type = type_hints.get(parameter_name, parameter.annotation)
            if isinstance(annotation_type, type) and annotation_type.__module__ not in (
                "builtins",
            ):
                referenced_types[annotation_type.__name__] = annotation_type
        return_type = type_hints.get("return", signature.return_annotation)
        if isinstance(return_type, type) and return_type.__module__ not in (
            "builtins",
        ):
            referenced_types[return_type.__name__] = return_type

        # Generate type-import lines using the centralized import generator.
        type_import_lines = "\n".join(
            get_imports_from_types(list(referenced_types.values()))
        )
        type_imports = type_import_lines + "\n" if type_import_lines else ""

        # Build field data for the Jinja2 template.
        fields = [
            {
                "name": parameter_name,
                "type_string": stringify_type_hint(
                    type_hints.get(parameter_name, parameter.annotation)
                ),
            }
            for parameter_name, parameter in signature.parameters.items()
            if parameter_name not in ("self", "cls")
        ]
        return_annotation_string = stringify_type_hint(
            type_hints.get("return", signature.return_annotation)
        )

        return self.code_generator.render(
            "function_case.py.jinja",
            base_module=base_module,
            base_class_name=base_class_name,
            class_name=class_name,
            function_name=function.__name__,
            import_line=callable_import.import_line,
            access_expression=callable_import.access_expression,
            type_imports=type_imports,
            fields=fields,
            return_type=return_annotation_string,
        )
