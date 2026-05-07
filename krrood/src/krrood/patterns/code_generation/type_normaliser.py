"""
Normaliser that converts Python type objects to source code strings.
"""

from __future__ import annotations

import dataclasses
from typing import Any, TypeVar, get_origin, get_args, Callable

from krrood.class_diagrams import ClassDiagram
from krrood.patterns.code_generation.import_name_resolver import ImportNameResolver


@dataclasses.dataclass
class TypeNormaliser:
    """
    Normalises Python type objects to consistent string names for code generation.
    """

    resolver: ImportNameResolver
    class_diagram: ClassDiagram
    class_name_getter: Callable[[type], str] | None = None

    def normalise(self, type_obj: Any) -> str:
        """Return a consistent string representation of a type for use in generated code.

        :param type_obj: The type object to normalise.
        :return: A string type name suitable for inclusion in generated source code.
        """
        if isinstance(type_obj, str):
            return self._handle_string_type(type_obj)

        if isinstance(type_obj, TypeVar):
            return self._handle_type_var(type_obj)

        origin = get_origin(type_obj)
        if origin is not None:
            return self._handle_generic_type(type_obj, origin)

        if isinstance(type_obj, type):
            return self._handle_class_type(type_obj)

        return self._handle_fallback_type(type_obj)

    def _handle_string_type(self, type_str: str) -> str:
        """Return a normalised name for a forward-reference string type.

        :param type_str: The forward-reference string to normalise.
        :return: The resolved or unchanged type name string.
        """
        if type_str not in self.resolver.name_to_module_map:
            resolved_module = self.resolver.resolve(type_str)
            if resolved_module:
                self.resolver.name_to_module_map[type_str] = resolved_module
        return type_str

    def _handle_generic_type(self, type_obj: Any, origin: Any) -> str:
        """Return a normalised name for a generic type such as ``List[str]``.

        :param type_obj: The generic type object.
        :param origin: The origin type returned by ``get_origin``.
        :return: A normalised string representation of the generic type.
        """
        alias_name = type_obj._name if hasattr(type_obj, "_name") else None
        alias_module = type_obj.__module__ if hasattr(type_obj, "__module__") else None
        if alias_name and alias_module and alias_module != "builtins":
            self.resolver.name_to_module_map.setdefault(alias_name, alias_module)
        origin_name = self.normalise(origin)
        args = get_args(type_obj)

        if args:
            arg_names = [self.normalise(arg) for arg in args]
            result = f"{origin_name}[{', '.join(arg_names)}]"
        else:
            result = origin_name

        return result.replace("typing.", "").replace("typing_extensions.", "")

    def _handle_type_var(self, type_var: TypeVar) -> str:
        """Return a normalised name for a TypeVar.

        :param type_var: The TypeVar to normalise.
        :return: The TypeVar name.
        """
        if hasattr(type_var, "__module__"):
            self.resolver.name_to_module_map[type_var.__name__] = type_var.__module__

        if type_var.__bound__ is not None:
            self.normalise(type_var.__bound__)

        return type_var.__name__

    def _handle_class_type(self, clazz: type) -> str:
        """Return a normalised name for a plain class type.

        :param clazz: The class to normalise.
        :return: The class name string.
        """
        if clazz is type(None):
            return "None"

        self.resolver.name_to_module_map[clazz.__name__] = clazz.__module__
        if self.class_name_getter is not None:
            return self.class_name_getter(clazz)
        return clazz.__name__

    def _handle_fallback_type(self, type_obj: Any) -> str:
        """Return a normalised name for an unrecognised type object.

        :param type_obj: The unrecognised type to normalise.
        :return: A string representation of the type.
        """
        if hasattr(type_obj, "__name__") and hasattr(type_obj, "__module__"):
            self.resolver.name_to_module_map[type_obj.__name__] = type_obj.__module__
        return str(type_obj)
