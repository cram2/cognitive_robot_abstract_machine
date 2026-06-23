from __future__ import annotations

import ast
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Type, Tuple

from typing_extensions import Optional


@dataclass
class DataclassException(Exception, ABC):
    """
    A base exception class for dataclass-based exceptions.
    Subclasses implement error_message() and suggest_correction(); both are evaluated at
    construction time and composed into the actual exception message (the args passed to
    Exception). A non-empty suggest_correction() is rendered as a trailing "Suggestion: ..." line.
    Subclasses that override __post_init__ must call super().__post_init__() at the end.
    """

    def __post_init__(self):
        # BaseException.__new__ bypasses the usual ABC instantiation check, so enforce it here.
        if getattr(type(self), "__abstractmethods__", None):
            raise TypeError(
                f"Can't instantiate abstract class {type(self).__name__} without an implementation of "
                f"{', '.join(sorted(type(self).__abstractmethods__))}."
            )
        message = self.error_message()
        correction = self.suggest_correction()
        if correction:
            message = f"{message}\nSuggestion: {correction}"
        super().__init__(message)

    def __str__(self) -> str:
        # Stdlib mixins like KeyError override __str__ to repr their args; always render the
        # plain message that __post_init__ baked into the args.
        return Exception.__str__(self)

    @abstractmethod
    def error_message(self) -> str:
        """
        :return: A human-readable description of what went wrong.
        """

    @abstractmethod
    def suggest_correction(self) -> str:
        """
        :return: Advice on how to fix the error, or an empty string if there is no specific advice.
        """


@dataclass
class InputError(DataclassException):
    """
    Raised when there is an error with user input.
    """


@dataclass
class MismatchingNumberOfGenericParametersAndResolvedTypes(DataclassException):
    """
    Raised when the number of generic parameters does not match the number of resolved types.
    """

    affected_class: Type
    """
    The class that has the generic parameters.
    """

    parameters: list[Type]
    """
    The generic parameters of the class.
    """

    resolved_types: Tuple[Type, ...]
    """
    The resolved types for the generic parameters.
    """

    def error_message(self) -> str:
        return (
            f"The number of generic type parameters in {self.affected_class.__name__} "
            f"({len(self.parameters)}) does not match the number of "
            f"provided arguments ({len(self.resolved_types)})."
            f"Parameters: {self.parameters}, resolved_types: {self.resolved_types}"
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class ModuleNotFoundForConvertingImportsToAbsolute(InputError):
    """
    Raised when the current module is not given and/or not found for relative import conversion.
    """

    path: Optional[str] = field(kw_only=True, default=None)
    """
    The path to the file that contains the relative import.
    """
    source_code: Optional[str] = field(kw_only=True, default=None)
    """
    The source code of the file that contains the relative import.
    """

    def error_message(self) -> str:
        return (
            f"Current module is required for relative import conversion, path: {self.path},"
            f" source code: {self.source_code}."
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class NoSourceDataToParseImportsFrom(InputError):
    """
    Raised when there is no source data given to parse imports from.
    """

    module: Optional[types.ModuleType] = None
    """
    The module to parse imports from.
    """
    file_path: Optional[str] = None
    """
    The file path to the module to parse imports from.
    """
    ast_tree: Optional[ast.AST] = None
    """
    The AST tree to parse imports from.
    """

    def error_message(self) -> str:
        return (
            f"No source data provided for import parsing, at least one of module, file_path, ast_tree should be give. "
            f"Instead got, module: {self.module},"
            f" file_path: {self.file_path}, ast_tree: {self.ast_tree}"
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class NoModuleSourceProvided(InputError):
    """
    Raised when there is no source module data given to parse imports from.
    """

    imported_module_path: Optional[str] = None
    """
    The file path to the module to parse imports from.
    """
    module_name: Optional[str] = None
    """
    The module to parse imports from.
    """

    def error_message(self) -> str:
        return (
            f"No source module data provided for import parsing, at least one of imported_module_path, module_name should be given. "
            f"Instead got, imported_module_path: {self.imported_module_path}, module_name: {self.module_name}"
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class NoDefaultValueFound(DataclassException):
    """
    Raised when no default value for a given field in a dataclass is found.
    """

    clazz: type
    """
    The class where the field is defined.
    """
    field_name: str
    """
    The name of the field for which no default value was found.
    """

    def error_message(self) -> str:
        return f"No default value for field '{self.field_name}' in class '{self.clazz.__name__}'"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class PackageNameNotFoundError(DataclassException):
    """
    Raised when a package name is not found in a given path.
    """

    package_name: str
    """
    The package name that was not found.
    """
    path: str
    """
    The path where the package name was not found.
    """

    def error_message(self) -> str:
        return f"Could not find {self.package_name} in {self.path}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class PathMissingRequiredPartsError(DataclassException):
    """
    Raised when a path does not contain all required parts.
    """

    required_parts: list[str]
    """
    The required parts that were missing from the path.
    """

    path: str
    """
    The path that was missing required parts.
    """

    def error_message(self) -> str:
        return f"Path '{self.path}' is missing required parts: {', '.join(self.required_parts)}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class SubprocessExecutionError(DataclassException):
    """
    Raised when a subprocess execution fails.
    """

    command: str
    """
    The command that was executed.
    """
    return_code: int
    """
    The return code of the subprocess.
    """
    stdout: str
    """
    The standard output of the subprocess.
    """
    stderr: str
    """
    The standard error of the subprocess.
    """

    def error_message(self) -> str:
        return (
            f"Command '{self.command}' failed with code {self.return_code}\nSTDOUT:\n{self.stdout}\nSTDERR:"
            f"\n{self.stderr}"
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class SourceDataNotProvided(InputError):
    """
    Raised when no source data is provided.
    """

    file_path: Optional[str] = None
    """
    The file path that was missing source data.
    """
    tree: Optional[ast.AST] = None
    """
    The AST tree that was missing source data.
    """
    source_code: Optional[str] = None
    """
    The source code that was missing.
    """

    def error_message(self) -> str:
        return (
            f"Either file_path, tree, or source must be provided, got file_path: {self.file_path},"
            f" tree: {self.tree}, source_code: {self.source_code}"
        )

    def suggest_correction(self) -> str:
        return ""
