from __future__ import annotations

import ast
import types
from dataclasses import dataclass, field

from typing_extensions import Optional


@dataclass
class DataclassException(Exception):
    """
    A base exception class for dataclass-based exceptions.
    The way this is used is by inheriting from it and setting the `message` field in the __post_init__ method,
    then calling the super().__post_init__() method.
    """

    message: str = field(kw_only=True, default=None)

    def __post_init__(self):
        super().__init__(self.message)


@dataclass
class InputError(DataclassException):
    """
    Raised when there is an error with user input.
    """


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

    def __post_init__(self):
        self.message = (
            f"Current module is required for relative import conversion, path: {self.path},"
            f" source code: {self.source_code}."
        )
        super().__post_init__()


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

    def __post_init__(self):
        self.message = (
            f"No source data provided for import parsing, at least one of module, file_path, ast_tree should be give. "
            f"Instead got, module: {self.module},"
            f" file_path: {self.file_path}, ast_tree: {self.ast_tree}"
        )
        super().__post_init__()


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

    def __post_init__(self):
        self.message = (
            f"No source module data provided for import parsing, at least one of imported_module_path, module_name should be given. "
            f"Instead got, imported_module_path: {self.imported_module_path}, module_name: {self.module_name}"
        )
        super().__post_init__()


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

    def __post_init__(self):
        self.message = f"No default value for field '{self.field_name}' in class '{self.clazz.__name__}'"
        super().__post_init__()


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

    def __post_init__(self):
        self.message = f"Could not find {self.package_name} in {self.path}"
        super().__post_init__()


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

    def __post_init__(self):
        self.message = f"Path '{self.path}' is missing required parts: {', '.join(self.required_parts)}"
        super().__post_init__()


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

    def __post_init__(self):
        self.message = (
            f"Command '{self.command}' failed with code {self.return_code}\nSTDOUT:\n{self.stdout}\nSTDERR:"
            f"\n{self.stderr}"
        )
        super().__post_init__()


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

    def __post_init__(self):
        self.message = (
            f"Either file_path, tree, or source must be provided, got file_path: {self.file_path},"
            f" tree: {self.tree}, source_code: {self.source_code}"
        )
        super().__post_init__()
