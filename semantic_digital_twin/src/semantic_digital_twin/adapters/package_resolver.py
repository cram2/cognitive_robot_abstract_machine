from __future__ import annotations

import glob
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import List, Optional

from semantic_digital_twin.exceptions import (
    ParsingError,
    PackageResolutionError,
    PathResolutionError,
)


class PackageLocator(ABC):
    """
    Abstract base class for package locators.
    """

    @abstractmethod
    def resolve(self, package_name: str) -> str:
        """
        Resolves a package name to its local filesystem path.
        """


@dataclass
class AmentPackageLocator(PackageLocator):
    """
    Resolves packages using ament.
    """

    def resolve(self, package_name: str) -> str:
        try:
            from ament_index_python.packages import get_package_share_directory

            return get_package_share_directory(package_name)
        except (ImportError, LookupError) as error:
            raise PackageResolutionError(
                package_name=package_name, details=f"ament: {error}"
            )


@dataclass
class ROSPackagePathLocator(PackageLocator):
    """
    Resolves packages using ROS_PACKAGE_PATH.
    """

    def resolve(self, package_name: str) -> str:
        for root in os.environ.get("ROS_PACKAGE_PATH", "").split(":"):
            if not root:
                continue
            for candidate in [root, os.path.join(root, package_name)]:
                if os.path.isdir(candidate) and root.endswith(package_name):
                    return candidate
        raise PackageResolutionError(
            package_name=package_name, details="not found in ROS_PACKAGE_PATH"
        )


@dataclass
class PrefixPathPackageLocator(PackageLocator):
    """
    Resolves packages by searching install prefixes directly on disk, without any ROS
    tooling installed.
    """

    additional_prefixes: List[str] = field(default_factory=list)
    """
    Extra install prefixes to search before the environment and workspace prefixes.
    """

    def _environment_prefixes(self) -> List[str]:
        """
        :return: Prefixes named by ``AMENT_PREFIX_PATH`` and ``CMAKE_PREFIX_PATH``.
        """
        prefixes = []
        for variable in ("AMENT_PREFIX_PATH", "CMAKE_PREFIX_PATH"):
            prefixes += [
                entry for entry in os.environ.get(variable, "").split(":") if entry
            ]
        return prefixes

    def _workspace_prefixes(self) -> List[str]:
        """
        :return: Install directories of common workspace layouts under the home
            directory and ``/opt/ros``.
        """
        home = os.path.expanduser("~")
        return [
            *glob.glob(os.path.join(home, "*_ws", "install")),
            *glob.glob(os.path.join(home, "*", "install")),
            *glob.glob("/opt/ros/*"),
        ]

    def search_prefixes(self) -> List[str]:
        """
        :return: Every install prefix searched, in order of precedence.
        """
        return [
            *self.additional_prefixes,
            *self._environment_prefixes(),
            *self._workspace_prefixes(),
        ]

    def resolve(self, package_name: str) -> str:
        prefixes = self.search_prefixes()
        for prefix in prefixes:
            for candidate in (
                os.path.join(prefix, package_name, "share", package_name),
                os.path.join(prefix, "share", package_name),
                os.path.join(prefix, package_name),
            ):
                if os.path.isdir(candidate):
                    return candidate
        raise PackageResolutionError(
            package_name=package_name,
            details=f"not found in any of {len(prefixes)} install prefixes",
        )


@dataclass
class ROSPackageLocator(PackageLocator):
    """
    Tries multiple package locators in order.
    """

    locators: List[PackageLocator] = field(
        default_factory=lambda: [
            AmentPackageLocator(),
            ROSPackagePathLocator(),
            PrefixPathPackageLocator(),
        ]
    )

    def resolve(self, package_name: str) -> str:
        errors = []
        for locator in self.locators:
            try:
                return locator.resolve(package_name)
            except ParsingError as error:
                errors.append(str(error))
        raise PackageResolutionError(
            package_name=package_name, details="; ".join(errors)
        )


class PathResolver(ABC):
    """
    Abstract base class for path resolvers.
    """

    @abstractmethod
    def supports(self, uri: str) -> bool:
        """
        Checks if the URI is supported by this resolver.
        """

    @abstractmethod
    def resolve(self, uri: str) -> str:
        """
        Resolves a URI to an absolute local file path.
        """


@dataclass
class PackageUriResolver(PathResolver):
    """
    Resolves package:// URIs.
    """

    locator: PackageLocator = field(default_factory=ROSPackageLocator)

    def supports(self, uri: str) -> bool:
        return uri.startswith("package://")

    def resolve(self, uri: str) -> str:
        rest = uri[len("package://") :]
        if "/" not in rest:
            package_name, relative_path = rest, ""
        else:
            package_name, relative_path = rest.split("/", 1)
        base = self.locator.resolve(package_name)
        return os.path.join(base, relative_path)


@dataclass
class FileUriResolver(PathResolver):
    """
    Resolves file:// URIs and plain filesystem paths.
    """

    base_directory: Optional[str] = None
    """
    The base directory to resolve relative paths from.
    """

    def supports(self, uri: str) -> bool:
        return uri.startswith("file://") or uri.startswith("/") or "://" not in uri

    def resolve(self, uri: str) -> str:
        if uri.startswith("file://"):
            if uri.startswith("file:///"):
                path = uri[len("file://") :]  # absolute
            else:
                path = uri.replace("file://", "", 1)  # relative
        else:
            path = uri

        if self.base_directory and not os.path.isabs(path):
            path = os.path.join(self.base_directory, path)

        return os.path.abspath(path)


@dataclass
class SearchPathFileResolver(PathResolver):
    """
    Resolves relative paths and relative ``file://`` URIs against several root
    directories, returning the first root that holds the file.

    Descriptions that are meant to be read with a resource search path state their
    relative paths from the root of the package rather than from the file that contains
    them, so the root has to be searched for.
    """

    root_directories: List[str] = field(default_factory=list)
    """
    The directories that relative paths are resolved against, in order of precedence.
    """

    def supports(self, uri: str) -> bool:
        return uri.startswith("file://") or "://" not in uri

    def resolve(self, uri: str) -> str:
        path = uri[len("file://") :] if uri.startswith("file://") else uri
        if os.path.isabs(path):
            return os.path.abspath(path)

        for directory in self.root_directories:
            candidate = os.path.join(directory, path)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

        raise PathResolutionError(
            uri=uri,
            details=f"not found below {', '.join(self.root_directories)}",
        )


@dataclass
class ModelUriResolver(PathResolver):
    """
    Resolves ``model://`` URIs against Gazebo model directories.

    A bare ``model://NAME`` resolves to the model's directory. Interpreting the
    ``model.config`` inside that directory is left to the format parser.
    """

    model_directories: List[str] = field(default_factory=list)
    """
    Directories that are searched for models before the environment is consulted.
    """

    def supports(self, uri: str) -> bool:
        return uri.startswith("model://")

    def search_directories(self) -> List[str]:
        """
        :return: All directories searched for models, in order of precedence.
        """
        environment_directories = [
            directory
            for directory in os.environ.get("GAZEBO_MODEL_PATH", "").split(":")
            if directory
        ]
        default_directory = os.path.join(os.path.expanduser("~"), ".gazebo", "models")
        return [*self.model_directories, *environment_directories, default_directory]

    def resolve(self, uri: str) -> str:
        rest = uri[len("model://") :]
        model_name, _, relative_path = rest.partition("/")
        search_directories = self.search_directories()
        for directory in search_directories:
            model_directory = os.path.join(directory, model_name)
            if not os.path.isdir(model_directory):
                continue
            return os.path.abspath(os.path.join(model_directory, relative_path))

        raise PathResolutionError(
            uri=uri,
            details=f"model '{model_name}' not found in {', '.join(search_directories)}",
        )


@dataclass
class CompositePathResolver(PathResolver):
    """
    Tries multiple path resolvers in order.
    """

    resolvers: List[PathResolver] = field(
        default_factory=lambda: [
            FileUriResolver(),
            PackageUriResolver(),
        ]
    )

    def supports(self, uri: str) -> bool:
        """
        Checks if the URI is supported by any of the resolvers.
        """
        return any(resolver.supports(uri) for resolver in self.resolvers)

    def resolve(self, uri: str) -> str:
        """
        Resolves a URI to an absolute local file path.
        """
        errors = []
        for resolver in self.resolvers:
            if not resolver.supports(uri):
                continue
            try:
                return resolver.resolve(uri)
            except ParsingError as error:
                errors.append(str(error))

        raise PathResolutionError(uri=uri, details="; ".join(errors))
