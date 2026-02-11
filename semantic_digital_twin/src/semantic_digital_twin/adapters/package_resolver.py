from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import List

from ..exceptions import ParsingError


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
            raise ParsingError(
                message=f"Ament could not resolve package '{package_name}': {error}"
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
            candidate = os.path.join(root, package_name)
            if os.path.isdir(candidate):
                return candidate
        raise ParsingError(
            message=f"Package '{package_name}' not found in ROS_PACKAGE_PATH."
        )


@dataclass
class ROSPackageLocator(PackageLocator):
    """
    Tries multiple package locators in order.
    """

    locators: List[PackageLocator] = field(
        default_factory=lambda: [AmentPackageLocator(), ROSPackagePathLocator()]
    )

    def resolve(self, package_name: str) -> str:
        errors = []
        for locator in self.locators:
            try:
                return locator.resolve(package_name)
            except ParsingError as error:
                errors.append(str(error))
        raise ParsingError(
            message=f"Could not resolve package '{package_name}'. Details: {'; '.join(errors)}"
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

    def supports(self, uri: str) -> bool:
        return uri.startswith("file://") or uri.startswith("/") or "://" not in uri

    def resolve(self, uri: str) -> str:
        path = uri
        if uri.startswith("file://"):
            path = (
                uri.replace("file://", "./", 1)
                if not uri.startswith("file:///")
                else uri[len("file://") :]
            )
        return path


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

        for resolver in self.resolvers:
            if isinstance(resolver, FileUriResolver):
                return resolver.resolve(uri)
        raise ParsingError(
            message=f"Could not resolve path '{uri}'. Details: {'; '.join(errors)}"
        )
