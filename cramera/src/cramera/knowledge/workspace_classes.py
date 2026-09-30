"""
Making the workspace's own classes nameable in an EQL query.

The architecture scan already knows every class of every workspace package; this turns
that knowledge into names a query may use, resolved to the real class on first use.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from enum import Enum

from typing_extensions import Any, ClassVar, Dict, List, Optional, Tuple

from cramera.knowledge.architecture_entities import PythonClass
from cramera.knowledge.architecture_scan import ArchitectureScanner

SOURCE_DIRECTORY_SEGMENT = "src"
"""
Path segment separating a workspace member from the package it installs: everything
after it is the class's importable module path.
"""

GENERATED_CLASS_SUFFIX = "DAO"
"""
Suffix of the ORM classes generated from the workspace's own dataclasses.
"""

GENERATED_ASSOCIATION_MARKER = "_association"
"""
Marker in the name of a generated ORM association class.
"""


class WorkspacePackage(Enum):
    """
    A package a query may name classes from, declared in the order a contested name is
    awarded: the packages a question about a scene is usually about come first.
    """

    SEMANTIC_DIGITAL_TWIN = "semantic_digital_twin"
    CORAPLEX = "coraplex"
    KRROOD = "krrood"
    SEGMIND = "segmind"
    EXPERIMENTS = "experiments"
    CRAMERA = "cramera"
    GISKARDPY = "giskardpy"
    PHYSICS_SIMULATORS = "physics_simulators"
    PROBABILISTIC_MODEL = "probabilistic_model"
    RANDOM_EVENTS = "random_events"
    ROBOKUDO = "robokudo"

    @classmethod
    def named(cls, package: str) -> Optional[WorkspacePackage]:
        """
        The package of this name, or None for one no query names classes from.

        :param package: Top-level package name as the architecture scan reports it.
        """
        return next(
            (member for member in cls if member.value == package),
            None,
        )

    @property
    def rank(self) -> int:
        """
        Position in the declaration order; the lower rank wins a contested name.
        """
        return list(type(self)).index(self)


@dataclass(frozen=True)
class ClassLocation:
    """
    One class a query may name, and where it is defined.
    """

    name: str
    """
    The class's own name, as a query writes it.
    """

    module: str
    """
    Importable dotted path of the module defining the class.
    """

    package: WorkspacePackage
    """
    Workspace package the class belongs to.
    """

    docstring_summary: str
    """
    First line of the class's docstring, or ``''``.
    """

    @classmethod
    def of_scanned_class(cls, scanned: PythonClass) -> Optional[ClassLocation]:
        """
        Where a scanned class can be named from, or None when it cannot be named.

        A class is nameable when it belongs to a workspace package a query ranges over,
        lives in that package's installed source tree, and was written by hand rather
        than generated as an ORM mapping.

        :param scanned: The class as the architecture scan reports it.
        """
        package = WorkspacePackage.named(scanned.package)
        if package is None or cls.is_generated(scanned.name):
            return None
        segments = scanned.module.split(".")
        if SOURCE_DIRECTORY_SEGMENT not in segments:
            return None
        module = ".".join(segments[segments.index(SOURCE_DIRECTORY_SEGMENT) + 1 :])
        if not module:
            return None
        return cls(
            name=scanned.name,
            module=module,
            package=package,
            docstring_summary=scanned.docstring_summary,
        )

    @staticmethod
    def is_generated(name: str) -> bool:
        """
        Whether a class of this name is an ORM mapping generated from another class.

        :param name: The class's own name.
        """
        return (
            name.endswith(GENERATED_CLASS_SUFFIX)
            or GENERATED_ASSOCIATION_MARKER in name
        )

    def load(self) -> Any:
        """
        Import the defining module and return the class itself.
        """
        return importlib.import_module(self.module).__dict__[self.name]


@dataclass
class WorkspaceClassIndex:
    """
    Every class of the workspace a query may name, by name.
    """

    locations: Dict[str, Tuple[ClassLocation, ...]] = field(default_factory=dict)
    """
    Where each nameable class is defined, best candidate first.
    """

    _of_root: ClassVar[Dict[str, WorkspaceClassIndex]] = {}
    """
    One index per scanned repository root, kept because a scan costs seconds on a cold
    cache, and keyed by root so pointing cramera at another checkout rebuilds it.
    """

    @classmethod
    def of_scanned_classes(cls, scanned: List[PythonClass]) -> WorkspaceClassIndex:
        """
        An index over the nameable classes among those scanned.

        :param scanned: Classes as the architecture scan reports them.
        """
        locations: Dict[str, List[ClassLocation]] = {}
        for scanned_class in scanned:
            location = ClassLocation.of_scanned_class(scanned_class)
            if location is not None:
                locations.setdefault(location.name, []).append(location)
        return cls(
            locations={
                name: tuple(sorted(candidates, key=lambda found: found.package.rank))
                for name, candidates in locations.items()
            }
        )

    @classmethod
    def of_repository(cls) -> WorkspaceClassIndex:
        """
        An index over the classes of the repository cramera was started from, built on
        first use and cached per repository root.
        """
        scanner = ArchitectureScanner.of_configured_root()
        if scanner.root not in cls._of_root:
            cls._of_root[scanner.root] = cls.of_scanned_classes(scanner.load().classes)
        return cls._of_root[scanner.root]

    @classmethod
    def reset(cls) -> None:
        """
        Drop every cached index so the next access rebuilds it.
        """
        cls._of_root = {}

    def candidates(self, name: str) -> Tuple[ClassLocation, ...]:
        """
        Every class of this name, the one a query gets first.

        :param name: The class name a query uses.
        """
        return self.locations.get(name, ())

    def names(self) -> List[str]:
        """
        Every class name a query may use, alphabetically.
        """
        return sorted(self.locations)

    def resolve(self, name: str) -> Optional[Any]:
        """
        The class a query means by this name, or None when no class has it.

        :param name: The class name a query uses.
        """
        candidates = self.candidates(name)
        return candidates[0].load() if candidates else None


@dataclass
class WorkspaceClassNamespace(Dict[str, Any]):
    """
    The namespace one EQL query is evaluated in, backed by the class index.

    A name seeded into it -- a ready-made variable, an entity type, an EQL factory --
    answers as itself; anything else is looked up in the index the first time the query
    uses it, so naming a class costs an import only when a query actually names one.
    """

    index: WorkspaceClassIndex = field(default_factory=WorkspaceClassIndex)
    """
    Where a name the namespace was not seeded with is looked up.
    """

    def __missing__(self, name: str) -> Any:
        """
        Resolve a name through the index, or report it as absent.

        Raising ``KeyError`` is what makes the interpreter report an unresolvable name
        in a query as the ``NameError`` it would be in any other Python expression.

        :param name: The name the query used.
        :raises KeyError: When no class of the workspace has that name.
        """
        resolved = self.index.resolve(name)
        if resolved is None:
            raise KeyError(name)
        self[name] = resolved
        return resolved
