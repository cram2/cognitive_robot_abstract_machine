"""
Entities scanned from the CRAM repository's static architecture.
"""

from __future__ import annotations

from dataclasses import dataclass

from cramera.knowledge.entity import NamedEntity

from typing_extensions import Tuple


@dataclass(unsafe_hash=True)
class ModuleGrouping(NamedEntity):
    """
    A named group of Python modules found by the architecture scan.

    Both levels the scan distinguishes share this: a workspace member and a subpackage
    inside one are each a name with modules and classes counted under it.
    """

    module_count: int
    """
    Number of Python modules in the grouping.
    """

    class_count: int
    """
    Number of classes defined in the grouping.
    """


@dataclass(unsafe_hash=True)
class Package(ModuleGrouping):
    """
    A top-level package of the CRAM repository, i.e. one workspace member.

    Not every importable package is one of these: ``coraplex`` is, ``coraplex.plans``
    is a :class:`SubPackage`. Only a workspace member carries a description.
    """

    description: str = ""
    """
    One-line description (curated, or the first README line).
    """


@dataclass(unsafe_hash=True)
class SubPackage(ModuleGrouping):
    """
    A qualified subpackage, e.g. ``coraplex.plans``.
    """

    package: str = ""
    """
    The top-level package this subpackage belongs to.
    """


@dataclass(unsafe_hash=True)
class PythonClass(NamedEntity):
    """
    A class found by the static scan of the CRAM repository.
    """

    package: str
    """
    Top-level package the class is defined in.
    """

    subpackage: str
    """
    Qualified subpackage (equal to ``package`` for top-level modules).
    """

    module: str
    """
    Repository-relative module path.
    """

    bases: Tuple[str, ...]
    """
    Names of the direct base classes.
    """

    methods: int
    """
    Number of methods defined on the class.
    """

    docstring_summary: str
    """
    First docstring line, or ``''``.
    """

    @property
    def qualified_name(self) -> str:
        """
        The class name prefixed with the module it is defined in.

        Unique across the scan, which is what the graph uses as the class's node id.
        """
        return self.module + "." + self.name
