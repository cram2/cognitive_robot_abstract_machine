"""
Static scan of the CRAM repository's architecture (packages, classes, imports).
"""

from __future__ import annotations

import ast
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import Any, ClassVar, Dict, List, Optional, Set, Tuple

from cramera import paths
from cramera.knowledge.architecture_entities import Package, PythonClass
from cramera.generated_json import GeneratedJson


@dataclass
class PackageDependency:
    """
    One import edge between two top-level packages.
    """

    source: str
    """
    Package doing the importing.
    """

    target: str
    """
    Package being imported.
    """


@dataclass
class ArchitectureScan:
    """
    The CRAM repository's static architecture graph.
    """

    packages: List[Package]
    """
    Every top-level package found.
    """

    classes: List[PythonClass]
    """
    Every class found across all packages.
    """

    dependency_edges: List[PackageDependency]
    """
    Cross-package import edges.
    """


@dataclass
class RawArchitectureScan:
    """
    One architecture scan in the flat, JSON-able shape the disk cache stores.

    :class:`ArchitectureScan` is the same content as real entities;
    :meth:`ArchitectureScanner._typed` converts between them. The raw form exists
    because the cache is plain JSON and a full scan takes seconds.
    """

    packages: List[Dict[str, Any]] = field(default_factory=list)
    """
    One dict per workspace member: name, description and its two counts.
    """

    classes: List[Dict[str, Any]] = field(default_factory=list)
    """
    One dict per scanned class, without its subpackage — that is derived on typing.
    """

    dependency_edges: List[Tuple[str, str]] = field(default_factory=list)
    """
    Sorted ``(importing package, imported package)`` pairs.
    """


@dataclass
class ArchitectureScanner:
    """
    Scans one CRAM repository's architecture, cached to disk between runs.

    The repository root is held here rather than read inside each method, so a scan is
    reproducible and can be pointed at another checkout without touching the
    environment.
    """

    DESCRIPTION_LENGTH_LIMIT: ClassVar[int] = 120
    """
    How much of a README's first line is kept as a package description.
    """

    ARCHITECTURE_CACHE_VERSION: ClassVar[int] = 3
    """
    Bumped whenever the cached scan's shape changes, so old caches are discarded.
    """

    SKIPPED_DIRECTORIES: ClassVar[Set[str]] = {
        "__pycache__",
        "node_modules",
        "doc",
        "docs",
        "resources",
        "build",
        "dist",
        "plugins",
    }
    """
    Directories never descended into during the architecture scan.
    """

    PACKAGE_DESCRIPTIONS: ClassVar[Dict[str, str]] = {
        "krrood": "knowledge representation & reasoning through OO design (home of EQL)",
        "coraplex": "the plan executive: designators, plans, locations",
        "pycram": "legacy plan executive (resources/demos)",
        "giskardpy": "constraint-based motion planning and control",
        "robokudo": "perception framework",
        "semantic_digital_twin": "semantic world model / digital twin",
        "segmind": "segmentation / vision models",
        "probabilistic_model": "probabilistic models and inference",
        "random_events": "sigma-algebra & random events for probabilistic reasoning",
        "physics_simulators": "physics simulator bindings",
        "experiments": "experiment scripts (incl. EQL experiments)",
        "test": "the test suites of all packages",
        "scripts": "maintenance scripts",
        "root": "top-level demo scripts (sterility test, wind turbine…)",
    }
    """
    Curated one-line descriptions for the well-known workspace packages.
    """

    root: str
    """
    Path of the CRAM repository being scanned.
    """

    @classmethod
    def of_configured_root(cls) -> "ArchitectureScanner":
        """
        A scanner for the repository :func:`cramera.paths.architecture_root` points at.
        """
        return cls(root=str(paths.architecture_root()))

    def scan(self) -> ArchitectureScan:
        """
        Statically scan the CRAM repository for its architecture graph.

        A pure ``ast`` parse — nothing is imported.
        """
        return self._typed(self._scan_raw())

    def load(self) -> ArchitectureScan:
        """
        :meth:`scan` behind a JSON disk cache.

        A full scan takes seconds, so results are cached in the data directory, keyed by
        the scanned root; a cache from another root is rescanned.
        """
        return self._typed(self._load_raw())

    def _architecture_cache(self) -> str:
        """
        Path of the scan cache — always in the writable data directory, because the
        scenes checkout may be read-only.
        """
        return os.path.join(str(paths.data_directory()), "architecture_cache.json")

    def _first_readme_line(self, directory: str) -> str:
        """
        The first non-empty line of a directory's README, or ``''``.

        :param directory: The directory to look for a README in.
        """
        for name in ("README.md", "readme.md"):
            readme_path = Path(directory) / name
            if not readme_path.is_file():
                continue
            text = readme_path.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                stripped = line.strip().lstrip("#").strip()
                if stripped:
                    return stripped[: self.DESCRIPTION_LENGTH_LIMIT]
        return ""

    def _scan_raw(self) -> RawArchitectureScan:
        """
        Statically scan the CRAM repository into the shape the disk cache stores.
        """
        raw = RawArchitectureScan()
        if not os.path.isdir(self.root):
            return raw

        package_directories = self._package_directories()
        package_names = set(package_directories)
        imports: Dict[str, Set[str]] = {}
        modules_per_package = {
            package: self._scan_package_modules(
                package, base, package_names, raw.classes, imports
            )
            for package, base in package_directories.items()
        }

        class_counts = Counter(entry["package"] for entry in raw.classes)
        raw.packages = [
            dict(
                name=package,
                description=self.PACKAGE_DESCRIPTIONS.get(package)
                or self._first_readme_line(directory),
                module_count=modules_per_package.get(package, 0),
                class_count=class_counts.get(package, 0),
            )
            for package, directory in package_directories.items()
        ]
        raw.dependency_edges = sorted(
            (source, target)
            for source, targets in imports.items()
            for target in targets
        )
        return raw

    def _package_directories(self) -> Dict[str, str]:
        """
        Every workspace member of the repository, plus ``root`` for its loose scripts.
        """
        directories = {"root": self.root}
        for entry in sorted(os.listdir(self.root)):
            directory = os.path.join(self.root, entry)
            if (
                os.path.isdir(directory)
                and not entry.startswith(".")
                and entry not in self.SKIPPED_DIRECTORIES
                and "egg-info" not in entry
            ):
                directories[entry] = directory
        return directories

    def _scan_package_modules(
        self,
        package: str,
        base: str,
        package_names: Set[str],
        classes: List[Dict[str, Any]],
        imports: Dict[str, Set[str]],
    ) -> int:
        """
        Parse every module of one package, collecting its classes and imports.

        :param package: Name of the package being scanned.
        :param base: Directory the package lives in.
        :param package_names: Every known package, so imports of other ones are edges.
        :param classes: Collects one dict per class found.
        :param imports: Collects the packages each package imports from.
        :return: How many modules were parsed.
        """
        module_count = 0
        for directory_path, directory_names, filenames in os.walk(base):
            directory_names[:] = [
                name
                for name in directory_names
                if not name.startswith(".") and name not in self.SKIPPED_DIRECTORIES
            ]
            if package == "root":
                directory_names[:] = []  # root package = top-level scripts only
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                path = os.path.join(directory_path, filename)
                tree = self._parsed_module(path)
                if tree is None:
                    continue
                module_count += 1
                module = os.path.relpath(path, self.root)[:-3].replace(os.sep, ".")
                self._collect_classes_and_imports(
                    tree, package, module, package_names, classes, imports
                )
        return module_count

    @staticmethod
    def _parsed_module(path: str) -> Optional[ast.Module]:
        """
        One module's syntax tree, or None when it cannot be parsed.

        A module the running interpreter cannot read (newer syntax, or a template)
        contributes nothing to the graph.

        :param path: Path of the module to parse.
        """
        source = Path(path).read_text(encoding="utf-8", errors="replace")
        try:
            return ast.parse(source)
        except SyntaxError:
            return None

    @staticmethod
    def _collect_classes_and_imports(
        tree: ast.Module,
        package: str,
        module: str,
        package_names: set,
        classes: List[Dict[str, Any]],
        imports: Dict[str, set],
    ) -> None:
        """
        Collect class definitions and cross-package imports from one module.

        :param tree: Parsed AST of the module.
        :param package: Name of the package the module belongs to.
        :param module: Dotted module path, used to qualify collected classes.
        :param package_names: Every known top-level package name, to recognize imports.
        :param classes: Output list class dicts are appended to.
        :param imports: Output mapping package name to the set of packages it imports;
            updated in place.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = tuple(
                    (
                        base.id
                        if isinstance(base, ast.Name)
                        else (base.attr if isinstance(base, ast.Attribute) else "?")
                    )
                    for base in node.bases
                )
                docstring_summary = (
                    (ast.get_docstring(node) or "").strip().split("\n")[0][:140]
                )
                methods = sum(
                    1
                    for member in node.body
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                )
                classes.append(
                    dict(
                        name=node.name,
                        package=package,
                        module=module,
                        bases=list(bases),
                        methods=methods,
                        docstring_summary=docstring_summary,
                    )
                )
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    roots = [alias.name.split(".")[0] for alias in node.names]
                elif node.level == 0:
                    roots = [(node.module or "").split(".")[0]]
                else:
                    roots = []
                for root in roots:
                    if root in package_names and root != package:
                        imports.setdefault(package, set()).add(root)

    def _load_cache(self, cram_root: str, require_classes: bool) -> Optional[tuple]:
        """
        The cached scan if it is usable, else None.

        A cache written for another repository root is not trusted (unless no repository
        exists at all, in which case any cache beats nothing).

        :param cram_root: The current CRAM repository root, for the cache's origin
            check.
        :param require_classes: Whether a cache with no classes should be rejected.
        """
        cache_path = Path(self._architecture_cache())
        if not cache_path.is_file():
            return None
        cached = GeneratedJson(cache_path).read()
        if not isinstance(cached, dict):
            return None
        if cached.get("version") != self.ARCHITECTURE_CACHE_VERSION:
            return None
        if os.path.isdir(cram_root) and cached.get("cram_root") != cram_root:
            return None
        if require_classes and not cached.get("classes"):
            return None
        return RawArchitectureScan(
            packages=cached["packages"],
            classes=cached["classes"],
            dependency_edges=[tuple(edge) for edge in cached["dependency_edges"]],
        )

    def _load_raw(self) -> RawArchitectureScan:
        """
        :meth:`_scan_raw` behind the JSON disk cache.
        """
        cram_root = self.root
        cached = self._load_cache(cram_root, require_classes=False)
        if cached is not None:
            return cached
        if not os.path.isdir(cram_root):
            return RawArchitectureScan()
        raw = self._scan_raw()
        if not raw.classes:
            # a checkout exists but yielded nothing (empty or partial clone) —
            # fall back to the cache rather than losing the architecture graph
            return self._load_cache(cram_root, require_classes=True) or raw
        cache_path = Path(self._architecture_cache())
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # written via a temporary file: a half-written cache would be read back as
        # a complete one on the next start
        temporary = cache_path.with_suffix(".part")
        temporary.write_text(
            json.dumps(
                {
                    "version": self.ARCHITECTURE_CACHE_VERSION,
                    "cram_root": cram_root,
                    "packages": raw.packages,
                    "classes": raw.classes,
                    "dependency_edges": raw.dependency_edges,
                }
            ),
            encoding="utf-8",
        )
        temporary.replace(cache_path)
        return raw

    @staticmethod
    def _subpackage_of(package: str, module: str) -> str:
        """
        Qualified subpackage of a module path.

        ``coraplex.src.coraplex.plans.designator`` → ``coraplex.plans``; top-level
        modules collapse onto the package itself.

        :param package: Name of the module's top-level package.
        :param module: Dotted module path.
        """
        segments = module.split(".")
        if segments and segments[0] == package:
            segments = segments[1:]
        while segments and segments[0] in ("src", package):
            segments = segments[1:]
        return package + "." + segments[0] if len(segments) >= 2 else package

    def _typed(self, raw: RawArchitectureScan) -> ArchitectureScan:
        """
        The cached, flat scan converted into real entities.

        :param raw: The scan as the cache stores it.
        """
        return ArchitectureScan(
            packages=[Package(**entry) for entry in raw.packages],
            classes=[
                PythonClass(
                    name=entry["name"],
                    package=entry["package"],
                    subpackage=self._subpackage_of(entry["package"], entry["module"]),
                    module=entry["module"],
                    bases=tuple(entry["bases"]),
                    methods=entry["methods"],
                    docstring_summary=entry["docstring_summary"],
                )
                for entry in raw.classes
            ],
            dependency_edges=[
                PackageDependency(source, target)
                for source, target in raw.dependency_edges
            ],
        )
