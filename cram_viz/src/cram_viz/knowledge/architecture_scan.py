"""
Static scan of the CRAM repository's architecture (packages, classes, imports).
"""

from __future__ import annotations

import ast
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Any, Dict, List, Optional, Tuple

from cram_viz import paths
from cram_viz.knowledge.architecture_entities import Package, PythonClass
from cram_viz.knowledge.scene_bundle import _read_json


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


class ArchitectureScanner:
    """
    Scans the CRAM repository's architecture, cached to disk between runs.
    """

    #: how much of a README's first line is kept as a package description
    DESCRIPTION_LENGTH_LIMIT = 120

    #: bumped whenever the cached scan's shape changes, so old caches are discarded
    ARCHITECTURE_CACHE_VERSION = 2

    #: directories never descended into during the architecture scan
    SKIP_DIRS = {
        "__pycache__",
        "node_modules",
        "doc",
        "docs",
        "resources",
        "build",
        "dist",
        "plugins",
    }

    #: curated one-line descriptions for the well-known workspace packages
    PKG_DESCRIPTIONS = {
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

    def scan(self) -> ArchitectureScan:
        """
        Statically scan the CRAM repository for its architecture graph.

        A pure ``ast`` parse — nothing is imported.
        """
        return self._typed(*self._scan_raw())

    def load(self) -> ArchitectureScan:
        """
        :meth:`scan` behind a JSON disk cache.

        A full scan takes seconds, so results are cached in the data directory, keyed by
        the scanned root; a cache from another root is rescanned.
        """
        return self._typed(*self._load_raw())

    def _cram_root(self) -> str:
        """
        The CRAM repository the architecture graph is scanned from.
        """
        return str(paths.architecture_root())

    def _architecture_cache(self) -> str:
        """
        Path of the scan cache — always in the writable data directory, because the
        scenes checkout may be read-only.
        """
        return os.path.join(str(paths.data_dir()), "arch_cache.json")

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

    def _scan_raw(
        self,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Tuple[str, str]]]:
        """
        Statically scan the CRAM repository, as plain JSON-able dicts/tuples — the shape
        the disk cache stores, converted to real types by :meth:`_typed`.
        """
        packages: List[Dict[str, Any]] = []
        classes: List[Dict[str, Any]] = []
        imports: Dict[str, set] = {}
        cram_root = self._cram_root()
        if not os.path.isdir(cram_root):
            return packages, classes, []

        package_dirs = {"root": cram_root}
        for entry in sorted(os.listdir(cram_root)):
            directory = os.path.join(cram_root, entry)
            if (
                os.path.isdir(directory)
                and not entry.startswith(".")
                and entry not in self.SKIP_DIRS
                and "egg-info" not in entry
            ):
                package_dirs[entry] = directory
        package_names = set(package_dirs)

        modules_per_package: Dict[str, int] = {}
        for package, base in package_dirs.items():
            module_count = 0
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = [
                    name
                    for name in dirnames
                    if not name.startswith(".") and name not in self.SKIP_DIRS
                ]
                if package == "root":
                    dirnames[:] = []  # root package = top-level scripts only
                for filename in filenames:
                    if not filename.endswith(".py"):
                        continue
                    path = os.path.join(dirpath, filename)
                    source = Path(path).read_text(encoding="utf-8", errors="replace")
                    try:
                        tree = ast.parse(source)
                    except SyntaxError:
                        # a module the running interpreter cannot parse (a newer
                        # syntax, or a template) contributes nothing to the graph
                        continue
                    module_count += 1
                    module = os.path.relpath(path, cram_root)[:-3].replace(os.sep, ".")
                    self._collect_classes_and_imports(
                        tree, package, module, package_names, classes, imports
                    )
            modules_per_package[package] = module_count

        class_counts = Counter(entry["package"] for entry in classes)
        for package in package_dirs:
            description = self.PKG_DESCRIPTIONS.get(package) or self._first_readme_line(
                package_dirs[package]
            )
            packages.append(
                dict(
                    name=package,
                    description=description,
                    module_count=modules_per_package.get(package, 0),
                    class_count=class_counts.get(package, 0),
                )
            )
        dependency_edges = sorted(
            (source, target)
            for source, targets in imports.items()
            for target in targets
        )
        return packages, classes, dependency_edges

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
                doc = (ast.get_docstring(node) or "").strip().split("\n")[0][:140]
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
                        doc=doc,
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
        cached = _read_json(cache_path)
        if not isinstance(cached, dict):
            return None
        if cached.get("version") != self.ARCHITECTURE_CACHE_VERSION:
            return None
        if os.path.isdir(cram_root) and cached.get("cram_root") != cram_root:
            return None
        if require_classes and not cached.get("classes"):
            return None
        return (
            cached["packages"],
            cached["classes"],
            [tuple(edge) for edge in cached["deps"]],
        )

    def _load_raw(
        self,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Tuple[str, str]]]:
        """
        :meth:`_scan_raw` behind the JSON disk cache.
        """
        cram_root = self._cram_root()
        cached = self._load_cache(cram_root, require_classes=False)
        if cached is not None:
            return cached
        if not os.path.isdir(cram_root):
            return [], [], []
        packages, classes, dependency_edges = self._scan_raw()
        if not classes:
            # a checkout exists but yielded nothing (empty or partial clone) —
            # fall back to the cache rather than losing the architecture graph
            return self._load_cache(cram_root, require_classes=True) or (
                packages,
                classes,
                dependency_edges,
            )
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
                    "packages": packages,
                    "classes": classes,
                    "deps": dependency_edges,
                }
            ),
            encoding="utf-8",
        )
        temporary.replace(cache_path)
        return packages, classes, dependency_edges

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

    def _typed(
        self,
        packages: List[Dict[str, Any]],
        classes: List[Dict[str, Any]],
        dependency_edges: List[Tuple[str, str]],
    ) -> ArchitectureScan:
        """
        The raw scan's dicts/tuples, converted into real entities.

        :param packages: Raw scanned packages, as plain dicts.
        :param classes: Raw scanned classes, as plain dicts.
        :param dependency_edges: Raw scanned package-to-package import edges.
        """
        return ArchitectureScan(
            packages=[Package(**entry) for entry in packages],
            classes=[
                PythonClass(
                    name=entry["name"],
                    package=entry["package"],
                    subpackage=self._subpackage_of(entry["package"], entry["module"]),
                    module=entry["module"],
                    bases=tuple(entry["bases"]),
                    methods=entry["methods"],
                    doc=entry["doc"],
                )
                for entry in classes
            ],
            dependency_edges=[
                PackageDependency(source, target) for source, target in dependency_edges
            ],
        )
