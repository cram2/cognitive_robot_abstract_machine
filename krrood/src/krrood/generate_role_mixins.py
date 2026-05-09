"""
krrood-generate-role-mixins — Offline Role Mixin Generator
===========================================================

Generates Role mixin files for all role patterns found in one or more
Python packages.  Run this **once before running tests**, or as a CI setup
step, so that test processes never need to reload modules at runtime (which
would break ``isinstance`` checks for already-imported class references).

Usage (console script, available after ``pip install krrood``):

    krrood-generate-role-mixins pycram semantic_digital_twin

Usage (module invocation, no install required):

    python -m krrood.generate_role_mixins pycram semantic_digital_twin

Arguments
---------
``package ...``
    One or more fully-qualified, importable Python package names to scan.
    Every dataclass in the package (and its sub-packages) is inspected; those
    that participate in the Role pattern receive generated mixin stub files.

Options
-------
``--check``
    Verify that all mixin files are semantically up-to-date without writing
    anything.  Exits 0 if all files match, 1 if any are missing or differ.
    Useful in CI to assert that committed mixin files are not stale.
    *Semantic* means whitespace and formatting differences are ignored; only
    actual code changes (new methods, changed annotations, etc.) count.

``--dry-run``
    Compute and print the generated content without writing any files.
    Useful for inspecting what *would* change before committing.

``-v`` / ``--verbose``
    Enable DEBUG-level logging for detailed progress information.

Exit codes
----------
``0``  All packages processed successfully (or up-to-date in --check mode).
``1``  One or more packages failed or have stale files (in --check mode).

Examples
--------
Generate mixins for a single package::

    krrood-generate-role-mixins pycram

Generate mixins for multiple packages in one call::

    krrood-generate-role-mixins pycram semantic_digital_twin

Check that committed mixin files are current (CI lint step)::

    krrood-generate-role-mixins --check pycram semantic_digital_twin

Inspect what would be written without touching the filesystem::

    krrood-generate-role-mixins pycram --dry-run

Run via the module entry point (useful when the package is not installed)::

    python -m krrood.generate_role_mixins pycram --verbose
"""

from __future__ import annotations

import argparse
import ast
import importlib
import logging
import sys
from dataclasses import is_dataclass
from pathlib import Path
from types import ModuleType

from krrood import logger as krrood_logger
from krrood.class_diagrams import ClassDiagram
from krrood.ormatic.utils import classes_of_package
from krrood.patterns.role.helpers import transform_roles_in_class_diagram


# ── semantic comparison ───────────────────────────────────────────────────────


def _are_semantically_equal(a: str, b: str) -> bool:
    """Return *True* if two Python source strings are semantically equivalent.

    Parses both strings to an AST and compares the dumps, so differences that
    are purely cosmetic (whitespace, blank lines, Black reformatting) are
    ignored.  Only actual code changes — new or modified statements, changed
    annotations, renamed identifiers — cause the comparison to return *False*.

    Falls back to exact string equality if either source fails to parse (e.g.
    a partially-written or corrupt file).

    :param a: First Python source string.
    :param b: Second Python source string.
    :return: *True* if semantically equivalent, *False* otherwise.
    """
    try:
        return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))
    except SyntaxError:
        return a == b


# ── staleness check ───────────────────────────────────────────────────────────


def _stale_files_for_package(package_name: str) -> list[Path]:
    """Return every mixin or transformed-source path that is missing or semantically stale.

    Runs ``transform(write=False)`` (in the current process, reloads happen) to
    obtain the *would-be* generated content, then compares it semantically with
    what is on disk.

    :param package_name: Fully-qualified import name of the package to check.
    :return: List of :class:`~pathlib.Path` objects for files that need writing.
    :raises ImportError: If *package_name* cannot be imported.
    """
    from krrood.patterns.role.helpers import _modules_with_roles
    from krrood.patterns.role.role_transformer import RoleTransformer

    package = importlib.import_module(package_name)
    all_classes = [c for c in classes_of_package(package) if is_dataclass(c)]
    class_diagram = ClassDiagram(all_classes)

    stale: list[Path] = []
    for module in _modules_with_roles(class_diagram):
        transformer = RoleTransformer(module)
        result = transformer.transform(write=False)
        for m, (transformed_source, mixin_content) in result.items():
            mixin_path = transformer.get_generated_file_path(m, is_mixin=True)
            if not mixin_path.exists() or not _are_semantically_equal(
                mixin_path.read_text(), mixin_content
            ):
                stale.append(mixin_path)
            original_path = RoleTransformer.get_module_file_path(m)
            if not _are_semantically_equal(original_path.read_text(), transformed_source):
                stale.append(original_path)
    return stale


# ── pytest helper ─────────────────────────────────────────────────────────────


def ensure_role_mixins_current_for_pytest(packages: list[str]) -> None:
    """Ensure role mixin files are semantically current; auto-restart pytest if not.

    Designed for use in ``pytest_configure`` hooks.  All checking and
    generation runs in subprocesses so the test process is **never** polluted
    by module reloads (which would break ``isinstance`` checks).

    Behaviour
    ~~~~~~~~~
    1. Spawns ``krrood-generate-role-mixins --check <packages>`` as a
       subprocess.
    2. Exit code 0 → all files are current; returns immediately.
    3. Exit code 1 → some files are stale:

       a. Spawns ``krrood-generate-role-mixins <packages>`` to regenerate.
       b. Decrements ``KRROOD_PYTEST_RERUN_COUNT`` (default ``2``).
       c. Replaces the current process with a fresh pytest run via
          ``os.execv`` so tests see the updated files in a clean import state.

    If *KRROOD_PYTEST_RERUN_COUNT* reaches 0 or regeneration fails, raises
    ``pytest.UsageError`` with a manual-run command in the message.

    :param packages: Package names to check (e.g. ``["pycram"]``).
    :raises pytest.UsageError: On regeneration failure or exceeded retry limit.
    """
    import os
    import subprocess

    import pytest

    remaining = int(os.environ.get("KRROOD_PYTEST_RERUN_COUNT", "2"))

    check_result = subprocess.run(
        [sys.executable, "-m", "krrood.generate_role_mixins", "--check"] + packages,
        capture_output=True,
    )
    if check_result.returncode == 0:
        return

    gen_result = subprocess.run(
        [sys.executable, "-m", "krrood.generate_role_mixins"] + packages,
    )
    if gen_result.returncode != 0:
        cmd = "krrood-generate-role-mixins " + " ".join(packages)
        raise pytest.UsageError(
            f"Role mixin regeneration failed.\nRun manually:\n\n    {cmd}"
        )

    if remaining <= 0:
        cmd = "krrood-generate-role-mixins " + " ".join(packages)
        raise pytest.UsageError(
            f"Role mixin files are still outdated after regeneration attempts.\n"
            f"Run manually:\n\n    {cmd}"
        )

    os.environ["KRROOD_PYTEST_RERUN_COUNT"] = str(remaining - 1)
    os.execv(sys.executable, [sys.executable, "-m", "pytest"] + sys.argv[1:])


# ── generation ────────────────────────────────────────────────────────────────


def generate_role_mixins_for_package(package_name: str, write: bool = True) -> None:
    """Scan *package_name* and generate (or preview) its role mixin stub files.

    Imports the package, collects every dataclass defined in it (recursively),
    builds a :class:`~krrood.class_diagrams.ClassDiagram`, and delegates to
    :func:`~krrood.patterns.role.helpers.transform_roles_in_class_diagram`.

    This function is safe to call from any context — it does **not** reload
    already-imported modules in the calling process because it is designed to
    run in a fresh process (e.g. a CI step or a dedicated pre-test script).

    :param package_name: Fully-qualified import name of the package to scan
        (e.g. ``"pycram"`` or ``"semantic_digital_twin"``).
    :param write: When *True* (default) write mixin and transformed source
        files to disk.  When *False* compute the content without writing
        (dry-run mode).
    :raises ImportError: If *package_name* cannot be imported.
    :raises Exception: Propagates any error raised during transformation.
    """
    package = importlib.import_module(package_name)
    all_classes = [c for c in classes_of_package(package) if is_dataclass(c)]
    class_diagram = ClassDiagram(all_classes)
    transform_roles_in_class_diagram(class_diagram, write=write)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``krrood-generate-role-mixins`` console script.

    :param argv: Argument list (defaults to ``sys.argv[1:]``).  Pass an
        explicit list for programmatic use or in tests.
    :return: ``0`` on full success / all-current (check mode),
             ``1`` if any package failed or has stale files (check mode).
    """
    parser = argparse.ArgumentParser(
        prog="krrood-generate-role-mixins",
        description=(
            "Generate Role mixin stub files for role patterns found in one or "
            "more Python packages.  Intended to be run offline (e.g. in CI or "
            "as a pre-commit step) so that test processes never reload modules "
            "at runtime."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  krrood-generate-role-mixins pycram\n"
            "  krrood-generate-role-mixins pycram semantic_digital_twin\n"
            "  krrood-generate-role-mixins --check pycram semantic_digital_twin\n"
            "  krrood-generate-role-mixins pycram --dry-run\n"
            "  python -m krrood.generate_role_mixins pycram --verbose\n"
        ),
    )
    parser.add_argument(
        "packages",
        nargs="+",
        metavar="package",
        help=(
            "Importable Python package name(s) to scan for role patterns "
            "(e.g. 'pycram' or 'semantic_digital_twin')."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Verify mixin files are semantically up-to-date without writing.  "
            "Exits 0 if all files match what would be generated, 1 if any are "
            "missing or contain semantic differences (whitespace and formatting "
            "are ignored).  Useful as a CI lint step."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Compute the generated content without writing any files.  "
            "Useful for previewing changes before committing."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging for detailed progress output.",
    )
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        krrood_logger.setLevel(logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    log = logging.getLogger(__name__)

    if args.check:
        stale_by_pkg: dict[str, list[Path]] = {}
        for package_name in args.packages:
            try:
                stale = _stale_files_for_package(package_name)
                if stale:
                    stale_by_pkg[package_name] = stale
            except Exception as exc:
                log.error("  Failed to check %s — %s", package_name, exc)
                stale_by_pkg[package_name] = []

        if stale_by_pkg:
            for pkg, paths in stale_by_pkg.items():
                if paths:
                    log.info(
                        "Stale mixin files in %s:\n  %s",
                        pkg,
                        "\n  ".join(str(p) for p in paths),
                    )
                else:
                    log.info("Check failed for %s (see errors above)", pkg)
            return 1
        return 0

    failed: list[str] = []
    for package_name in args.packages:
        try:
            log.info("Processing package: %s", package_name)
            generate_role_mixins_for_package(package_name, write=not args.dry_run)
            action = "Dry-run complete" if args.dry_run else "Done"
            log.info("  %s: %s", action, package_name)
        except Exception as exc:
            log.error("  Failed: %s — %s", package_name, exc)
            failed.append(package_name)

    if failed:
        log.error("The following packages failed: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
