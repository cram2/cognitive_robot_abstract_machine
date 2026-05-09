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
``--dry-run``
    Compute and print the generated content without writing any files.
    Useful for inspecting what *would* change before committing.

``-v`` / ``--verbose``
    Enable DEBUG-level logging for detailed progress information.

Exit codes
----------
``0``  All packages processed successfully.
``1``  One or more packages failed (details logged to stderr).

Examples
--------
Generate mixins for a single package::

    krrood-generate-role-mixins pycram

Generate mixins for multiple packages in one call::

    krrood-generate-role-mixins pycram semantic_digital_twin

Inspect what would be written without touching the filesystem::

    krrood-generate-role-mixins pycram --dry-run

Run via the module entry point (useful when the package is not installed)::

    python -m krrood.generate_role_mixins pycram --verbose
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from dataclasses import is_dataclass

from krrood import logger as krrood_logger
from krrood.class_diagrams import ClassDiagram
from krrood.ormatic.utils import classes_of_package
from krrood.patterns.role.helpers import transform_roles_in_class_diagram


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


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``krrood-generate-role-mixins`` console script.

    :param argv: Argument list (defaults to ``sys.argv[1:]``).  Pass an
        explicit list for programmatic use or in tests.
    :return: ``0`` on full success, ``1`` if any package failed.
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
