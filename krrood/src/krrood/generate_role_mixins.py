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
import enum
import importlib
import logging
import sys
from dataclasses import is_dataclass
from pathlib import Path
from types import ModuleType

from krrood import logger as krrood_logger
from krrood.class_diagrams import AllFieldsIntrospector, ClassDiagram
from krrood.ormatic.utils import classes_of_package
from krrood.patterns.code_generation.generated_code_file_writer import (
    has_class_definitions,
)
from krrood.patterns.role.helpers import transform_roles_in_class_diagram
from krrood.utils import run_black_on_file, run_ruff_on_file

# ── source normalisation ──────────────────────────────────────────────────────


def _format_source(source: str) -> str:
    """Return *source* after running the same ruff + black pipeline used by the writer.

    Writes *source* to a temporary file, applies ``ruff --fix`` then ``black``,
    and reads the result back.  If any step fails (e.g. a syntax error in the
    generated code), the original string is returned unchanged.

    :param source: Python source code string to format.
    :return: Formatted source string, or *source* unchanged on error.
    """
    import tempfile

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(source)
            tmp_path = tmp.name
        run_ruff_on_file(tmp_path)
        run_black_on_file(tmp_path)
        with open(tmp_path, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logging.getLogger(__name__).warning(
            "Formatting generated source failed with error: %s, returning unformatted source",
            e,
            exc_info=True,
        )
        return source
    finally:
        try:
            import os

            os.unlink(tmp_path)
        except Exception as e:
            logging.getLogger(__name__).debug(
                "Failed to delete temporary file %s: %s",
                tmp_path,
                e,
            )


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


# ── package resolution ─────────────────────────────────────────────────────────


def _resolve_package(package: str | ModuleType) -> ModuleType:
    """Resolve a package specifier to a :class:`ModuleType`.

    :param package: Dotted package name or an already-imported module object.
    :return: The imported package module.
    :raises TypeError: If *package* is neither a string nor a ModuleType.
    :raises ImportError: If *package* is a string and cannot be imported.
    """
    if isinstance(package, ModuleType):
        return package
    if isinstance(package, str):
        return importlib.import_module(package)
    raise TypeError(
        f"Expected a package name (str) or ModuleType, got {type(package).__name__}"
    )


# ── staleness check ───────────────────────────────────────────────────────────


def _stale_files_for_package(package: str | ModuleType) -> list[Path]:
    """Return every mixin or transformed-source path that is missing or semantically stale.

    Runs ``transform(write=False)`` (in the current process, reloads happen) to
    obtain the *would-be* generated content, then compares it semantically with
    what is on disk.

    :param package: Dotted package name or an already-imported module object.
    :return: List of :class:`~pathlib.Path` objects for files that need writing.
    :raises ImportError: If *package* is a string and cannot be imported.
    :raises TypeError: If *package* is neither a string nor a ModuleType.
    """
    from krrood.patterns.role.helpers import _modules_with_roles
    from krrood.patterns.role.role_transformer import RoleTransformer

    package = _resolve_package(package)
    all_classes = [c for c in classes_of_package(package) if is_dataclass(c)]
    class_diagram = ClassDiagram(all_classes, introspector=AllFieldsIntrospector())

    log = logging.getLogger(__name__)
    stale: list[Path] = []
    for module in _modules_with_roles(class_diagram):
        try:
            transformer = RoleTransformer(module)
            result = transformer.transform(write=False)
        except Exception as e:
            log.exception(
                "  Failed to transform module %s for staleness check with error: %s",
                module.__name__,
                e,
            )
            continue
        for m, (transformed_source, mixin_content) in result.items():
            mixin_path = transformer.get_generated_file_path(m, is_mixin=True)
            if has_class_definitions(mixin_content):
                formatted_mixin = _format_source(mixin_content)
                if not mixin_path.exists() or not _are_semantically_equal(
                    mixin_path.read_text(), formatted_mixin
                ):
                    stale.append(mixin_path)
            elif mixin_path.exists():
                stale.append(mixin_path)
            original_path = RoleTransformer.get_module_file_path(m)
            formatted_source = _format_source(transformed_source)
            if not _are_semantically_equal(original_path.read_text(), formatted_source):
                stale.append(original_path)
    return stale


# ── pytest helper ─────────────────────────────────────────────────────────────


def _get_env_with_pythonpath() -> dict[str, str]:
    """Return a copy of the current environment with PYTHONPATH set to sys.path."""
    import os
    import sys

    env = os.environ.copy()
    current_path = os.pathsep.join(sys.path)
    env["PYTHONPATH"] = current_path
    return env


class _ActionResult(enum.Enum):
    """Return-code for :func:`_ensure_role_mixins_current`."""

    RETURN = enum.auto()  # all current, return immediately
    RESTART = enum.auto()  # regenerated, restart pytest
    ERROR = enum.auto()  # regeneration failed or retries exhausted


def _ensure_role_mixins_current(
    packages: list[str | ModuleType],
) -> tuple[_ActionResult, int | str]:
    """Core logic — pure decision, never exits the process.

    Runs ``--check`` and regenerate subprocesses, then returns a decision
    tuple of ``(action, payload)``:

    * ``(RETURN, 0)`` — all files are current; caller should return.
    * ``(RESTART, exitcode)`` — files were regenerated; caller should restart
      pytest and exit with *exitcode*.
    * ``(ERROR, msg)`` — regeneration failed or retry count exhausted; caller
      should raise ``pytest.UsageError(msg)``.

    This function is safe to call from tests because it neither calls
    :func:`os._exit` nor raises :class:`pytest.UsageError` directly.
    """
    import os
    import subprocess
    import sys

    _package_names: list[str] = []
    for p in packages:
        if isinstance(p, ModuleType):
            _package_names.append(p.__name__)
        elif isinstance(p, str):
            _package_names.append(p)
        else:
            raise TypeError(
                f"Expected package name (str) or ModuleType, got {type(p).__name__}"
            )

    remaining = int(os.environ.get("KRROOD_PYTEST_RERUN_COUNT", "2"))

    check_result = subprocess.run(
        [sys.executable, "-m", "krrood.generate_role_mixins", "--check"]
        + _package_names,
        capture_output=True,
        env=_get_env_with_pythonpath(),
    )
    if check_result.returncode == 0:
        return _ActionResult.RETURN, 0

    if check_result.stderr:
        print(check_result.stderr.decode(), file=sys.stderr)

    gen_result = subprocess.run(
        [sys.executable, "-m", "krrood.generate_role_mixins"] + _package_names,
        env=_get_env_with_pythonpath(),
    )
    if gen_result.returncode != 0:
        cmd = "krrood-generate-role-mixins " + " ".join(_package_names)
        return (
            _ActionResult.ERROR,
            f"Role mixin regeneration failed.\nRun manually:\n\n    {cmd}",
        )

    if remaining <= 0:
        cmd = "krrood-generate-role-mixins " + " ".join(_package_names)
        return (
            _ActionResult.ERROR,
            f"Role mixin files are still outdated after regeneration attempts.\n"
            f"Run manually:\n\n    {cmd}",
        )

    os.environ["KRROOD_PYTEST_RERUN_COUNT"] = str(remaining - 1)
    result = subprocess.run(
        [sys.executable, "-m", "pytest"] + sys.argv[1:],
        env=_get_env_with_pythonpath(),
    )
    return _ActionResult.RESTART, result.returncode


def ensure_role_mixins_current_for_pytest(
    packages: list[str | ModuleType],
) -> None:
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
          ``os._exit`` so tests see the updated files in a clean import state.

    If *KRROOD_PYTEST_RERUN_COUNT* reaches 0 or regeneration fails, raises
    ``pytest.UsageError`` with a manual-run command in the message.

    :param packages: Package names or module objects to check
        (e.g. ``["pycram"]`` or ``[import pycram; pycram]``).
    :raises TypeError: If any element is neither a string nor a ModuleType.
    :raises pytest.UsageError: On regeneration failure or exceeded retry limit.
    """
    import os
    import subprocess

    import pytest

    action, payload = _ensure_role_mixins_current(packages)
    if action is _ActionResult.RETURN:
        return
    if action is _ActionResult.ERROR:
        raise pytest.UsageError(payload)
    result = subprocess.run(
        [sys.executable, "-m", "pytest"] + sys.argv[1:],
        env=_get_env_with_pythonpath(),
    )
    os._exit(result.returncode)


# ── generation ────────────────────────────────────────────────────────────────


def generate_role_mixins_for_package(package: str | ModuleType, write: bool = True) -> None:
    """Scan *package* and generate (or preview) its role mixin stub files.

    Imports the package, collects every dataclass defined in it (recursively),
    builds a :class:`~krrood.class_diagrams.ClassDiagram`, and delegates to
    :func:`~krrood.patterns.role.helpers.transform_roles_in_class_diagram`.

    This function is safe to call from any context — it does **not** reload
    already-imported modules in the calling process because it is designed to
    run in a fresh process (e.g. a CI step or a dedicated pre-test script).

    :param package: Dotted package name or an already-imported module object
        (e.g. ``"pycram"`` or ``import pycram; pycram``).
    :param write: When *True* (default) write mixin and transformed source
        files to disk.  When *False* compute the content without writing
        (dry-run mode).
    :raises ImportError: If *package* is a string and cannot be imported.
    :raises TypeError: If *package* is neither a string nor a ModuleType.
    :raises Exception: Propagates any error raised during transformation.
    """
    package = _resolve_package(package)
    all_classes = [c for c in classes_of_package(package) if is_dataclass(c)]
    class_diagram = ClassDiagram(all_classes, introspector=AllFieldsIntrospector())
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
        krrood_logger.setLevel(logging.DEBUG)
        logging.getLogger(__name__).setLevel(logging.DEBUG)
    else:
        krrood_logger.setLevel(logging.INFO)
        logging.getLogger(__name__).setLevel(logging.INFO)

    log = logging.getLogger(__name__)

    if args.check:
        stale_by_pkg: dict[str, list[Path]] = {}
        for package_name in args.packages:
            try:
                stale = _stale_files_for_package(package_name)
                if stale:
                    stale_by_pkg[package_name] = stale
            except Exception as e:
                log.exception("  Failed to check %s, with error: %s", package_name, e)
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
        except Exception as e:
            log.exception("  Failed: %s, with error: %s", package_name, e)
            failed.append(package_name)

    if failed:
        log.error("The following packages failed: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
