"""
This module measures the maintainability of ORMatic's impact on the CRAM
repository by analyzing lines of code and maintainability indices of developer-
authored files versus auto-generated files. It computes various metrics to
understand the reduction of manual maintenance effort.
"""

import ast
import pathlib
from dataclasses import dataclass
from typing import List

from radon.metrics import mi_rank, mi_visit

from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    TypstRenderer,
)

_REPO_ROOT = pathlib.Path(__file__).parents[4]
_coraplex_INTERFACE = (
    _REPO_ROOT / "coraplex" / "src" / "coraplex" / "orm" / "ormatic_interface.py"
)
_CRAM_COMPONENTS = ("coraplex", "semantic_digital_twin")


def _count_loc(path: pathlib.Path) -> int:
    """Count non-blank lines in a Python source file."""
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _maintainability_index(source: str) -> float:
    """Return the radon Maintainability Index (0–100) for *source*."""
    return round(mi_visit(source, multi=True), 1)


def _count_alternative_mappings(model_path: pathlib.Path) -> int:
    """Count AlternativeMapping subclasses defined in *model_path*."""
    if not model_path.exists():
        return 0
    tree = ast.parse(model_path.read_text(encoding="utf-8"))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and any("AlternativeMapping" in ast.unparse(b) for b in node.bases)
    )


def _count_mapped_classes(interface_path: pathlib.Path) -> int:
    """
    Count DataAccessObject classes in *interface_path*.

    Excludes the ``DeclarativeBase`` subclass and association-table helpers
    (those inheriting from ``AssociationDataAccessObject``).
    """
    tree = ast.parse(interface_path.read_text(encoding="utf-8"))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and not any("DeclarativeBase" in ast.unparse(b) for b in node.bases)
        and not any("AssociationDataAccessObject" in ast.unparse(b) for b in node.bases)
    )


@dataclass
class MaintainabilityResult(ExperimentResult):
    """
    Single-row summary of ORMatic's maintainability impact on CRAM.

    Compares the developer-maintained model.py files (coraplex +
    semantic_digital_twin) against the generated ormatic_interface.py.
    """

    maintainability_index: float
    """
    radon MI score (0–100) computed on the concatenated coraplex and
    semantic_digital_twin model.py sources.  Higher is more maintainable.
    """

    lines_of_code: int
    """
    Non-blank LOC across the coraplex and semantic_digital_twin model.py files —
    the code a developer authors and maintains when using ORMatic.
    """

    generated_lines_of_code: int
    """
    Non-blank LOC of coraplex's ormatic_interface.py — the SQLAlchemy mapping
    code ORMatic generates automatically, which would otherwise need manual
    maintenance.
    """

    mapped_classes: int
    """
    Number of DataAccessObject classes in the generated interface (excludes
    the DeclarativeBase root and association-table helpers).
    """

    alternative_mappings: int
    """
    Total AlternativeMapping subclasses across coraplex's and
    semantic_digital_twin's model.py files — the manual mapping effort that
    ORMatic cannot automate.
    """

    loc_ratio: float
    """Ratio of generated LOC to maintained LOC (generated / maintained)."""

    maintainability_rank: str
    """radon MI rank: A (>19), B (9–19), or C (≤9)."""


def _find_cram_model_files(repo_root: pathlib.Path) -> List[pathlib.Path]:
    """Return ``*/orm/model.py`` files for coraplex and semantic_digital_twin only."""
    return sorted(
        p
        for p in repo_root.rglob("orm/model.py")
        if "__pycache__" not in p.parts
        and "test" not in p.parts
        and ".git" not in p.parts
        and p.relative_to(repo_root).parts[0] in _CRAM_COMPONENTS
    )


def measure_maintainability(
    repo_root: pathlib.Path = _REPO_ROOT,
    cram_interface: pathlib.Path = _coraplex_INTERFACE,
) -> MaintainabilityResult:
    """
    Measure ORMatic's maintainability impact on the CRAM repository.

    Scans the coraplex and semantic_digital_twin model.py files (the
    developer-maintained ORMatic input) and compares them against coraplex's
    ormatic_interface.py (the generated output that would otherwise require
    manual maintenance).

    :param repo_root: Root of the CRAM repository to scan.
    :param cram_interface: Path to the generated CRAM ormatic_interface.py.
    :return: A single result summarising the maintainability comparison.
    """
    model_files = _find_cram_model_files(repo_root)
    combined_source = "\n".join(f.read_text(encoding="utf-8") for f in model_files)

    loc = sum(_count_loc(f) for f in model_files)
    generated_loc = _count_loc(cram_interface)

    mi = _maintainability_index(combined_source)
    return MaintainabilityResult(
        maintainability_index=mi,
        maintainability_rank=mi_rank(mi),
        lines_of_code=loc,
        generated_lines_of_code=generated_loc,
        mapped_classes=_count_mapped_classes(cram_interface),
        alternative_mappings=sum(_count_alternative_mappings(f) for f in model_files),
        loc_ratio=round(generated_loc / loc, 2),
    )


def main():
    result = measure_maintainability()
    print(TypstRenderer(ExperimentsTable([result])).render_table())
    print()
    ratio = round(result.generated_lines_of_code / result.lines_of_code, 1)
    print(
        f"ORMatic reduces ORM maintenance from {result.generated_lines_of_code} LOC "
        f"to {result.lines_of_code} LOC — a {ratio}x reduction."
    )


if __name__ == "__main__":
    main()
