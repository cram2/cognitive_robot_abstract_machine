import sys
from pathlib import Path
import runpy

from krrood.class_diagrams import ClassDiagram
from krrood.symbol_graph.symbol_graph import SymbolGraph, Symbol
from krrood.ontomatic.property_descriptor.attribute_introspector import (
    DescriptorAwareIntrospector,
)
from krrood.utils import recursive_subclasses

from semantic_digital_twin.world import World


def pytest_configure(config):
    # Ensure ORM classes are generated before tests run
    repo_root = Path(__file__).resolve().parents[2]
    generate_orm_path = (
        repo_root / "semantic_digital_twin" / "scripts" / "generate_orm.py"
    )
    runpy.run_path(str(generate_orm_path), run_name="__main__")

    # Build the symbol graph
    SymbolGraph.clear()
    class_diagram = ClassDiagram(
        recursive_subclasses(Symbol) + [World],
        introspector=DescriptorAwareIntrospector(),
    )
    SymbolGraph(_class_diagram=class_diagram)

    # Ensure role mixin files for semantic_digital_twin are semantically current.
    # If any are missing or outdated, regenerates them in a subprocess and
    # restarts pytest automatically (up to KRROOD_PYTEST_RERUN_COUNT times, default 2).
    from krrood.generate_role_mixins import ensure_role_mixins_current_for_pytest

    ensure_role_mixins_current_for_pytest(["semantic_digital_twin"])
