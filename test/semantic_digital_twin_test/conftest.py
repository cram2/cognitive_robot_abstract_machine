from krrood.entity_query_language.symbol_graph import SymbolGraph

from semantic_digital_twin.world import World
import runpy
from pathlib import Path


def pytest_configure(config):
    # Ensure ORM classes are generated before tests run
    repo_root = Path(__file__).resolve().parents[2]
    generate_orm_path = repo_root / "semantic_digital_twin" / "scripts" / "generate_orm.py"
    # Execute the ORM generation script as a standalone module
    runpy.run_path(str(generate_orm_path), run_name="__main__")
    SymbolGraph()
