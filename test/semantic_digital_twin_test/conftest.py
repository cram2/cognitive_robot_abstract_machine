import os

import pytest
import sqlalchemy

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.symbol_graph.symbol_graph import SymbolGraph, Symbol
from krrood.ontomatic.property_descriptor.attribute_introspector import (
    DescriptorAwareIntrospector,
)
from krrood.ormatic.utils import create_engine, drop_database
from krrood.utils import recursive_subclasses
from semantic_digital_twin.adapters.urdf import URDFParser

from semantic_digital_twin.world import World
import runpy
from pathlib import Path


def pytest_configure(config):

    worker = os.environ.get("PYTEST_XDIST_WORKER")

    if not worker:
        # Ensure ORM classes are generated before tests run
        repo_root = Path(__file__).resolve().parents[2]
        generate_orm_path = (
            repo_root / "semantic_digital_twin" / "scripts" / "generate_orm.py"
        )
        # Execute the ORM generation script as a standalone module
        runpy.run_path(str(generate_orm_path), run_name="__main__")
    # Build the symbol graph
    SymbolGraph.clear()
    class_diagram = ClassDiagram(
        recursive_subclasses(Symbol) + [World],
        introspector=DescriptorAwareIntrospector(),
    )
    SymbolGraph(_class_diagram=class_diagram)


@pytest.fixture
def in_memory_session_maker():
    """
    A session maker for an empty database that several sessions can share.

    ``uri=true`` belongs into the query string: the pysqlite dialect reads it from there,
    and without it sqlite opens a file named ``file::memory:`` in the working directory
    instead of a shared in-memory database, which then outlives the test.
    """
    from semantic_digital_twin.orm.ormatic_interface import Base

    engine = create_engine(
        "sqlite+pysqlite:///file::memory:?cache=shared&uri=true",
        connect_args={"check_same_thread": False},
    )
    drop_database(engine)
    Base.metadata.create_all(engine)
    yield sqlalchemy.orm.sessionmaker(bind=engine)
    engine.dispose()


@pytest.fixture
def table_world():
    urdf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "semantic_digital_twin",
        "resources",
        "urdf",
    )
    table_path = os.path.join(urdf_dir, "table.urdf")

    return URDFParser.from_file(file_path=table_path).parse()
