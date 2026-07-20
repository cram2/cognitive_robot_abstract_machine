import logging
import os
import tempfile
from dataclasses import is_dataclass

import pytest
from sqlalchemy.orm import configure_mappers, sessionmaker

import krrood.entity_query_language.orm.model
import krrood.symbol_graph.symbol_graph
from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.entity_query_language.predicate import (
    HasTypes,
    HasType,
)
from krrood.ormatic.data_access_objects.alternative_mappings import *  # type: ignore
from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.type_dict import TypeDict
from krrood.ormatic.utils import classes_of_module, create_engine
from krrood.ormatic.utils import drop_database
from krrood.patterns.role import Role
from krrood.symbol_graph.symbol_graph import SymbolGraph
from krrood.utils import recursive_subclasses
from .dataset import (
    example_classes,
    semantic_world_like_classes,
    alternative_mappings_construction_order,
)
from .dataset.example_classes import (
    KRROODPhysicalObject,
    NotMappedParent,
    ChildNotMapped,
    ConceptType,
    JSONSerializableClass,
)
from .dataset.role_and_ontology import (
    university_ontology_like_classes_without_descriptors,
    role_takers_in_another_module,
    classes_for_testing_role_recursion_error,
)
from .dataset.semantic_world_like_classes import *
from .test_eql.conf.world.doors_and_drawers import DoorsAndDrawersWorld
from .test_eql.conf.world.handles_and_containers import (
    HandlesAndContainersWorld,
    InferredCabinetsWorld,
)


def generate_sqlalchemy_interface():
    """
    Generate the SQLAlchemy interface file before tests run.

    This ensures the file exists before any imports attempt to use it, solving
    krrood_test isolation issues when running all tests.
    """
    # build the symbol graph
    SymbolGraph.clear()
    symbol_graph = SymbolGraph(packages=["krrood", "test.krrood"])

    # collect all classes that need persistence
    all_classes = {c.clazz for c in symbol_graph._class_diagram.wrapped_classes}
    all_classes |= {
        alternative_mapping.original_class()
        for alternative_mapping in recursive_subclasses(AlternativeMapping)
    }
    all_classes |= set(classes_of_module(krrood.symbol_graph.symbol_graph))
    all_classes |= set(classes_of_module(example_classes))
    all_classes |= set(classes_of_module(semantic_world_like_classes))
    all_classes |= set(
        classes_of_module(university_ontology_like_classes_without_descriptors)
    )
    all_classes |= set(classes_of_module(role_takers_in_another_module))
    all_classes |= set(classes_of_module(classes_for_testing_role_recursion_error))
    all_classes |= set(classes_of_module(alternative_mappings_construction_order))
    all_classes |= {Symbol, Role}

    # remove classes that don't need persistence
    all_classes -= {HasType, HasTypes, ContainsType}
    all_classes -= {NotMappedParent, ChildNotMapped, JSONSerializableClass}

    # only keep dataclasses
    all_classes = {
        c
        for c in all_classes
        if is_dataclass(c) and not issubclass(c, AlternativeMapping)
    }

    all_classes |= {FunctionType}
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("krrood").setLevel(logging.DEBUG)

    class_diagram = ClassDiagram(
        list(sorted(all_classes, key=lambda c: c.__name__, reverse=True))
    )

    instance = ORMatic(
        class_dependency_graph=class_diagram,
        type_mappings=TypeDict({KRROODPhysicalObject: ConceptType}),
        alternative_mappings=recursive_subclasses(AlternativeMapping),
    )

    instance.make_all_tables()

    file_path = os.path.join(
        os.path.dirname(__file__), "dataset", "ormatic_interface.py"
    )

    # Write atomically: under pytest-xdist every worker regenerates this shared file at import
    # time, so a plain truncating write lets one worker read a half-written (or empty) file while
    # another is still writing it, which surfaces as "cannot import name ..." at collection and
    # "different tests collected between gw0 and gw1". Writing to a temporary file in the same
    # directory and os.replace-ing it into place means every reader always sees a complete file.
    directory = os.path.dirname(file_path)
    with tempfile.NamedTemporaryFile(
        "w", dir=directory, prefix="ormatic_interface.", suffix=".py.tmp", delete=False
    ) as temporary_file:
        instance.to_sqlalchemy_file(temporary_file)
        temporary_path = temporary_file.name
    os.replace(temporary_path, file_path)

    return instance


def pytest_configure(config):
    """
    Set log levels before krrood_test collection.
    """
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)


# Generate ormatic_interface.py at module level, before the star import below.
# This must happen here (not in a pytest hook) because hooks run after the
# conftest module is fully imported, which means the import on the next line
# would fail if the generated file is stale or missing.
generate_sqlalchemy_interface()

try:
    from .dataset.ormatic_interface import *
except ImportError:
    pass


@pytest.fixture
def handles_and_containers_world() -> World:
    world = HandlesAndContainersWorld().create()
    return world


@pytest.fixture
def inferred_cabinets_world() -> World:
    world = InferredCabinetsWorld().create()
    return world


@pytest.fixture
def doors_and_drawers_world() -> World:
    world = DoorsAndDrawersWorld().create()
    SymbolGraph()
    return world


@pytest.fixture(autouse=True)
def cleanup_after_test():
    # Rebuild a clean, package-scoped symbol graph before each test. Scoping to the same
    # packages as generate_sqlalchemy_interface keeps Symbol subclasses imported from other
    # packages (e.g. coraplex / semantic_digital_twin pulled in during collection) out of the
    # class diagram, so they cannot leak in and corrupt type resolution for later tests.
    SymbolGraph.clear()
    SymbolGraph(packages=["krrood", "test.krrood"])
    yield
    SymbolGraph().clear()


@pytest.fixture(scope="session")
def engine():
    configure_mappers()
    engine = create_engine("sqlite:///:memory:")
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def session(engine):
    session_maker = sessionmaker(engine)
    session = session_maker()
    yield session
    session.close()


@pytest.fixture
def database(engine, session):
    Base.metadata.create_all(engine)
    yield
    drop_database(engine)
