"""
Fixtures shared by the semantic_digital_twin tests: worlds, a mesh and database
sessions.
"""

from __future__ import annotations

# %% ORM interfaces

# Built before the imports below, which read a mapped datastructure: pytest imports every
# conftest of a run before calling any hook, so a hook would fire too late. The build runs
# once per process and never on an xdist worker.
from cognitive_robot_abstract_machine.orm_interfaces import OrmPackage

from ..orm_interface_build import regenerate_orm_interfaces

regenerate_orm_interfaces(OrmPackage.SEMANTIC_DIGITAL_TWIN)


import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np
import pytest
import sqlalchemy
import trimesh
from typing_extensions import Callable  # noqa: E402

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.symbol_graph.symbol_graph import SymbolGraph, Symbol
from krrood.ontomatic.property_descriptor.attribute_introspector import (
    DescriptorAwareIntrospector,
)
from krrood.ormatic.utils import create_engine, drop_database
from krrood.utils import recursive_subclasses
from semantic_digital_twin.adapters.package_resolver import (  # noqa: E402
    PackageUriResolver,
)
from semantic_digital_twin.adapters.urdf import URDFParser

from semantic_digital_twin.world import World


def pytest_configure(config):
    # Build the symbol graph
    SymbolGraph.clear()
    class_diagram = ClassDiagram(
        classes=recursive_subclasses(Symbol) + [World],
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
    """
    :return: A world holding the table of the iai_objects package.
    """
    return URDFParser.from_file(
        file_path="package://iai_objects/urdf/table.urdf"
    ).create_world()


# %% flat meshes


@pytest.fixture
def millimetre_wall_strip() -> trimesh.Trimesh:
    """
    A flat, zero-volume quad one millimetre wide and 2.8 metres tall, as architectural
    USD scenes contain, in which V-HACD finds no convex parts.
    """
    return trimesh.Trimesh(
        vertices=np.array(
            [
                [0.001, 0.0, 2.8],
                [0.0, 0.0, 2.8],
                [0.001, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        faces=np.array([[2, 1, 0], [1, 2, 3]]),
        process=False,
    )


# %% the ORM mappers the tests use


@pytest.fixture(scope="session", autouse=True)
def register_orm_mappers() -> None:
    """
    Load semantic_digital_twin's ORM interface, whose import registers the mappers of it
    and the interfaces it builds on for every test.
    """
    import semantic_digital_twin.orm.ormatic_interface  # noqa: F401


# %% ROS packages holding mesh files


@pytest.fixture
def mesh_package_installer(monkeypatch) -> Callable[[Path], Path]:
    """
    :return: A function that installs the package ``mesh_package``, holding a copy of
        the jeroen cup, into the ament prefix it is given, makes that prefix the only
        one searched and returns the installed mesh file.
    """
    jeroen_cup = Path(
        PackageUriResolver().resolve(uri="package://iai_objects/objects/jeroen_cup.stl")
    )

    def install(prefix: Path) -> Path:
        """
        Installs ``mesh_package`` into ``prefix``.

        :return: The installed mesh file.
        """
        share = prefix / "share"
        package_index = share / "ament_index" / "resource_index" / "packages"
        package_index.mkdir(parents=True)
        (package_index / "mesh_package").touch()
        mesh_file = share / "mesh_package" / "meshes" / jeroen_cup.name
        mesh_file.parent.mkdir(parents=True)
        shutil.copy(jeroen_cup, mesh_file)
        monkeypatch.setenv("AMENT_PREFIX_PATH", str(prefix))
        return mesh_file

    return install


@pytest.fixture
def installed_mesh_file(mesh_package_installer, tmp_path) -> Path:
    """
    :return: A mesh file in the share directory of the package ``mesh_package``, which
        is registered in an ament prefix.
    """
    return mesh_package_installer(tmp_path)
