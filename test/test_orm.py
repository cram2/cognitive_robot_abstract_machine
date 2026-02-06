import logging
import os
import sys
from dataclasses import dataclass
from os.path import dirname
from typing import Optional

from ormatic.ormatic import logger, ORMatic
from ormatic.utils import recursive_subclasses
from sqlacodegen.generators import TablesGenerator
from sqlalchemy import create_engine
from sqlalchemy.orm import registry, Session

# ----------------------------------------------------------------------------------------------------------------------
# This script generates the ORM classes for the segmind package.
# Dataclasses can be mapped automatically to the ORM model
# using the ORMatic library, they just have to be registered in the classes list.
# Classes that are self_mapped and explicitly_mapped are already mapped in the model.py file. Look there for more
# information on how to map them.
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class ParentMappedClass:
    a: int
    b: Optional[int] = None

@dataclass
class ChildMappedClass(ParentMappedClass):
    c: Optional[int] = None

@dataclass
class ChildNotMappedClass(ParentMappedClass):
    d: Optional[int] = None


# create set of classes that should be mapped
classes = set()
classes |= set(recursive_subclasses(ORMatic))
classes |= {ParentMappedClass, ChildMappedClass}


def generate_orm():
    """
    Generate the ORM classes for the pycram package.
    """
    # Set up logging
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    mapper_registry = registry()
    engine = create_engine('sqlite:///:memory:')
    session = Session(engine)

    # Create an ORMatic object with the classes to be mapped
    ormatic = ORMatic(list(classes), mapper_registry)

    # Generate the ORM classes
    ormatic.make_all_tables()

    # Create the tables in the database
    mapper_registry.metadata.create_all(session.bind)

    # Write the generated code to a file
    generator = TablesGenerator(mapper_registry.metadata, session.bind, [])

    with open(os.path.join(dirname(__file__), 'ormatic_interface.py'), 'w') as f:
        ormatic.to_python_file(generator, f)


def test_generate_orm():
    # This will Succeed
    child_not_mapped = ChildNotMappedClass(a=1, b=2 , d=3)
    assert child_not_mapped.a == 1
    assert child_not_mapped.b == 2
    assert child_not_mapped.d == 3
    generate_orm()
    child_mapped = ChildMappedClass(a=1, b=2, c=3)
    assert child_mapped.a == 1
    assert child_mapped.b == 2
    assert child_mapped.c == 3
    # This will Fail
    child_not_mapped = ChildNotMappedClass(a=1, b=2, d=3)
    assert child_not_mapped.a == 1
    assert child_not_mapped.b == 2
    assert child_not_mapped.d == 3
