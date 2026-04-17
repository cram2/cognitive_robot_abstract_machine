import logging
import os
import sys

import pycram
from ormatic.ormatic import logger, ORMatic
from ormatic.utils import classes_of_module, recursive_subclasses, ORMaticExplicitMapping
from pycram.datastructures import pose
from pycram.datastructures.dataclasses import FrozenObject, RayResult, MultiverseRayResult, MultiverseContactPoint, \
    ReasoningResult, \
    MultiverseMetaData, VirtualMobileBaseJoints, Rotations, TextAnnotation, VirtualJoint, ContactPointsList, \
    ClosestPointsList, State, CollisionCallbacks, MultiBody, Colors, ManipulatorData
from sqlacodegen.generators import TablesGenerator
from sqlalchemy import create_engine
from sqlalchemy.orm import registry, Session

from segmind.datastructures import events, mixins
from segmind.datastructures.events import InsertionEvent
from segmind.detectors.atomic_event_detectors import AtomicEventDetector

# ----------------------------------------------------------------------------------------------------------------------
# This script generates the ORM classes for the segmind package.
# Dataclasses can be mapped automatically to the ORM model
# using the ORMatic library, they just have to be registered in the classes list.
# Classes that are self_mapped and explicitly_mapped are already mapped in the model.py file. Look there for more
# information on how to map them.
# ----------------------------------------------------------------------------------------------------------------------

# create set of classes that should be mapped
classes = set()
classes |= set(recursive_subclasses(ORMaticExplicitMapping))
classes |= set(classes_of_module(events)) - {InsertionEvent}
# classes |= set(classes_of_module(mixins))
pycram_dataclasses = set(classes_of_module(pycram.datastructures.dataclasses))
pycram_dataclasses -= {RayResult, MultiverseRayResult, MultiverseContactPoint, ReasoningResult, MultiverseMetaData,
                       VirtualMobileBaseJoints, Rotations, TextAnnotation, VirtualJoint,
                       MultiBody, CollisionCallbacks, Colors, ManipulatorData}
pycram_dataclasses -= set(recursive_subclasses(State)) | {State}
classes |= pycram_dataclasses
classes |= set(classes_of_module(pose))
classes -= set(recursive_subclasses(AtomicEventDetector)) | {AtomicEventDetector}

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

    path = os.path.abspath(os.path.join(os.getcwd(), '../src/segmind/orm/'))
    with open(os.path.join(path, 'ormatic_interface.py'), 'w') as f:
        ormatic.to_python_file(generator, f)


if __name__ == '__main__':
    generate_orm()
