import os
import time

from sqlalchemy.orm import sessionmaker

from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine, drop_database
from semantic_digital_twin.adapters.sage_10k_dataset.loader import Sage10kDatasetLoader
from semantic_digital_twin.adapters.sage_10k_dataset.processing import (
    create_hsrb_in_world,
)
from semantic_digital_twin.adapters.sage_10k_dataset.semantic_annotations import (
    Sage10kNonShittyScenes,
)
from semantic_digital_twin.orm.ormatic_interface import *

current_time = time.time()
print("creating database")
engine = create_engine(os.getenv("SAGE10k_DATABASE_URI"))
drop_database(engine)
Base.metadata.create_all(engine)
session = sessionmaker(engine)()
print(f"creating the database took {time.time() - current_time:.2f} seconds")

current_time = time.time()
print("loading scene")
loader = Sage10kDatasetLoader()
scene = loader.create_scene(Sage10kNonShittyScenes.TV_STUDIO)
world = scene.create_world()
print(f"Loading the scene took {time.time() - current_time:.2f} seconds")

current_time = time.time()
print("loading robot")
pr2 = create_hsrb_in_world(world)
print(f"Loading the robot took {time.time() - current_time:.2f} seconds")

current_time = time.time()
print("saving to database")
dao = to_dao(world)
session.add(dao)
session.commit()
print(f"Saving to database took {time.time() - current_time:.2f} seconds")
print(f"Added world to database with database_id: {dao.database_id}")
