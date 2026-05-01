import os

from sqlalchemy.orm import sessionmaker

from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine, drop_database
from semantic_digital_twin.adapters.sage_10k_dataset.loader import Sage10kDatasetLoader
from semantic_digital_twin.adapters.sage_10k_dataset.processing import (
    create_pr2_in_world,
)
from semantic_digital_twin.adapters.sage_10k_dataset.semantic_annotations import (
    Sage10kNonShittyScenes,
)
from semantic_digital_twin.orm.ormatic_interface import *

engine = create_engine(os.getenv("SAGE10k_DATABASE_URI"))
drop_database(engine)
Base.metadata.create_all(engine)
session = sessionmaker(engine)()

loader = Sage10kDatasetLoader()
scene = loader.create_scene(Sage10kNonShittyScenes.GYM)
world = scene.create_world()

pr2 = create_pr2_in_world(world)


dao = to_dao(world)

session.add(dao)
session.commit()
print(f"Added world to database with database_id: {dao.database_id}")
