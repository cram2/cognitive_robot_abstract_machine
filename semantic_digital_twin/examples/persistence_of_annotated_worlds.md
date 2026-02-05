---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(persistence-of-annotated-worlds)=
# Persistence of annotated worlds

The semantic digital twin comes with an ORM attached to it that is derived from the python datastructures.
The ORM can be used to serialize entire worlds into an SQL database and retrieve them later. The semantic annotations are stored alongside the kinematic information.
The queried worlds are full objects that can be reconstructed into the original objects without any problems.
The resulting SQL databases are perfect entry points for machine learning.

Concepts used:
- [](loading-worlds)
- [ORMatic](https://github.com/tomsch420/ormatic)

Let's go into an example where we create a world, store it, retrieve and reconstruct it.

First, let's load a world from a URDF file.

```{code-cell} ipython3
import logging
import os
from pkg_resources import resource_filename

from sqlalchemy import select
from sqlalchemy.orm import Session

from krrood.ormatic.utils import create_engine
from krrood.ormatic.dao import to_dao

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.orm.ormatic_interface import *
from semantic_digital_twin.semantic_annotations.semantic_annotations import Table


logging.disable(logging.CRITICAL)
# set up an in memory database
engine = create_engine('sqlite:///:memory:')
session = Session(engine)
Base.metadata.create_all(bind=session.bind)

# load the table world from urdf
urdf_dir = os.path.join(resource_filename("semantic_digital_twin", "../../"), "resources", "urdf")
table = os.path.join(urdf_dir, "table.urdf")
world = URDFParser.from_file(table).parse()
```

Next, we create a semantic annotation that describes the table.

```{code-cell} ipython3
table_semantic_annotation = Table(root=[b for b in world.bodies if "top" in str(b.name)][0])
with world.modify_world():
    world.add_semantic_annotation(table_semantic_annotation)
print(table_semantic_annotation)
```

Now, let's store the world to a database. For that, we need to convert it to its data access object which than can be stored in the database.

```{code-cell} ipython3
dao = to_dao(world)
session.add(dao)
session.commit()
```

We can now query the database about the world and reconstruct it to the original instance. As you can see the semantic annotations are also available and fully working.

```{code-cell} ipython3
queried_world = session.scalars(select(WorldMappingDAO)).one()
reconstructed_world = queried_world.from_dao()
table = [semantic_annotation for semantic_annotation in reconstructed_world.semantic_annotations if isinstance(semantic_annotation, Table)][0]
print(table)
print(table.points_on_supporting_surface(2))
```

## Maintaining the ORM 🧰

You can maintain the ORM by maintaining the [generate_orm.py](https://github.com/cram2/semantic_digital_twin/blob/main/scripts/generate_orm.py).
In there you have to list all the classes you want to generate mappings for and perhaps some type decorators for advanced use cases.
Whenever you write a new dataclass that should appear or has semantic meaningful content make sure it appears in the set of classes.
Pay attention to the logger during generation and see if it understands your datastructures correctly.


## The sharp bits 🔪
The world class manages the dependencies of the bodies in the world. Whenever you retrieve a body or connection, it comes as a data access object that is disconnected from the world itself.
The relationships to the world exist and can be joined. However, when you reconstruct something else but the world, the reconstructed object does not have a world available. You can always reconstruct the entire world by querying for the objects world instead.


## Accessing a permanent database

This tutorial used an in memory database for the purpose of demonstration.
If you want to permanently store worlds, you have to
- Install an RDBMS that is supported by SQLAlchemy. (I recommend [PostgreSQL](https://www.postgresql.org/download/))
- Create a user and database in your RDBMS, for instance with [this script](https://github.com/cram2/cognitive_robot_abstract_machine/blob/main/semantic_digital_twin/scripts/create_postgres_database_and_user_if_not_exists.sql). The script contains the documentation on how to run itself.
- Set the environment variable `SEMANTIC_DIGITAL_TWIN_DATABASE_URI` to the connection string of your RDBMS, for instance by adding `export SEMANTIC_DIGITAL_TWIN_DATABASE_URI=postgresql://semantic_digital_twin:a_very_strong_password_here@localhost:5432/semantic_digital_twin` to your bashrc.
- Create a session for database interaction, for instance with `semantic_digital_twin_sessionmaker()()`