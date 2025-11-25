.. interface_usage

Interface Usage
===============

After generating the interface for the ORM you can start adding and retrieving objects to the database.
Below is an example that used classes and objects from the test dataset provided in the testing package of KRROOD.

.. code-block:: python

   from sqlalchemy import select, create_engine
   from sqlalchemy.orm import Session

   from dataset.example_classes import *
   from dataset.sqlalchemy_interface import *
   from krrood.ormatic.dao import (
       to_dao
   )

   engine = create_engine("sqlite:///:memory:")
   Base.metadata.create_all(engine)
   session = Session(engine)

   k1 = KinematicChain("a")
   k2 = KinematicChain("b")
   torso = Torso("t", [k1, k2])
   torso_dao = TorsoDAO.to_dao(torso)

   session.add(torso_dao)
   session.commit()

   queried_torso = session.scalars(select(TorsoDAO)).one()
   assert queried_torso == torso_dao
