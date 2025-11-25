Using EQL With ORMatic
==============================================

ORMatic can translate high-level EQL expressions into SQLAlchemy `select(...) <https://docs.sqlalchemy.org/en/20/orm/queryguide/select.html#writing-select-statements-for-orm-mapped-classes>`_ statements.

- Build expressions over variables that carry the underlying Python type.
- Use :py:func:`krrood.ormatic.eql_interface.eql_to_sql` to get a translator you can evaluate against your session.

Example:

.. code-block:: python

   from sqlalchemy import create_engine
   from sqlalchemy.orm import Session

   from entity_query_language.entity import let
   from entity_query_language import or_, in_

   from dataset.example_classes import Position
   from dataset.sqlalchemy_interface import Base, PositionDAO

   from krrood.ormatic.eql_interface import eql_to_sql

   engine = create_engine('sqlite:///:memory:')
   session = Session(engine)
   Base.metadata.create_all(engine)

   # Insert sample data
   session.add_all([
       PositionDAO.to_dao(Position(1, 2, 3)),
       PositionDAO.to_dao(Position(1, 2, 4)),
       PositionDAO.to_dao(Position(2, 9, 10)),
   ])
   session.commit()

   # Build an EQL expression
   position = let(type_=Position)  # domain is irrelevant for translation
   expr = position.z > 3

   # Translate and execute
   translator = eql_to_sql(expr, session)
   rows = translator.evaluate()  # → PositionDAO rows with z > 3

   # More complex logic
   expr2 = or_(position.z == 4, position.x == 2)
   translator2 = eql_to_sql(expr2, session)
   rows2 = translator2.evaluate()

   # Membership
   expr3 = in_(position.x, [1, 7])
   translator3 = eql_to_sql(expr3, session)
   rows3 = translator3.evaluate()

Notes on the EQL translator:

- Variable resolution uses :py:func:`krrood.ormatic.dao.get_dao_class` to map EQL variables to DAO classes.
- Attribute comparisons on a single table are supported, including ``==``, ``!=``, ``>``, ``>=``, ``<``, ``<=``, ``in``, logical ``and``/``or``, and ``not``.
- If a variable’s DAO class cannot be found, a specific EQL translation error is raised to help diagnose missing mappings.
- EQL translation is not complete; feel free to extend it.
