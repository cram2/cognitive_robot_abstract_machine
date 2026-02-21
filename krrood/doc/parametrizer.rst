Parametrization of Objects
==========================

The ``Parameterizer`` class allows you to work with underspecified python objects.

The ``Parameterizer`` first converts the input object into a ``DataAccessObject`` (DAO) and then recursively traverses
its attributes and relationships to create a ``Parameterization``. Make sure that a corresponding DAO exists for the
objects you want to parameterize with.

To parameterize an underspecified object, you first need to create the structure of an underspecified object.
You can control which fields are included and whether they have a fixed value:

- **Ellipsis (``...``)**: Signals that the field should be parameterized as a variable. You should only set leaf types to Ellipsis.
- **Concrete Value**: If a field has a value (e.g., ``3.14``), it is parameterized as a variable, and the parameters will contain the assignment (e.g., ``{PositionDAO.y: 3.14}``).
- **None**: If a field is set to ``None``, it is completely ignored and will not be included in the parameterization. This is also true for relationships, so you can use this mechanism to skip parts of objects altogether.



.. code-block:: python

    from krrood.probabilistic_knowledge.parameterizer import Parameterizer
    from dataset.example_classes import Position, Orientation, Pose

    # Parameterize all fields of a Position
    obj = Pose(position=Position(x=..., y=3.14, z=None), orientation=None)
    parameterization = Parameterizer().parameterize(obj)
    print(parameterization)

