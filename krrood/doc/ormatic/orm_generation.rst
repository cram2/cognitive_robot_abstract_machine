.. ormatic_generation

Generating an ORMatic Interface
===============================

A typical workflow for generating an ORMatic interface consists of these two steps

Identify candidate classes
--------------------------

- Choose the dataclasses that represent your persistent domain.
- Optionally include explicit mapping classes
- Optionally provide custom type mappings

Create a script generating the interface
----------------------------------------

After identifying the persistent part of your domain, I recommend creating a script that generates the interface.
The script for generating the test ORM interface for KRROOD looks like this:

.. code-block:: python

    import os
    from dataclasses import is_dataclass

    import krrood.entity_query_language.orm.model
    import krrood.entity_query_language.symbol_graph
    from dataset.example_classes import (
        PhysicalObject,
        NotMappedParent,
        ChildNotMapped,
        ConceptType,
    )
    from dataset.semantic_world_like_classes import *
    from krrood.class_diagrams.class_diagram import ClassDiagram
    from krrood.entity_query_language.predicate import (
        HasTypes,
        HasType,
    )
    from krrood.entity_query_language.symbol_graph import SymbolGraph
    from krrood.ormatic.dao import AlternativeMapping
    from krrood.ormatic.ormatic import ORMatic
    from krrood.ormatic.utils import classes_of_module
    from krrood.utils import recursive_subclasses

    # build the symbol graph
    symbol_graph = SymbolGraph()

    # collect all classes that need persistence
    all_classes = {c.clazz for c in symbol_graph._class_diagram.wrapped_classes}
    all_classes |= {
        alternative_mapping.original_class()
        for alternative_mapping in recursive_subclasses(AlternativeMapping)
    }
    all_classes |= set(classes_of_module(krrood.entity_query_language.symbol_graph))
    all_classes |= {Symbol}

    # remove classes that don't need persistence
    all_classes -= {HasType, HasTypes, ContainsType}
    all_classes -= {NotMappedParent, ChildNotMapped}

    # only keep dataclasses
    all_classes = {c for c in all_classes if is_dataclass(c)}

    class_diagram = ClassDiagram(
        list(sorted(all_classes, key=lambda c: c.__name__, reverse=True))
    )

    instance = ORMatic(
        class_dependency_graph=class_diagram,
        type_mappings={
            PhysicalObject: ConceptType,
        },
        alternative_mappings=recursive_subclasses(AlternativeMapping),
    )

    instance.make_all_tables()

    file_path = os.path.join(os.path.dirname(__file__), "dataset", "ormatic_interface.py")

    with open(file_path, "w") as f:
        instance.to_sqlalchemy_file(f)


Extending an existing ORMatic interface (Optional)
--------------------------------------------------

If your application depends on a package that already has an ORMatic interface, you can extend it by doing the
following:

.. code-block:: python

    import dataset.ormatic_interface
    from dataset.dataset_extension import AggregatorOfExternalInstances, CustomPosition
    from krrood.class_diagrams import ClassDiagram
    from krrood.ormatic.ormatic import ORMatic
    from krrood.ormatic.utils import get_classes_of_ormatic_interface

    # import classes from the existing interface
    classes, alternative_mappings, type_mappings = get_classes_of_ormatic_interface(
        dataset.ormatic_interface
    )

    # specify new classes
    classes += [CustomPosition, AggregatorOfExternalInstances]

    # create the new ormatic interface
    class_diagram = ClassDiagram(
        list(sorted(classes, key=lambda c: c.__name__, reverse=True))
    )
    instance = ORMatic(
        class_diagram,
        type_mappings=type_mappings,
        alternative_mappings=alternative_mappings,
    )
    instance.make_all_tables()


    new_interface_file = "ormatic_interface.py"
    with open(new_interface_file, "w") as f:
        instance.to_sqlalchemy_file(f)
