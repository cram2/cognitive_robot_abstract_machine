import importlib
import importlib.util
import sys
from krrood.class_diagrams import ClassDiagram
from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.utils import get_classes_of_ormatic_interface
from ..dataset.dataset_extension import AggregatorOfExternalInstances, CustomPosition
from ..dataset import ormatic_interface


def test_extension(tmp_path):
    """
    Test that existing ormatic interfaces can be extended
    """
    # import classes from the existing interface
    classes, alternative_mappings, type_mappings = get_classes_of_ormatic_interface(
        ormatic_interface
    )
    assert type_mappings == ormatic_interface.Base.type_mappings
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

    # write to tempfile
    new_interface_file = tmp_path / "ormatic_interface.py"
    with open(new_interface_file, "w") as f:
        instance.to_sqlalchemy_file(f)

    # Import the generated module
    spec = importlib.util.spec_from_file_location(
        "ormatic_interface", new_interface_file
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # make it discoverable during exec
    spec.loader.exec_module(module)

    new_classes, new_alternative_mappings, new_type_mappings = (
        get_classes_of_ormatic_interface(module)
    )

    assert set(cls.__name__ for cls in classes) == set(
        cls.__name__ for cls in new_classes
    )

    assert set(cls.__name__ for cls in alternative_mappings) == set(
        cls.__name__ for cls in new_alternative_mappings
    )
