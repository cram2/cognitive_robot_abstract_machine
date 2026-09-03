import importlib
import importlib.util
import sys
from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.helper import (
    get_classes_of_ormatic_interface,
    OrmaticInterfaceInformation,
)
from ..dataset.dataset_extension import AggregatorOfExternalInstances, CustomPosition
from ..dataset import ormatic_interface


def test_extension(tmp_path):
    """
    Test that existing ormatic interfaces can be extended.
    """
    # import classes from the existing interface
    interface_info = get_classes_of_ormatic_interface(ormatic_interface)
    classes = interface_info.classes
    alternative_mappings = interface_info.alternative_mappings
    type_mappings = interface_info.type_mappings
    assert type_mappings == ormatic_interface.Base.type_mappings
    # specify new classes; the pre-existing ones stay externally mapped, not regenerated
    new_domain_classes = [CustomPosition, AggregatorOfExternalInstances]
    classes += new_domain_classes

    # create the new ormatic interface
    class_diagram = ClassDiagram(
        list(sorted(classes, key=lambda c: c.__name__, reverse=True))
    )
    instance = ORMatic(
        class_diagram,
        interface_information=OrmaticInterfaceInformation(
            type_mappings=type_mappings,
            alternative_mappings=alternative_mappings,
            externally_mapped_classes=interface_info.externally_mapped_classes,
        ),
    )
    instance.make_all_tables()

    # write to tempfile
    new_interface_file = tmp_path / "ormatic_interface.py"
    with open(new_interface_file, "w") as f:
        instance.to_sqlalchemy_file(f)

    # Import the generated module
    spec = importlib.util.spec_from_file_location(
        "extended_ormatic_interface", new_interface_file
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # make it discoverable during exec
    spec.loader.exec_module(module)

    new_interface_info = get_classes_of_ormatic_interface(module)

    # only the genuinely new classes are (re)declared in the extended interface
    assert set(cls.__name__ for cls in new_interface_info.classes) == set(
        cls.__name__ for cls in new_domain_classes
    )

    # everything the base interface already mapped is still reachable, just reused
    assert set(cls.__name__ for cls in classes).issubset(
        {cls.__name__ for cls in new_interface_info.externally_mapped_classes}
    )
    assert set(cls.__name__ for cls in alternative_mappings).issubset(
        {cls.__name__ for cls in new_interface_info.externally_mapped_classes}
    )
