from dataclasses import make_dataclass
from pathlib import Path

from krrood.class_diagrams.module_generation import (
    DataclassRenderer,
    ModuleRenderer,
)
from krrood.symbol_graph.symbol_graph import Symbol
from krrood.utils import module_and_class_name


def test_module_generation():

    in_memory_class = make_dataclass(
        cls_name="InMemoryClass", bases=(Symbol,), fields=[]
    )

    in_memory_child_class = make_dataclass(
        cls_name="InMemoryChildClass", bases=(in_memory_class,), fields=[]
    )

    description = ModuleRenderer.from_dataclasses(
        [in_memory_class, in_memory_child_class]
    )

    # generate a file in the dataset containing the parsing result
    path = Path(__file__).parent.parent / "dataset" / "generated_module.py"
    description.write_to_file(path)

    from ..dataset.generated_module import InMemoryClass, InMemoryChildClass

    assert issubclass(InMemoryChildClass, InMemoryClass)
