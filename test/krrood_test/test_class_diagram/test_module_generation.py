from dataclasses import make_dataclass

from krrood.class_diagrams.module_generation import (
    DataclassDescription,
    ModuleDescription,
)
from krrood.symbol_graph.symbol_graph import Symbol
from krrood.utils import module_and_class_name


def test_module_generation():

    in_memory_class = make_dataclass(
        cls_name="InMemoryClass", bases=(Symbol,), fields=[]
    )

    module_and_class_name(in_memory_class)

    in_memory_child_class = make_dataclass(
        cls_name="InMemoryChildClass", bases=(in_memory_class,), fields=[]
    )

    description = ModuleDescription.from_dataclasses(
        [in_memory_class, in_memory_child_class]
    )
    print(description)
