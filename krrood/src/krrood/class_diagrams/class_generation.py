from dataclasses import dataclass
from typing import List


@dataclass
class DataclassDescription:
    class_name: str
    bases: List[str]
    attributes: List[str]


class ModuleDescription:
    classes: List[DataclassDescription]
