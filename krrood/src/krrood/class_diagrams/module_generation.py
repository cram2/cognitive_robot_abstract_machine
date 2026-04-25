from dataclasses import dataclass, fields
from typing import List, Set, Type, Optional


@dataclass
class FieldDescription:
    name: str
    type: str


@dataclass
class DataclassDescription:
    name: str
    bases: List[str]
    fields: List[FieldDescription]

    @classmethod
    def from_dataclass(cls, clazz: Type):
        return cls(
            name=clazz.__name__,
            bases=[base.__name__ for base in clazz.__bases__],
            fields=[
                FieldDescription(name=field.name, type=field.type)
                for field in fields(clazz)
            ],
        )


@dataclass
class ModuleDescription:
    imports: Set[str]
    classes: List[DataclassDescription]

    @classmethod
    def from_dataclasses(cls, classes: List[Type]):

        dataclass_descriptions = [
            DataclassDescription.from_dataclass(clazz) for clazz in classes
        ]
        # fetch all imports

        return cls(imports=set(), classes=dataclass_descriptions)
