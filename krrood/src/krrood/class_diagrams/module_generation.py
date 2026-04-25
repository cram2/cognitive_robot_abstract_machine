import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Set, Type, Optional

import jinja2

from krrood.utils import run_black_on_file


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

    def write_to_file(self, path: Path):
        template_dir = os.path.join(os.path.dirname(__file__), "..", "jinja_templates")
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        template = env.get_template("python_module.py.jinja")

        # Render the template
        output = template.render(
            ormatic=self.ormatic,
        )

        with open(path, "w") as file:
            # Write the output to the file
            file.write(output)

        # format the output with black
        run_black_on_file(str(file.name))
