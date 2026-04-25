from __future__ import annotations
import os
from dataclasses import dataclass, fields, field
from pathlib import Path
from typing import List, Set, Type, Optional, Tuple

import jinja2

from krrood.utils import run_black_on_file, module_and_class_name


@dataclass
class DataclassRenderer:
    """
    Rendering definition to write dataclasses that have been created in memory into a python module.

    .. note:: Fields are not rendered yet.
    """

    type_: Type
    """
    The dataclass to render.
    """

    imports: set[str] = field(default_factory=set, init=False)
    """
    The imports that need to be added to the module.
    They are calculated from the dataclass.
    """

    base_classes: list[str] = field(default_factory=list, init=False)
    """
    The base classes that the dataclass inherits from as list of strings.
    """

    def __post_init__(self):
        self._initialized_base_classes()

    def _initialized_base_classes(self):
        for base in self.type_.__bases__:
            if base.__module__ == self.type_.__module__:
                self.base_classes.append(base.__name__)
            else:
                self.base_classes.append(module_and_class_name(base))
                self.imports.add(base.__module__)


@dataclass
class ModuleRenderer:
    """
    Rendering definition to write dataclasses that have been created in memory into a python module.
    """

    classes: List[DataclassRenderer] = field(default_factory=list)
    """
    Classes that should be rendered.
    """

    imports: Set[str] = field(default_factory=set)
    """
    Imports collected from all classes
    """

    def _update_imports(self):
        for clazz in self.classes:
            self.imports.update(clazz.imports)

    @classmethod
    def from_dataclasses(cls, classes: List[Type]):
        dataclass_descriptions = [DataclassRenderer(clazz) for clazz in classes]
        result = cls(classes=dataclass_descriptions)
        result._update_imports()
        return result

    def write_to_file(self, path: Path):
        """
        Write the module to a file.

        :param path: The path to write the module to.
        """
        template_dir = os.path.join(os.path.dirname(__file__), "..", "jinja_templates")
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        template = env.get_template("python_module.py.jinja")

        # Render the template
        output = template.render(
            module_description=self,
        )

        with open(path, "w") as file:
            # Write the output to the file
            file.write(output)

        # format the output with black
        run_black_on_file(str(file.name))
