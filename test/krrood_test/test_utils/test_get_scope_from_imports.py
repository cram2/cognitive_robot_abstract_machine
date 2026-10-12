import ast
import inspect
import os
import math
from dataclasses import dataclass, fields
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict
import pytest
from typing_extensions import Optional

from krrood.utils import get_scope_from_imports
from krrood.exceptions import SourceDataNotProvided

from ..dataset import deferred_import_owner
from ..dataset.deferred_import_owner import Reading, SensorBindingItsReading
from ..dataset.type_checking_cycle_owner import Content, HolderBindingItsContent
from ..dataset.type_checking_import_of_missing_module import (
    OwnerOfAnnotationFromMissingModule,
)


def test_get_scope_from_imports_basic():
    source = "import os\nimport math as m"
    scope = get_scope_from_imports(source=source)
    assert scope["os"] == os
    assert scope["m"] == math


def test_get_scope_from_imports_from():
    source = "from pathlib import Path\nfrom collections import defaultdict as dd"
    scope = get_scope_from_imports(source=source)
    assert scope["Path"] == Path
    assert scope["dd"] == defaultdict


def test_get_scope_from_imports_wildcard():
    source = "from math import *"
    scope = get_scope_from_imports(source=source)
    # Check for some functions from math
    assert scope["sin"] == math.sin
    assert scope["cos"] == math.cos
    assert scope["pi"] == math.pi


def test_get_scope_from_imports_invalid():
    source = "import non_existent_module_xyz"
    with pytest.raises(
        ModuleNotFoundError, match="No module named 'non_existent_module_xyz'"
    ):
        scope = get_scope_from_imports(source=source)


def test_get_scope_from_imports_tree():
    source = "import os"
    tree = ast.parse(source)
    scope = get_scope_from_imports(tree=tree)
    assert scope["os"] == os


def test_get_scope_from_imports_no_input():
    with pytest.raises(SourceDataNotProvided):
        get_scope_from_imports()


def test_get_scope_from_imports_relative(tmp_path):
    root = tmp_path / "my_package"
    root.mkdir()
    (root / "__init__.py").touch()

    sub = root / "sub"
    sub.mkdir()
    (sub / "__init__.py").touch()

    module_a = sub / "module_a.py"
    module_a.write_text("class A: pass")

    module_b = sub / "module_b.py"
    module_b.write_text("from .module_a import A")

    # We need to make sure 'my_package' is importable or at least can be found
    # Since get_import_path_from_path uses filesystem, we might need to adjust sys.path if importlib is used
    # But wait, get_scope_from_imports calls importlib.import_module(module_name, package=package_name)
    # where package_name is derived from get_import_path_from_path

    import sys

    sys.path.append(str(tmp_path))
    try:
        scope = get_scope_from_imports(file_path=str(module_b))
        assert "A" in scope
        from my_package.sub.module_a import A

        assert scope["A"] == A
    finally:
        sys.path.remove(str(tmp_path))


# %% imports whose module cannot be found


def test_scope_holds_the_other_imports_when_one_targets_a_missing_module():
    """
    A file importing a name from a module that does not exist must still yield the scope
    its remaining imports define, instead of aborting the whole scope build.
    """
    source_path = inspect.getsourcefile(OwnerOfAnnotationFromMissingModule)

    scope = get_scope_from_imports(file_path=source_path)

    assert scope["Optional"] == Optional
    assert scope["dataclass"] == dataclass
    assert "GeneratedMapping" not in scope


# %% function-local imports


def test_scope_excludes_names_imported_inside_functions():
    """
    A name imported inside a function body is bound only within that function, so it is
    not part of the module's scope.
    """
    scope = get_scope_from_imports(file_path=deferred_import_owner.__file__)

    assert "ReadingOfBoundSensor" not in scope


def test_subclass_binding_during_import_narrows_field_despite_deferred_import():
    """
    Binding a generic parameter while the module is still being imported narrows the
    field, even though the module defers an import of a module importing it back.
    """
    reading_field = next(
        field for field in fields(SensorBindingItsReading) if field.name == "reading"
    )

    assert reading_field.type == Optional[Reading]


# %% circular imports seen only while type checking


def test_subclass_binding_during_import_narrows_field_despite_type_checking_cycle():
    """
    Binding a generic parameter while the module is still being imported narrows the
    field, even though one of the module's ``TYPE_CHECKING`` imports targets a module
    that imports it back.
    """
    content_field = next(
        field for field in fields(HolderBindingItsContent) if field.name == "content"
    )

    assert content_field.type == Optional[Content]
