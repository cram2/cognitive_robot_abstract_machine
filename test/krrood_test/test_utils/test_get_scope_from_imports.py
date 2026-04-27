import ast
import os
import math
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict
import pytest
from krrood.utils import get_scope_from_imports
from krrood.exceptions import SourceDataNotProvided


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
    # Should not raise exception, but log warning
    scope = get_scope_from_imports(source=source)
    assert "non_existent_module_xyz" not in scope

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
