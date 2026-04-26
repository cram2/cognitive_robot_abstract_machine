import os
from pathlib import Path
from collections import defaultdict
import sys
from krrood.utils import get_scope_from_imports


def test_get_type_checking_imports(tmp_path):
    source = """
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import os
    from collections import defaultdict
    import math as m
    from pathlib import Path as P
"""
    test_file = tmp_path / "test_imports.py"
    test_file.write_text(source)
    
    # Currently it returns a list of libcst nodes, but we want it to return a scope
    scope = get_scope_from_imports(str(test_file))
    
    assert isinstance(scope, dict)
    assert scope["os"] == os
    assert scope["defaultdict"] == defaultdict
    assert "m" in scope
    import math
    assert scope["m"] == math
    assert scope["P"] == Path

def test_get_type_checking_imports_typing_extensions(tmp_path):
    source = """
import typing_extensions
if typing_extensions.TYPE_CHECKING:
    import sys
"""
    test_file = tmp_path / "test_imports_te.py"
    test_file.write_text(source)
    
    scope = get_scope_from_imports(str(test_file))
    
    assert isinstance(scope, dict)
    assert scope["sys"] == sys

def test_get_type_checking_imports_already_in_sys_modules(tmp_path):
    # Ensure a module is in sys.modules
    import json
    
    source = """
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import json
"""
    test_file = tmp_path / "test_imports_sys.py"
    test_file.write_text(source)
    
    scope = get_scope_from_imports(str(test_file))
    assert scope["json"] == json
