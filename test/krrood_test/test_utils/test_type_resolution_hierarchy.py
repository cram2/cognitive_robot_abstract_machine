import sys
from pathlib import Path

from krrood.class_diagrams.utils import get_type_hints_of_object


def test_reproduce_name_error_in_hierarchy(tmp_path: Path):
    # Create module_a.py
    module_a_path = tmp_path / "module_a.py"
    module_a_path.write_text("""
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from module_target import Target

class Base:
    target: 'Target'
""")

    # Create module_target.py
    module_target_path = tmp_path / "module_target.py"
    module_target_path.write_text("""
class Target:
    pass
""")

    # Create module_b.py
    module_b_path = tmp_path / "module_b.py"
    module_b_path.write_text("""
from module_a import Base
class Child(Base):
    pass
""")

    # Add tmp_path to sys.path
    sys.path.insert(0, str(tmp_path))

    try:
        import module_b

        Child = module_b.Child

        # This SHOULD now succeed because Target is found in module_a.py which is in Child's hierarchy
        hints = get_type_hints_of_object(Child)

        assert "target" in hints
        assert hints["target"].__name__ == "Target"
        assert hints["target"].__module__ == "module_target"

    finally:
        sys.path.pop(0)
        for mod in ["module_a", "module_b", "module_target"]:
            if mod in sys.modules:
                del sys.modules[mod]


def test_hierarchy_resolution_prioritizes_closest_match(tmp_path: Path):
    # Create module_a.py with Base referring to Target (intended to be TargetA)
    (tmp_path / "module_a.py").write_text("""
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from module_target_a import Target

class Base:
    target: 'Target'
""")

    # Create module_target_a.py
    (tmp_path / "module_target_a.py").write_text("""
class Target:
    pass
""")

    # Create module_b.py with Parent(Base) and its own Target (TargetB)
    (tmp_path / "module_b.py").write_text("""
from module_a import Base
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from module_target_b import Target

class Parent(Base):
    pass
""")

    # Create module_target_b.py
    (tmp_path / "module_target_b.py").write_text("""
class Target:
    pass
""")

    # Create module_c.py with Child(Parent)
    (tmp_path / "module_c.py").write_text("""
from module_b import Parent
class Child(Parent):
    pass
""")

    sys.path.insert(0, str(tmp_path))
    try:
        import module_c

        Child = module_c.Child

        # When resolving Target for Child, it should first look in module_c (none),
        # then module_b (Parent's module). In module_b, Target is TargetB.
        hints = get_type_hints_of_object(Child)

        assert hints["target"].__module__ == "module_target_b"

    finally:
        sys.path.pop(0)
        for mod in [
            "module_a",
            "module_b",
            "module_c",
            "module_target_a",
            "module_target_b",
        ]:
            if mod in sys.modules:
                del sys.modules[mod]


def test_hierarchy_resolution_stops_at_external_modules():
    from krrood.class_diagrams.utils import is_external_module
    import typing
    import typing_extensions

    assert is_external_module(typing) is True
    assert is_external_module(typing_extensions) is True
    assert is_external_module(sys.modules["builtins"]) is True

    # Check a standard library module
    import json

    assert is_external_module(json) is True

    # Check this module (should NOT be external if it's in the project)
    import krrood.class_diagrams.utils

    assert is_external_module(krrood.class_diagrams.utils) is False
