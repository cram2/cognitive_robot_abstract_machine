import os
import sys
from os.path import dirname
from unittest import TestCase

from krrood.ripple_down_rules.utils import (
    make_set,
    get_imports_from_types,
)
from krrood.utils import get_scope_from_imports, get_relative_import


class UtilsTestCase(TestCase):

    def test_extract_imports_from_file(self):
        # Test with a file that has imports
        file_path = "test_file.py"
        with open(file_path, "w") as f:
            f.write("import os\n")
            f.write("from krrood.ripple_down_rules.utils import make_set\n")
            f.write("print('Hello World')\n")

        expected_scope = {"os": os, "make_set": make_set}
        actual_imports = get_scope_from_imports(file_path)
        self.assertEqual(expected_scope, actual_imports)

        # Clean up
        os.remove(file_path)

    def test_get_relative_import(self):
        package_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "krrood",
            "src",
            "krrood",
            "ripple_down_rules",
        )

        target_file = os.path.join(package_dir, "datastructures", "case.py")
        imported_module = os.path.join(package_dir, "rdr.py")

        rel_import = get_relative_import(target_file, imported_module)
        expected_import = "..rdr"
        assert rel_import == expected_import

        imported_module = os.path.join(package_dir, "datastructures", "case.py")
        target_file = os.path.join(package_dir, "rdr.py")

        rel_import = get_relative_import(target_file, imported_module)
        expected_import = ".datastructures.case"
        assert rel_import == expected_import

        target_file = os.path.join(
            package_dir, "datastructures", "nested_data", "case.py"
        )
        imported_module = os.path.join(package_dir, "rdr.py")

        rel_import = get_relative_import(target_file, imported_module)
        expected_import = "...rdr"
        assert rel_import == expected_import

    def test_get_imports_from_types(self):
        from .datasets import World, Species

        imports = get_imports_from_types([World, Species])

        assert imports == [
            "from test.krrood_test.test_ripple_down_rules.datasets import Species, World"
        ]

        from krrood.ripple_down_rules.rdr import GeneralRDR

        package_dir = dirname(sys.modules["krrood.ripple_down_rules"].__file__)
        target_file = os.path.join(package_dir, "datastructures", "case.py")
        imports = get_imports_from_types([GeneralRDR], target_file, "ripple_down_rules")
        assert imports == ["from ..rdr import GeneralRDR"]
