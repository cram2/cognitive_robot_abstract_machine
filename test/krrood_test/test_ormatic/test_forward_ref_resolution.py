"""
Tests for forward reference resolution in the ORMatic module.

This test reproduces the issue where get_type_hints fails when there are
multiple forward references that need to be resolved iteratively.
"""

import sys

from krrood.class_diagrams.class_diagram import ClassDiagram


class TestForwardReferenceResolution:
    """Tests for forward reference resolution in ClassDiagram."""

    def test_multiple_forward_references_resolution(self):
        """
        Test that ClassDiagram can resolve multiple forward references.

        This reproduces the bug where:
        1. get_type_hints fails with NameError for the first unresolved type
        2. The code catches the error and adds that type to the namespace
        3. get_type_hints is called again but fails for another unresolved type

        The fix should handle multiple NameErrors iteratively.
        """
        # Ensure isolated modules are NOT in sys.modules before the test
        # This simulates the condition where forward reference types are not loaded
        modules_to_remove = [
            key for key in list(sys.modules.keys()) if "isolated_forward_ref" in key
        ]
        for mod in modules_to_remove:
            del sys.modules[mod]

        # Import IsolatedTypeAlpha so it's in sys.modules and can be found
        # But IsolatedTypeBeta is NOT imported into the local namespace, so
        # it won't be found there
        from test.krrood_test.dataset.isolated_forward_ref_types import (
            IsolatedTypeAlpha,
        )

        # Now import the isolated classes - the types they reference
        # (IsolatedTypeAlpha, IsolatedTypeBeta) are under TYPE_CHECKING
        # so they won't be imported at runtime
        from test.krrood_test.dataset.isolated_forward_ref_classes import (
            IsolatedClassWithMultipleMixins,
            IsolatedMixinAlpha,
            IsolatedMixinBeta,
        )

        # Create a ClassDiagram with classes that have forward references
        # IsolatedTypeAlpha IS in sys.modules (imported above)
        # IsolatedTypeBeta is NOT in sys.modules (not imported)
        # This should trigger the bug where:
        # 1. First get_type_hints fails for IsolatedTypeAlpha
        # 2. IsolatedTypeAlpha is found in sys.modules and added to namespace
        # 3. Second get_type_hints is called but fails for IsolatedTypeBeta
        classes = [
            IsolatedClassWithMultipleMixins,
            IsolatedMixinAlpha,
            IsolatedMixinBeta,
        ]

        # This should not raise a NameError - the fix should handle multiple
        # forward references iteratively
        class_diagram = ClassDiagram(classes)

        # Verify that the class diagram was created successfully
        assert class_diagram is not None

        # Verify that the main classes are in the diagram
        class_names = {wc.clazz.__name__ for wc in class_diagram.wrapped_classes}
        assert "IsolatedClassWithMultipleMixins" in class_names
