import inspect
from dataclasses import dataclass, fields, is_dataclass, field
from pathlib import Path
from typing import Dict, Any, Type, Optional, Sequence

import libcst as cst
import libcst.metadata as metadata
import pytest

from krrood.patterns.role.role_stub_generator_v2 import RoleStubGeneratorV2
from ..dataset.role_and_ontology import (
    university_ontology_like_classes_without_descriptors,
)

# ---------------------------------------------------------------------------
# CST helpers
# ---------------------------------------------------------------------------


def parse_stub(source: str) -> cst.Module:
    return cst.parse_module(source)


def _get_class_defs(tree: cst.Module) -> Dict[str, cst.ClassDef]:
    """Return a name→ClassDef mapping for every top-level class in the module."""
    return {
        stmt.name.value: stmt
        for stmt in tree.body
        if isinstance(stmt, cst.SimpleStatementLine) is False
        and isinstance(stmt, cst.ClassDef)
    }


def _keyword_value(keywords: Sequence[cst.Arg], name: str) -> Optional[str]:
    """Return the string representation of a keyword argument, or None."""
    for arg in keywords:
        if arg.keyword and arg.keyword.value == name:
            return _code(arg.value)
    return None


def _code(node: cst.CSTNode) -> str:
    """Render a CST node back to source text (whitespace-normalised)."""
    return cst.parse_module("").code_for_node(node).strip()


def _dataclass_decorator_kwargs(cls: cst.ClassDef) -> Dict[str, str]:
    """
    Extract keyword arguments from the @dataclass(...) decorator of a ClassDef.
    Returns an empty dict when the decorator has no arguments.
    """
    for dec in cls.decorators:
        dec_node = dec.decorator
        if isinstance(dec_node, cst.Name) and dec_node.value == "dataclass":
            return {}
        if isinstance(dec_node, cst.Call):
            func = dec_node.func
            if isinstance(func, cst.Name) and func.value == "dataclass":
                return {
                    arg.keyword.value: _code(arg.value)
                    for arg in dec_node.args
                    if arg.keyword is not None
                }
    return {}


def _base_names(cls: cst.ClassDef) -> list[str]:
    """Return a list of base-class name strings."""
    bases = []
    for arg in cls.bases:
        bases.append(_code(arg.value))
    return bases


def _get_field_stmts(cls: cst.ClassDef) -> list[cst.AnnAssign]:
    """Return annotated-assignment nodes that represent dataclass fields."""
    result = []
    for stmt in cls.body.body:
        # SimpleStatementLine wraps AnnAssign
        if isinstance(stmt, cst.SimpleStatementLine):
            for s in stmt.body:
                if isinstance(s, cst.AnnAssign):
                    result.append(s)
    return result


def _normalize_call_kwargs(code_str: Optional[str]) -> Optional[str]:
    """
    Normalize a function call string by sorting its keyword arguments alphabetically.
    e.g. 'field(kw_only=True, default=None)' == 'field(default=None, kw_only=True)'
    """
    if code_str is None:
        return None
    try:
        expr = cst.parse_expression(code_str)
    except cst.ParserSyntaxError:
        return code_str
    if not isinstance(expr, cst.Call):
        return code_str

    # Separate positional from keyword args, sort keywords by name
    positional = [arg for arg in expr.args if arg.keyword is None]
    keyword = sorted(
        [arg for arg in expr.args if arg.keyword is not None],
        key=lambda arg: arg.keyword.value,
    )

    # Rebuild with clean comma whitespace
    all_args = positional + keyword
    new_args = []
    for i, arg in enumerate(all_args):
        is_last = i == len(all_args) - 1
        comma = cst.MaybeSentinel.DEFAULT if not is_last else cst.MaybeSentinel.DEFAULT
        new_args.append(
            arg.with_changes(
                comma=(
                    cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))
                    if not is_last
                    else cst.MaybeSentinel.DEFAULT
                )
            )
        )

    new_expr = expr.with_changes(args=new_args)
    return _code(new_expr)


def _field_info(ann_assign: cst.AnnAssign) -> Dict[str, str]:
    name = _code(ann_assign.target)
    annotation = _code(ann_assign.annotation.annotation)
    raw_default = _code(ann_assign.value) if ann_assign.value is not None else None
    default = _normalize_call_kwargs(raw_default)
    return {"name": name, "annotation": annotation, "default": default}


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------


@dataclass
class StubComparator:
    """
    Compares two parsed .pyi stubs using libcst, without executing them.
    """

    generated_tree: cst.Module
    expected_tree: cst.Module

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gen_classes(self) -> Dict[str, cst.ClassDef]:
        return _get_class_defs(self.generated_tree)

    def _exp_classes(self) -> Dict[str, cst.ClassDef]:
        return _get_class_defs(self.expected_tree)

    # ------------------------------------------------------------------
    # Public comparison methods (same API as before)
    # ------------------------------------------------------------------

    def compare_class_existence(self):
        """Verifies that all classes in the expected stub exist in the generated stub."""
        gen_names = set(self._gen_classes())
        exp_names = set(self._exp_classes())
        assert gen_names == exp_names, (
            f"Missing classes: {exp_names - gen_names}. "
            f"Extra classes: {gen_names - exp_names}"
        )

    def compare_class_hierarchy(self):
        """Verifies that class bases match between stubs."""
        gen_classes = self._gen_classes()
        exp_classes = self._exp_classes()
        for name, exp_cls in exp_classes.items():
            gen_cls = gen_classes[name]
            assert _base_names(gen_cls) == _base_names(exp_cls), (
                f"Base classes of {name} mismatch: "
                f"got {_base_names(gen_cls)!r}, expected {_base_names(exp_cls)!r}"
            )

    def compare_field_details(self):
        """Verifies that all fields, their types, and defaults match."""
        gen_classes = self._gen_classes()
        exp_classes = self._exp_classes()
        for name, exp_cls in exp_classes.items():
            gen_cls = gen_classes[name]
            self._compare_fields(name, gen_cls, exp_cls)

    def _compare_fields(
        self, cls_name: str, gen_cls: cst.ClassDef, exp_cls: cst.ClassDef
    ):
        gen_fields = {
            f["name"]: f for f in (_field_info(a) for a in _get_field_stmts(gen_cls))
        }
        exp_fields = {
            f["name"]: f for f in (_field_info(a) for a in _get_field_stmts(exp_cls))
        }

        assert set(gen_fields) == set(exp_fields), (
            f"Fields of {cls_name} mismatch: "
            f"missing={set(exp_fields) - set(gen_fields)}, "
            f"extra={set(gen_fields) - set(exp_fields)}"
        )

        for field_name, exp_f in exp_fields.items():
            gen_f = gen_fields[field_name]
            assert gen_f["annotation"] == exp_f["annotation"], (
                f"Annotation of {cls_name}.{field_name} mismatch: "
                f"got {gen_f['annotation']!r}, expected {exp_f['annotation']!r}"
            )
            assert gen_f["default"] == exp_f["default"], (
                f"Default of {cls_name}.{field_name} mismatch: "
                f"got {gen_f['default']!r}, expected {exp_f['default']!r}"
            )

    def compare_dataclass_params(self):
        """Verifies that @dataclass decorator arguments match."""
        gen_classes = self._gen_classes()
        exp_classes = self._exp_classes()
        for name, exp_cls in exp_classes.items():
            gen_cls = gen_classes[name]
            gen_kwargs = _dataclass_decorator_kwargs(gen_cls)
            exp_kwargs = _dataclass_decorator_kwargs(exp_cls)
            assert gen_kwargs == exp_kwargs, (
                f"@dataclass kwargs of {name} mismatch: "
                f"got {gen_kwargs!r}, expected {exp_kwargs!r}"
            )

    def compare_field_order(self):
        """
        Verifies that fields appear in the same order.
        (Bonus check not present in the original — order matters for dataclasses.)
        """
        field(kw_only=True)
        gen_classes = self._gen_classes()
        exp_classes = self._exp_classes()
        for name, exp_cls in exp_classes.items():
            gen_cls = gen_classes[name]
            gen_order = [
                f["name"]
                for f in (_field_info(a) for a in _get_field_stmts(gen_cls))
                if f["default"]
                and ("init=False" not in f["default"])
                and ("kw_only=True" not in f["default"])
            ]
            exp_order = [
                f["name"]
                for f in (_field_info(a) for a in _get_field_stmts(exp_cls))
                if f["default"]
                and ("init=False" not in f["default"])
                and ("kw_only=True" not in f["default"])
            ]
            assert (
                gen_order == exp_order
            ), f"Field order of {name} mismatch: got {gen_order!r}, expected {exp_order!r}"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_comparators():
    generator = RoleStubGeneratorV2(
        university_ontology_like_classes_without_descriptors
    )
    generated_stub = generator.generate_stub()

    comparators = []

    for module in generated_stub:
        path = Path(module.__file__)
        expected_stub_path = path.with_name(f"_ground_truth_{path.name}").with_suffix(
            ".pyi"
        )

        with open(expected_stub_path, "r") as f:
            expected_stub_content = f.read()

        gen_tree = parse_stub(generated_stub[module])
        exp_tree = parse_stub(expected_stub_content)

        comparators.append(StubComparator(gen_tree, exp_tree))

    return comparators  # no cleanup needed — no sys.modules pollution


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.order("first")
def test_stub_generation_smoke():
    generator = RoleStubGeneratorV2(
        university_ontology_like_classes_without_descriptors
    )
    stub = generator.generate_stub(write=True)
    assert generator.path.exists()


def test_full_stub_comparison_class_existence(stub_comparators):
    """Tests that all classes defined in the expected stub exist in the generated stub."""
    for stub_comparator in stub_comparators:
        stub_comparator.compare_class_existence()


def test_full_stub_comparison_class_hierarchy(stub_comparators):
    """Tests that the class hierarchy (base classes) matches between stubs."""
    for stub_comparator in stub_comparators:
        stub_comparator.compare_class_hierarchy()


def test_full_stub_comparison_field_details(stub_comparators):
    """Tests that all fields, their types, and defaults match between stubs."""
    for stub_comparator in stub_comparators:
        stub_comparator.compare_field_details()


def test_full_stub_comparison_dataclass_params(stub_comparators):
    """Tests that @dataclass decorator arguments match between stubs."""
    for stub_comparator in stub_comparators:
        stub_comparator.compare_dataclass_params()


def test_full_stub_comparison_field_order(stub_comparators):
    """Tests that fields appear in the same order between stubs."""
    for stub_comparator in stub_comparators:
        stub_comparator.compare_field_order()
