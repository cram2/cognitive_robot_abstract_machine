from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Dict, Sequence, Optional, Tuple, Any, Set

from krrood.patterns.role.role_transformer import TransformationMode

GROUND_TRUTH = TransformationMode.GROUND_TRUTH.value
TRANSFORMED = TransformationMode.TRANSFORMED.value

import libcst as cst


def _resolve_relative_import(package: str, num_dots: int, module: str) -> str:
    """Resolve a relative import like ``from .foo import X`` to absolute form
    ``from pkg.foo import X`` given the *package* of the module containing it.

    ``.`` means the package itself, ``..`` means the parent package, etc.
    """
    parts = package.split(".")
    if num_dots > len(parts) + 1:
        return module  # too many dots — leave unresolved
    base_parts = parts[: len(parts) - (num_dots - 1)]
    base = ".".join(base_parts)
    return f"{base}.{module}" if base else module

# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------


@dataclass
class ModuleComparator:
    """
    Compares two python modules using libcst, without executing them.
    """

    generated_tree: cst.Module
    expected_tree: cst.Module
    ground_truth_package: Optional[str] = None

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
        """Verifies that all classes in the ground truth module exist in the generated module."""
        gen_names = set(self._gen_classes())
        exp_names = set(self._exp_classes())
        assert gen_names == exp_names, (
            f"Missing classes: {exp_names - gen_names}. "
            f"Extra classes: {gen_names - exp_names}"
        )

    def compare_class_hierarchy(self):
        """Verifies that class bases match between modules."""
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
                if "field" not in str(f["default"])
                or (
                    ("init=False" not in f["default"])
                    and ("kw_only=True" not in f["default"])
                )
            ]
            exp_order = [
                f["name"]
                for f in (_field_info(a) for a in _get_field_stmts(exp_cls))
                if "field" not in str(f["default"])
                or (
                    ("init=False" not in f["default"])
                    and ("kw_only=True" not in f["default"])
                )
            ]
            assert (
                gen_order == exp_order
            ), f"Field order of {name} mismatch: got {gen_order!r}, expected {exp_order!r}"

    def compare_method_details(self):
        """Tests that all methods, properties, their parameters, and return types match."""
        for name, gen_cls in self._gen_classes().items():
            if name in self._exp_classes():
                exp_cls = self._exp_classes()[name]
                self._compare_methods(name, gen_cls, exp_cls)

    def _compare_methods(
        self, cls_name: str, gen_cls: cst.ClassDef, exp_cls: cst.ClassDef
    ):
        gen_methods = {
            func.name.value: _method_info(func) for func in _get_method_stmts(gen_cls)
        }
        exp_methods = {
            func.name.value: _method_info(func) for func in _get_method_stmts(exp_cls)
        }

        missing = set(exp_methods.keys()) - set(gen_methods.keys())
        extra = set(gen_methods.keys()) - set(exp_methods.keys())

        assert not missing, f"Class {cls_name} is missing methods: {missing}"
        assert not extra, f"Class {cls_name} has extra methods: {extra}"

        for name in exp_methods:
            assert gen_methods[name] == exp_methods[name], (
                f"Method {cls_name}.{name} details mismatch.\n"
                f"Expected: {exp_methods[name]}\n"
                f"Got: {gen_methods[name]}"
            )

    def compare_imports(self):
        """Verifies that all imports match between modules."""
        import pprint
        gen_imports = self._get_imports(self.generated_tree)
        exp_imports = self._get_imports(self.expected_tree, package=self.ground_truth_package)
        assert gen_imports == exp_imports, (
            f"Imports mismatch:\n"
            f"got: {pprint.pformat(gen_imports)}\n"
            f"expected: {pprint.pformat(exp_imports)}"
        )

    def _get_imports(
        self, tree: cst.Module, package: Optional[str] = None
    ) -> Dict[str, Set[str]]:
        from collections import defaultdict

        def normalize(text: str) -> str:
            return (
                text.replace(GROUND_TRUTH, "")
                .replace(TRANSFORMED, "")
                .replace("typing_extensions", "typing")
            )

        class ImportCollector(cst.CSTVisitor):
            def __init__(self):
                self.imports = defaultdict(set)

            def visit_Import(self, node: cst.Import) -> None:
                for name in node.names:
                    mod_name = normalize(_code(name.name))
                    if name.asname:
                        mod_name += f" as {name.asname.name.value}"
                    self.imports["import"].add(mod_name)

            def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
                mod_name = normalize(_code(node.module)) if node.module else ""
                if node.relative and package:
                    mod_name = _resolve_relative_import(
                        package, len(node.relative), mod_name
                    )
                    key = f"from {mod_name}"
                else:
                    dots = "." * len(node.relative)
                    key = f"from {dots}{mod_name}"
                for name in node.names:
                    if isinstance(name.name, cst.ImportStar):
                        self.imports[key].add("*")
                    else:
                        name_str = normalize(_code(name.name))
                        if name.asname:
                            name_str += f" as {name.asname.name.value}"
                        self.imports[key].add(name_str)

        collector = ImportCollector()
        tree.visit(collector)
        return dict(collector.imports)


def get_module_comparators(
    generated_modules: Dict[ModuleType, Tuple[str, str]], stubs: bool = False
):
    comparators = []

    for module, (transformed_code, mixin_code) in generated_modules.items():
        # Transformed module
        expected_transformed, transformed_package = get_ground_truth_module_source(
            module, is_mixin=False
        )
        comparators.append(
            get_comparator_for_modules(
                transformed_code, expected_transformed, transformed_package
            )
        )

        # Mixin module
        expected_mixin, mixin_package = get_ground_truth_module_source(
            module, is_mixin=True
        )
        comparators.append(
            get_comparator_for_modules(mixin_code, expected_mixin, mixin_package)
        )

    return comparators  # no cleanup needed — no sys.modules pollution


def get_ground_truth_module_source(
    generated_module: ModuleType, is_mixin: bool = False
) -> Tuple[str, str]:
    path = Path(generated_module.__file__)
    module_name = path.stem
    package = generated_module.__package__
    if is_mixin:
        ground_truth_name = f"{GROUND_TRUTH}{module_name}_role_mixins.py"
        ground_truth_path = path.parent / "role_mixins" / ground_truth_name
        package = f"{package}.role_mixins"
    else:
        ground_truth_name = f"{GROUND_TRUTH}{TRANSFORMED}{module_name}.py"
        ground_truth_path = path.parent / ground_truth_name

    with open(ground_truth_path, "r") as f:
        return f.read(), package


def get_comparator_for_modules(
    source_of_module_to_be_validated: str,
    source_of_ground_truth_module: str,
    ground_truth_package: Optional[str] = None,
):
    return ModuleComparator(
        parse_module(source_of_module_to_be_validated),
        parse_module(source_of_ground_truth_module),
        ground_truth_package=ground_truth_package,
    )


# ---------------------------------------------------------------------------
# CST helpers
# ---------------------------------------------------------------------------


def parse_module(source: str) -> cst.Module:
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


def _get_method_stmts(cls: cst.ClassDef) -> list[cst.FunctionDef]:
    """Return function definition nodes that represent methods."""
    methods = []
    for item in cls.body.body:
        if isinstance(item, cst.FunctionDef):
            methods.append(item)
    return methods


def _method_info(func: cst.FunctionDef) -> Dict[str, Any]:
    """Extract decorators, parameters, and return type from a FunctionDef."""
    decorators = []
    for decorator in func.decorators:
        decorators.append(_code(decorator.decorator))

    params = []
    for param in func.params.params:
        param_name = param.name.value
        param_type = (
            _code(param.annotation.annotation) if param.annotation else None
        )
        params.append((param_name, param_type))

    return_type = _code(func.returns.annotation) if func.returns else None

    return {
        "decorators": sorted(decorators),
        "params": params,
        "return_type": return_type,
    }
