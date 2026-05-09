from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from krrood.class_diagrams import ClassDiagram
from krrood.class_diagrams.class_diagram import WrappedSpecializedGeneric
from krrood.patterns.role import Role
from krrood.patterns.role.exceptions import MissingRoleMixinsError
from krrood.patterns.role.role_transformer import RoleTransformer


def _modules_with_roles(class_diagram: ClassDiagram) -> list[ModuleType]:
    """Return the deduplicated list of modules that contain direct Role subclasses."""
    result: list[ModuleType] = []
    for wrapped_class in class_diagram.wrapped_classes:
        if (
            not isinstance(wrapped_class, WrappedSpecializedGeneric)
            and Role in wrapped_class.clazz.__bases__
        ):
            module = sys.modules[wrapped_class.clazz.__module__]
            if module not in result:
                result.append(module)
    return result


def transform_roles_in_class_diagram(
    class_diagram: ClassDiagram, write: bool = True
) -> None:
    """
    Transform role classes that are in the given class diagram.

    This function is intended for **offline use** (e.g. from the
    ``krrood-generate-role-mixins`` CLI or CI setup scripts).  Calling it
    inside a live test process will reload taker modules, breaking
    ``isinstance`` checks for already-imported class references.

    :param class_diagram: Class diagram to transform.
    :param write: When *True* (default) write the generated mixin files and
        the transformed original sources to disk.  When *False* the
        transformation is computed but no files are written (dry-run).
    """
    for module in _modules_with_roles(class_diagram):
        transformer = RoleTransformer(module)
        transformer.transform(write=write)


def check_role_mixin_files(
    class_diagram: ClassDiagram,
    packages: list[str],
) -> None:
    """
    Verify that all role mixin files required by *class_diagram* exist on disk.

    Intended for use in ``pytest_configure`` hooks: call this instead of
    :func:`transform_roles_in_class_diagram` so that the test process never
    triggers a module reload.  If any mixin file is absent the function raises
    :class:`~krrood.patterns.role.exceptions.MissingRoleMixinsError` with a
    human-readable message that includes the exact ``krrood-generate-role-mixins``
    command needed to regenerate the files.

    :param class_diagram: Class diagram to inspect.
    :param packages: The package names used to build *class_diagram*; included
        in the error message so users know exactly what to pass to the CLI.
    :raises MissingRoleMixinsError: If one or more mixin files are missing.
    """
    missing: list[Path] = []
    for module in _modules_with_roles(class_diagram):
        transformer = RoleTransformer(module)
        all_modules: list[ModuleType] = list(transformer.taker_modules)
        if transformer.module not in all_modules:
            all_modules.append(transformer.module)
        for m in all_modules:
            mixin_path = transformer.get_generated_file_path(m, is_mixin=True)
            if not mixin_path.exists():
                missing.append(mixin_path)

    if missing:
        packages_str = " ".join(packages)
        files_str = "\n    ".join(str(p) for p in missing)
        raise MissingRoleMixinsError(
            f"Role mixin files are missing. Run:\n\n"
            f"    krrood-generate-role-mixins {packages_str}\n\n"
            f"Missing files:\n    {files_str}"
        )
