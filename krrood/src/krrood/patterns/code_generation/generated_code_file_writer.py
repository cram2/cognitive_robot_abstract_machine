"""
File writer for generated module sources.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import ModuleType
from typing import Callable

from typing_extensions import Dict, Tuple

from krrood.utils import run_black_on_file, run_ruff_on_file


@dataclasses.dataclass
class GeneratedCodeFileWriter:
    """
    Writes transformed module sources and generated sources to the file system.
    """

    def write(
        self,
        module_sources: Dict[ModuleType, Tuple[str, str]],
        get_path_fn: Callable[[ModuleType, bool], Path],
    ) -> None:
        """Write all transformed sources to disk and run formatters on each file.

        :param module_sources: Mapping of module to (transformed_source, generated_source) tuples.
        :param get_path_fn: Callable that accepts (module, is_generated) and returns a Path.
        """
        generated_paths: list[Path] = []
        for module, (module_source, generated_source) in module_sources.items():
            original_path = get_path_fn(module, False)
            generated_path = get_path_fn(module, True)
            self._ensure_package_exists(generated_path.parent)
            for path, content in [
                (original_path, module_source),
                (generated_path, generated_source),
            ]:
                with open(path, "w") as f:
                    f.write(content)
                generated_paths.append(path)

        for path in generated_paths:
            run_ruff_on_file(str(path))
            run_black_on_file(str(path))

    @staticmethod
    def _ensure_package_exists(folder: Path) -> None:
        """Create the package directory and its ``__init__.py`` if they do not exist."""
        folder.mkdir(exist_ok=True)
        init_file = folder / "__init__.py"
        if not init_file.exists():
            init_file.touch()
