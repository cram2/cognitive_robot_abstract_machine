"""
Generates a committed :class:`VerbalizationResult` snapshot module from a
:class:`VerbalizationResultsOfPackage`.

Uses :class:`~krrood.code_generation.generator.CodeGenerator` so the module is produced
rather than hand-transcribed: call :func:`regenerate_verbalization_results` from a
package's own ``conftest.py`` at import time, so the committed file is always fresh and a
wording change shows up as an ordinary diff to review before committing.
"""

from __future__ import annotations

import importlib.resources
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from typing_extensions import Any, Dict, Tuple, Type, Union

from krrood.code_generation.formatting import (
    run_ruff_check_on_file,
    run_ruff_format_on_file,
)
from krrood.code_generation.generator import CodeGenerator
from krrood.code_generation.imports import get_imports_from_types
from krrood.code_generation.type_hints import value_to_source
from krrood.entity_query_language.predicate import SymbolicCallable
from krrood.entity_query_language.testing.result_verification import (
    VerbalizationResultsOfPackage,
    VerbalizationResult,
)


@dataclass
class VerbalizationResultGenerator:
    """
    Renders the Python source of a ``results`` snapshot module for a
    :class:`VerbalizationResultsOfPackage`.
    """

    snapshot: VerbalizationResultsOfPackage
    """
    The snapshot whose covered callables and renderings this generator emits.
    """

    code_generator: CodeGenerator = field(init=False)
    """
    Renderer bound to this package's templates directory.
    """

    def __post_init__(self):
        templates = importlib.resources.files(__package__) / "templates"
        self.code_generator = CodeGenerator(template_directory=str(templates))

    def covered_callables(self) -> Tuple[Type[SymbolicCallable], ...]:
        """
        The callables this generator emits an entry for.

        :return: the discovered callables that implement their own verbalization
            fragment, in the order they appear in the generated module.
        """
        return tuple(
            cls
            for cls in self.snapshot.discovered_callables()
            if self.snapshot.has_fragment(cls)
        )

    def covered_results(self) -> Tuple[VerbalizationResult, ...]:
        """
        The data the generated module declares.

        :return: one :class:`VerbalizationResult` per covered callable, built directly
            from the snapshot's own rendering.
        """
        return tuple(
            VerbalizationResult(cls, self.snapshot.rendered_result(cls))
            for cls in self.covered_callables()
        )

    @staticmethod
    def _entry(result: VerbalizationResult) -> Dict[str, Any]:
        return {
            "class_name": result.callable_class.__qualname__,
            "sentence": value_to_source(result.sentence),
        }

    def generate(self) -> str:
        """:return: the Python source of a module declaring ``results``, one
        :class:`VerbalizationResult` per covered callable."""
        results = self.covered_results()
        imports = get_imports_from_types(
            [
                VerbalizationResult,
                *(result.callable_class for result in results),
            ]
        )
        entries = [self._entry(result) for result in results]
        return self.code_generator.render(
            "verbalization_results.py.jinja", imports=imports, entries=entries
        )

    def write(self, path: Union[str, Path]) -> None:
        """
        Render :meth:`generate` to *path* and format it with Ruff.

        :param path: The file to write the generated module to.
        """
        Path(path).write_text(self.generate())
        run_ruff_check_on_file(str(path))
        run_ruff_format_on_file(str(path))


def regenerate_verbalization_results(
    package: ModuleType, destination: Union[str, Path]
) -> None:
    """
    Regenerate *package*'s committed ``results`` snapshot module at *destination*.

    Writes to a temporary file in *destination*'s own directory and replaces it
    atomically, so a concurrent reader (e.g. a pytest-xdist worker) never observes a
    partially written file. Call this from a package's own ``conftest.py`` at import
    time, so this is a one-line addition for any package, not a bespoke script.

    :param package: The package whose symbolic callables are discovered and rendered.
    :param destination: The module file to (re)write.
    """
    destination = Path(destination)
    generator = VerbalizationResultGenerator(
        snapshot=VerbalizationResultsOfPackage(package=package, results=())
    )
    with tempfile.NamedTemporaryFile(
        "w", dir=destination.parent, suffix=".py.tmp", delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    generator.write(temporary_path)
    temporary_path.replace(destination)
