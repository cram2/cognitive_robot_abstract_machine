"""
Finds the imports of a generated ORM interface that run when a module is loaded.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Iterator, override

from cognitive_robot_abstract_machine.orm_interfaces import (
    INTERFACE_FILE_NAME,
    OrmScriptFileName,
)
from krrood.development_tools.agents.checker import CodeRuleChecker, CodeRuleViolation
from krrood.source_analysis.import_graph import ModuleLevelImportFinder

# %% the check


@dataclass(kw_only=True, frozen=True)
class ModuleLevelOrmInterfaceImportChecker(CodeRuleChecker):
    """
    Finds every import of a generated ORM interface that runs when its module is loaded,
    in every module but an interface's generator.
    """

    @override
    def violations(self, *, path: Path) -> Iterator[CodeRuleViolation]:
        if path.name == OrmScriptFileName.GENERATOR:
            return
        finder = ModuleLevelImportFinder()
        finder.visit(ast.parse(path.read_text(), filename=str(path)))
        interface_module = Path(INTERFACE_FILE_NAME).stem
        for statement in finder.imports:
            imported_modules = [alias.name for alias in statement.names]
            if isinstance(statement, ast.ImportFrom):
                imported_modules.append(statement.module or "")
            if not any(
                interface_module in module.split(".") for module in imported_modules
            ):
                continue
            yield CodeRuleViolation(
                path=path,
                first_line=statement.lineno,
                last_line=statement.end_lineno,
                description="imports a generated ORM interface when the module is "
                "loaded; import it inside the functions that need it",
            )
