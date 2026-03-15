from __future__ import annotations

from dataclasses import dataclass

from .university_ontology_like_classes_without_descriptors import (
    ProfessorAsFirstRole,
)


@dataclass(eq=False)
class AssistantProfessorAsSubClassOfARoleInAnotherModule(ProfessorAsFirstRole): ...
