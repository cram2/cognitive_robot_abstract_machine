from dataclasses import dataclass


@dataclass
class TypeReferencedOnlyInAnnotations:
    """A type that is referenced solely from another class's field annotations.

    Stands in for annotation-only field types (such as ``semantic_digital_twin``'s
    ``EndEffector`` referenced by ``coraplex``'s ``GraspDescription.end_effector``) without
    depending on those packages.
    """

    name: str = ""
