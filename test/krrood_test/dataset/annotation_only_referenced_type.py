from dataclasses import dataclass


@dataclass
class TypeReferencedOnlyInAnnotations:
    """
    A type that is referenced solely from another class's field annotations.

    Stands in for a field type that another package declares but never constructs, so
    the mapper has to discover it from the annotation alone.
    """

    name: str = ""
