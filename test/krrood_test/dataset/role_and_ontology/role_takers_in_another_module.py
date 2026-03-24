from dataclasses import dataclass


@dataclass(eq=False)
class RoleTakerInAnotherModule:
    original_attribute: str
