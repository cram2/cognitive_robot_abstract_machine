from dataclasses import dataclass

value = 1
"""
The value.
"""
@dataclass
class Holder:
    """Holds things."""
    content: int
