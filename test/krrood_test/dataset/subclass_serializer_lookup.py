"""
Prints the qualified name of the serializer found for a subclass of a serialized class.
"""

from __future__ import annotations

import pathlib

from krrood.adapters.json_serializer import JSONSerializableTypeRegistry

# %% looking up the serializer

if __name__ == "__main__":
    serializer = JSONSerializableTypeRegistry().get_external_serializer(
        clazz=pathlib.PurePosixPath
    )
    print(serializer.__qualname__)
