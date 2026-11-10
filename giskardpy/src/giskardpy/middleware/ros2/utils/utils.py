from __future__ import division

import xacro

from semantic_digital_twin.adapters.package_resolver import CompositePathResolver


def load_xacro(path: str) -> str:
    path = CompositePathResolver().resolve(path)
    doc = xacro.process_file(path, mappings={"radius": "0.9"})
    return doc.toprettyxml(indent="  ")
