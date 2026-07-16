from __future__ import division

import os

import xacro
from typing_extensions import Dict, Optional

from semantic_digital_twin.adapters.package_resolver import CompositePathResolver


def load_xacro(path: str, mappings: Optional[Dict[str, str]] = None) -> str:
    path = CompositePathResolver().resolve(path)
    doc = xacro.process_file(path, mappings={"radius": "0.9", **(mappings or {})})
    return doc.toprettyxml(indent="  ")


def is_in_github_workflow():
    return "GITHUB_WORKFLOW" in os.environ
