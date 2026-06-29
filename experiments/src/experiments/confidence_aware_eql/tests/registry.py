"""
registry.py — auto-discover domain plugins in domains/.

A module qualifies as a domain if it defines DOMAIN and SPEC. Files whose names
start with "_" (e.g. _template.py) are skipped. This is what lets you add a
domain by simply creating a file.
"""

import importlib
import pkgutil

import domains

REQUIRED_ATTRS = ("DOMAIN", "SPEC")


def discover() -> dict:
    """Return {domain_name: module} for every valid domain plugin."""
    found = {}
    for info in pkgutil.iter_modules(domains.__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"domains.{info.name}")
        if all(hasattr(module, a) for a in REQUIRED_ATTRS):
            found[module.DOMAIN.name] = module
    return dict(sorted(found.items()))
