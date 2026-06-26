import importlib
import pkgutil

import domains

REQUIRED_ATTRS = ("DOMAIN", "SPEC")


def discover() -> dict:
    found = {}
    for info in pkgutil.iter_modules(domains.__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{domains.__name__}.{info.name}")
        if all(hasattr(module, attribute) for attribute in REQUIRED_ATTRS):
            found[module.DOMAIN.name] = module
    return dict(sorted(found.items()))
