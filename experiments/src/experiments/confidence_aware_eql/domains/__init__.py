"""
Domain plugins. Each module here defines ONE domain and is auto-discovered.

A domain module must expose:
    DOMAIN      : a confidence_engine.Domain
    SPEC        : {class_name: {feature_name: (mean,std) | category_label}}
Optionally (used by the demo and tests):
    FAMILIAR    : list[(name, object_dict)]   expected to PASS
    ANOMALOUS   : list[(name, object_dict)]   expected to be FLAGGED
    INCOMPLETE  : list[(name, object_dict)]   missing/unknown tag -> FLAGGED

Files starting with "_" (e.g. _template.py) are ignored by discovery.
"""
