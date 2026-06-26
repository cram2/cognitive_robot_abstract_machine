"""Bathroom domain (note: 4 features incl. a boolean — different schema)."""

from experiments.confidence_aware_eql.engine import Domain, Feature

DOMAIN = Domain("bathroom", [
    Feature("weight", "continuous"),
    Feature("size", "continuous"),
    Feature("material", "categorical",
            categories={"cotton": 0, "plastic": 1, "ceramic": 2, "glass": 3}),
    Feature("waterproof", "categorical", categories={"no": 0, "yes": 1}),
])

SPEC = {
    "towel":       {"weight": (0.40, 0.08),  "size": (0.50, 0.05), "material": "cotton",  "waterproof": "no"},
    "soap_bottle": {"weight": (0.30, 0.05),  "size": (0.18, 0.02), "material": "plastic", "waterproof": "yes"},
    "toothbrush":  {"weight": (0.02, 0.005), "size": (0.18, 0.01), "material": "plastic", "waterproof": "yes"},
    "mirror":      {"weight": (1.50, 0.30),  "size": (0.40, 0.05), "material": "glass",   "waterproof": "yes"},
    "sink_cup":    {"weight": (0.15, 0.03),  "size": (0.10, 0.02), "material": "ceramic", "waterproof": "yes"},
}

FAMILIAR = [
    ("normal_towel",  {"weight": 0.40, "size": 0.50, "material": "cotton",  "waterproof": "no"}),
    ("normal_soap",   {"weight": 0.30, "size": 0.18, "material": "plastic", "waterproof": "yes"}),
    ("normal_mirror", {"weight": 1.50, "size": 0.40, "material": "glass",   "waterproof": "yes"}),
]

ANOMALOUS = [
    ("impossible_soap", {"weight": 60.0, "size": 0.18, "material": "plastic", "waterproof": "yes"}),
    ("lead_towel",      {"weight": 35.0, "size": 0.50, "material": "cotton",  "waterproof": "no"}),
]

INCOMPLETE = [
    ("no_waterproof", {"weight": 0.30, "size": 0.18, "material": "plastic", "waterproof": None}),
]
