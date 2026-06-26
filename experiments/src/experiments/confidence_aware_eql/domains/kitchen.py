"""Kitchen domain."""

from experiments.confidence_aware_eql.engine import Domain, Feature

DOMAIN = Domain("kitchen", [
    Feature("weight", "continuous"),
    Feature("size", "continuous"),
    Feature("material", "categorical",
            categories={"ceramic": 0, "glass": 1, "metal": 2, "plastic": 3}),
])

                                                                           
SPEC = {
    "cup":     {"weight": (0.25, 0.05), "size": (0.10, 0.02), "material": "ceramic"},
    "glass":   {"weight": (0.30, 0.05), "size": (0.12, 0.02), "material": "glass"},
    "plate":   {"weight": (0.60, 0.10), "size": (0.24, 0.02), "material": "ceramic"},
    "pitcher": {"weight": (2.50, 0.30), "size": (0.25, 0.03), "material": "glass"},
    "pot":     {"weight": (3.00, 0.40), "size": (0.30, 0.03), "material": "metal"},
}

FAMILIAR = [
    ("normal_cup",     {"weight": 0.22, "size": 0.10, "material": "ceramic"}),
    ("normal_pitcher", {"weight": 2.50, "size": 0.25, "material": "glass"}),
    ("normal_pot",     {"weight": 3.00, "size": 0.30, "material": "metal"}),
]

ANOMALOUS = [
    ("impossible_cup", {"weight": 50.0, "size": 0.10, "material": "glass"}),               
    ("heavy_plate",    {"weight": 40.0, "size": 0.24, "material": "ceramic"}),
    ("tiny_anvil",     {"weight": 25.0, "size": 0.05, "material": "metal"}),
]

INCOMPLETE = [
    ("no_material", {"weight": 0.30, "size": 0.09, "material": None}),
    ("unknown_mat", {"weight": 0.30, "size": 0.09, "material": "uranium"}),                  
]
