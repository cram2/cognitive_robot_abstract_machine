"""
TEMPLATE — copy me to add a new domain.

  1. Copy this file to domains/<your_domain>.py  (no leading underscore!)
  2. Fill in DOMAIN, SPEC, and the example objects below.
  3. Run:        python run.py          (your domain appears in the menu)
  4. Test:       python -m pytest -q    (your domain is auto-tested)

You do NOT edit confidence_engine/, run.py, registry.py, or the test suite.
"""

from experiments.confidence_aware_eql.engine import Domain, Feature

                                                                             
DOMAIN = Domain("dining", [
    Feature("weight", "continuous"),
    Feature("size", "continuous"),
    Feature("material", "categorical",
            categories={"glass": 0, "metal": 1, "linen": 2}),
])

                                                                            
                                                                          
SPEC = {
    "wine_glass": {"weight": (0.20, 0.03),  "size": (0.20, 0.02), "material": "glass"},
    "fork":       {"weight": (0.05, 0.01),  "size": (0.18, 0.01), "material": "metal"},
    "napkin":     {"weight": (0.02, 0.005), "size": (0.30, 0.03), "material": "linen"},
}

                                                                             
FAMILIAR = [
    ("normal_glass", {"weight": 0.20, "size": 0.20, "material": "glass"}),
    ("normal_fork",  {"weight": 0.05, "size": 0.18, "material": "metal"}),
]
ANOMALOUS = [
    ("impossible_glass", {"weight": 50.0, "size": 0.20, "material": "glass"}),
]
INCOMPLETE = [
    ("no_material", {"weight": 0.20, "size": 0.20, "material": None}),
]
