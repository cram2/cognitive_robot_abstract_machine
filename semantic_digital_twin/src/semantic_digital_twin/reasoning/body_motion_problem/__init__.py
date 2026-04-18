"""
Body Motion Problem (BMP) framework.

Implements the Law of Task-Achieving Body Motion:
  ∀R, E, Π, G, τ:
    SatisfiesRequest(Π, G_final) ∧
    Causes(τ, G_final, Φ, I_Φ) ∧
    CanPerform(R, τ)
    ⟹ CanAchieve(R, E, Π, τ)

This package provides the abstract framework types and predicates.
Domain-specific instantiations live in subpackages (e.g., container_manipulation).
"""

from semantic_digital_twin.reasoning.body_motion_problem.types import (
    TEEClass,
    TaskRequest,
    Motion,
    Effect,
    MonotoneIncreasingEffect,
    MonotoneDecreasingEffect,
    OutOfScopeError,
    PhysicsModel,
)
from semantic_digital_twin.reasoning.body_motion_problem.predicates import (
    Causes,
    SatisfiesRequest,
    CanPerform,
)

__all__ = [
    "TEEClass",
    "TaskRequest",
    "Motion",
    "Effect",
    "MonotoneIncreasingEffect",
    "MonotoneDecreasingEffect",
    "OutOfScopeError",
    "PhysicsModel",
    "Causes",
    "SatisfiesRequest",
    "CanPerform",
]
