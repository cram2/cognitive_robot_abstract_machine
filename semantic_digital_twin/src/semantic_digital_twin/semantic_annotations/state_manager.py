"""
Temporal state management for the Semantic Digital Twin (SDT).
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, Deque, List, Optional, Tuple, Type, TypeVar

from semantic_digital_twin.semantic_annotations.object_state import ObjectState

# Generic type variable bound to ObjectState child-classes
T = TypeVar("T", bound=ObjectState)


class StateUnknownError(Exception):
    pass


class LowConfidenceError(Exception):
    pass


class StaleStateError(Exception):
    pass


class DynamicStateManager:
    """
    Memory-bounded temporal SDT state manager.
    Tracks polymorphic state streams dynamically for world entities.
    """

    def __init__(
        self,
        max_history_per_object: int = 200,
        max_history_seconds: Optional[float] = 300.0,
    ):
        self.max_history_size = max_history_per_object
        self.max_history_seconds = max_history_seconds

        # Polymorphic ring-buffer timeline per entity ID
        self._state_history: Dict[str, Deque[ObjectState]] = defaultdict(
            lambda: deque(maxlen=self.max_history_size)
        )

    # ------------------------------------------------------------------------
    # Update State Transactions
    # ------------------------------------------------------------------------

    def update_state(self, entity_id: str, new_state: ObjectState) -> None:
        """Appends a new polymorphic state observation into the entity's history tracking thread."""
        if new_state.timestamp is None:
            new_state.timestamp = time.time()

        history = self._state_history[entity_id]

        if history:
            # Drop duplicates by verifying against the last recorded slice of the SAME type
            last_matching_type = self._get_latest_of_type(history, type(new_state))
            if last_matching_type and self._is_duplicate(last_matching_type, new_state):
                return  # Block duplicate frame junk

        history.append(new_state)
        self._cleanup_old(entity_id)

    # ------------------------------------------------------------------------
    # State Retrieval Engines
    # ------------------------------------------------------------------------

    def get_current_state_by_type(self, entity_id: str, state_class: Type[T]) -> T:
        """Retrieves the latest available slice of a specific subclass type definition."""
        history = self._state_history.get(entity_id)
        if not history:
            raise StateUnknownError(
                f"No state history trace recorded for entity '{entity_id}'"
            )

        target_state = self._get_latest_of_type(history, state_class)
        if not target_state:
            raise StateUnknownError(
                f"No state segment matching type '{state_class.__name__}' active for entity '{entity_id}'"
            )
        return target_state

    def get_state_history(self, entity_id: str) -> List[ObjectState]:
        """Returns the full chronological interleaved state log list for an entity."""
        return list(self._state_history.get(entity_id, []))

    def get_state_at(
        self, entity_id: str, state_class: Type[T], timestamp: float | int
    ) -> Optional[T]:
        """Finds the most recent state subclass frame matching the provided query timestamp."""
        history = self._state_history.get(entity_id, [])
        for state in reversed(history):
            if (
                isinstance(state, state_class)
                and state.timestamp is not None
                and state.timestamp <= timestamp
            ):
                return state
        return None

    # ------------------------------------------------------------------------
    # Temporal Transition Parsing
    # ------------------------------------------------------------------------

    def get_transitions(self, entity_id: str) -> List[Tuple[ObjectState, ObjectState]]:
        """Maps discrete step-change transitions matching structural bounds across time tracks."""
        history = self._state_history.get(entity_id, [])
        transitions = []
        last_seen_by_type: Dict[str, ObjectState] = {}

        for curr_state in history:
            stype = curr_state.state_type
            prev_state = last_seen_by_type.get(stype)

            if prev_state is not None:
                # Direct structural check against internal enum updates
                if hasattr(prev_state, "state") and hasattr(curr_state, "state"):
                    if getattr(prev_state, "state") != getattr(curr_state, "state"):
                        transitions.append((prev_state, curr_state))

            last_seen_by_type[stype] = curr_state

        return transitions

    # ------------------------------------------------------------------------
    # Validation Rules (Exposed Hooks for Predicate Execution)
    # ------------------------------------------------------------------------

    def validate_state_safety(
        self, state: ObjectState, min_confidence: float, max_age_sec: Optional[float]
    ) -> None:
        """Guards predicate calculations from tracking processing noise or stale artifacts."""
        if state.confidence < min_confidence:
            raise LowConfidenceError(
                f"State confidence ({state.confidence}) is below required threshold ({min_confidence})"
            )

        if max_age_sec is not None and state.timestamp is not None:
            age = time.time() - state.timestamp
            if age > max_age_sec:
                raise StaleStateError(
                    f"State observation is {age:.2f}s old (exceeds {max_age_sec:.2f}s safety margin)"
                )

    # ------------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------------

    def _get_latest_of_type(
        self, history: Deque[ObjectState], state_class: Type[T]
    ) -> Optional[T]:
        """Crawls backwards up the history sequence to locate the last specific type instance."""
        for state in reversed(history):
            if isinstance(state, state_class):
                return state
        return None

    def _cleanup_old(self, entity_id: str) -> None:
        if self.max_history_seconds is None:
            return

        now = time.time()
        history = self._state_history[entity_id]

        while history:
            oldest_state = history[0]
            if (
                oldest_state.timestamp is not None
                and (now - oldest_state.timestamp) > self.max_history_seconds
            ):
                history.popleft()
            else:
                break

    def _is_duplicate(self, a: ObjectState, b: ObjectState) -> bool:
        if type(a) != type(b) or a.source != b.source:
            return False
        return (
            getattr(a, "state", None) == getattr(b, "state", None)
            and abs(a.confidence - b.confidence) < 1e-3
        )
