from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from weakref import ReferenceType

from typing_extensions import TYPE_CHECKING, Any, Dict, Iterator, List

if TYPE_CHECKING:
    from krrood.patterns.role import Role


@dataclass
class RoleRegistry:
    """
    Runtime inverse index from a role taker to the roles attached to it.

    Roles use pure composition: a role references its taker, but a taker holds no reference
    back to its roles. This index maintains that reverse direction so a membership query
    ("which roles does this entity have?") can be answered without the taker knowing about
    roles.

    Entries are held weakly and keyed by taker identity, so the index never keeps a role or a
    taker alive and never conflates two distinct takers that happen to compare equal.
    """

    _roles_by_taker_identity: Dict[int, List[ReferenceType[Role]]] = field(
        default_factory=dict
    )
    """
    Maps the identity of a taker to weak references to the roles attached to it.
    """

    def register(self, role: Role) -> None:
        """
        Index *role* under every taker in its taker chain.

        Indexing under the whole chain, rather than only the immediate taker, lets a
        query from any entity in the chain find the roles layered above it.

        :param role: The role to index.
        """
        for role_taker in role.all_role_takers:
            self._index_role_under_taker(role, role_taker)

    def roles_of(self, role_taker: Any) -> Iterator[Role]:
        """
        Yield the live roles attached to *role_taker* directly or transitively.

        Dead weak references are pruned from the taker's bucket as a side effect of
        reading it.

        :param role_taker: The taker to query.
        """
        bucket = self._roles_by_taker_identity.get(id(role_taker))
        if bucket is None:
            return
        live_roles = [
            role for role in (reference() for reference in bucket) if role is not None
        ]
        bucket[:] = [weakref.ref(role) for role in live_roles]
        yield from live_roles

    def _index_role_under_taker(self, role: Role, role_taker: Any) -> None:
        """
        Add *role* to the bucket of *role_taker*, creating the bucket on first use.

        A finalizer drops the bucket when the taker is garbage collected, so a later
        object that reuses the taker's identity never inherits stale roles.

        :param role: The role to index.
        :param role_taker: The taker to index it under.
        """
        taker_identity = id(role_taker)
        bucket = self._roles_by_taker_identity.get(taker_identity)
        if bucket is None:
            bucket = []
            self._roles_by_taker_identity[taker_identity] = bucket
            weakref.finalize(
                role_taker, self._roles_by_taker_identity.pop, taker_identity, None
            )
        bucket.append(weakref.ref(role))
