"""Gateway package: the only modules that touch krrood directly.

Other llmr modules import plain data structures from here and stay krrood-free.

  introspect   — classify action dataclass fields into :class:`FieldKind`.
  world_reader — read SymbolGraph contents and resolve Symbol classes by name.
  match_reader — snapshot krrood Match expressions into :class:`MatchData` / :class:`MatchSlot`.
"""

from llmr.bridge.introspect import (
    ActionSchema,
    FieldKind,
    FieldSpec,
    PycramIntrospector,
    introspect,
)
from llmr.bridge.match_reader import (
    MatchData,
    MatchSlot,
    finalize_match,
    read_match,
    required_match,
    unresolved_required_fields,
    write_slot_value,
)
from llmr.bridge.world_reader import (
    WorldSerializationOptions,
    body_bounding_box,
    body_display_name,
    body_xyz,
    get_instances,
    resolve_symbol_class,
    serialize_world_from_symbol_graph,
)

__all__ = [
    # introspect
    "ActionSchema",
    "FieldKind",
    "FieldSpec",
    "PycramIntrospector",
    "introspect",
    # match_reader
    "MatchData",
    "MatchSlot",
    "finalize_match",
    "read_match",
    "required_match",
    "unresolved_required_fields",
    "write_slot_value",
    # world_reader
    "WorldSerializationOptions",
    "body_bounding_box",
    "body_display_name",
    "body_xyz",
    "get_instances",
    "resolve_symbol_class",
    "serialize_world_from_symbol_graph",
]
