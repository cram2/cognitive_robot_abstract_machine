"""Converts LISP-style CRAM plan strings from the ad_graph workflow into PyCRAM PartialDesignators.

Three layers: CRAMParser (tokenise → role dict) → CRAMActionPlan (symbolic IR) →
CRAMToPyCRAMMapper (resolve bodies → PartialDesignator). CRAMToPyCRAMSerializer wires all three.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 – Tokenizer & Recursive S-expression Parser
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Split a CRAM expression string into atomic tokens.

    Parens become individual tokens; whitespace is discarded; all other
    characters are gathered into atom tokens (may include colons like ``:tag``).
    """
    tokens: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in "()":
            tokens.append(c)
            i += 1
        elif c.isspace():
            i += 1
        else:
            j = i
            while j < n and text[j] not in "() \t\n\r":
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


def _parse_expr(tokens: List[str], pos: int) -> Tuple[Any, int]:
    """Recursively parse one S-expression starting at *pos*.

    Returns ``(parsed_value, new_pos)`` where *parsed_value* is either:
    - a ``str``  for an atom token, or
    - a ``list`` for a parenthesised sub-expression.

    Tolerates a missing final ``)`` (graceful degradation for LLM output
    that may be slightly malformed).
    """
    if pos >= len(tokens):
        raise ValueError("Unexpected end of tokens while parsing CRAM expression")

    tok = tokens[pos]

    if tok == "(":
        pos += 1
        items: List[Any] = []
        while pos < len(tokens) and tokens[pos] != ")":
            item, pos = _parse_expr(tokens, pos)
            items.append(item)
        if pos < len(tokens):
            pos += 1  # consume ')'
        # else: tolerate missing closing paren
        return items, pos

    if tok == ")":
        raise ValueError(f"Unexpected ')' at token position {pos}")

    return tok, pos + 1  # atom


def parse_cram(text: str) -> Any:
    """Parse a CRAM S-expression string into a nested Python list.

    The outermost expression is returned. Use ``extract_roles`` to interpret
    the resulting tree.
    """
    tokens = _tokenize(text.strip())
    if not tokens:
        return []
    expr, _ = _parse_expr(tokens, 0)
    return expr


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 – Role Extractor (tree → flat dict)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_entity(node: Any) -> Dict[str, Any]:
    """Recursively extract ``{tag, type, props}`` from a CRAM entity node.

    Handles:
    - ``(:tag NAME (an object ...))``
    - ``(an object (type TYPE k v k v ...))``
    - bare atom strings
    """
    result: Dict[str, Any] = {"tag": None, "type": None, "props": {}}

    if isinstance(node, str):
        result["tag"] = node
        return result

    if not isinstance(node, list) or not node:
        return result

    head = node[0]

    # (:tag NAME DESCRIPTOR)
    if head == ":tag":
        result["tag"] = node[1] if len(node) > 1 else None
        if len(node) > 2:
            inner = _extract_entity(node[2])
            result["type"] = inner.get("type")
            result["props"] = inner.get("props", {})
        return result

    # (an/a DESCRIPTOR_KIND (type TYPE ...) ...)
    if head in ("an", "a") and len(node) >= 2:
        for child in node[2:]:
            if not isinstance(child, list) or not child:
                continue
            if child[0] == "type":
                result["type"] = child[1] if len(child) > 1 else None
                # Remaining atoms in (type T k1 v1 k2 v2 ...) are property pairs
                pairs = child[2:]
                i = 0
                while i + 1 < len(pairs):
                    if isinstance(pairs[i], str) and isinstance(pairs[i + 1], str):
                        result["props"][pairs[i]] = pairs[i + 1]
                        i += 2
                    else:
                        i += 1
            else:
                # Other sub-expressions contribute as flat props
                k = child[0]
                v = child[1] if len(child) > 1 else None
                if isinstance(k, str) and isinstance(v, str):
                    result["props"][k] = v
        return result

    return result


def _extract_location(node: Any) -> Dict[str, Any]:
    """Extract ``{tag, type, props, relation}`` from a CRAM location node.

    Handles ``(a location (on/in/at (:tag ...)))`` patterns.
    """
    result: Dict[str, Any] = {
        "tag": None,
        "type": None,
        "props": {},
        "relation": None,
    }

    if not isinstance(node, list) or not node:
        return result

    head = node[0]

    if head in ("a", "an") and len(node) >= 2 and node[1] == "location":
        for child in node[2:]:
            if not isinstance(child, list) or not child:
                continue
            relation = child[0]
            result["relation"] = relation
            if len(child) > 1:
                entity = _extract_entity(child[1])
                result["tag"] = entity.get("tag")
                result["type"] = entity.get("type")
                result["props"] = entity.get("props", {})
        return result

    # Fallback: treat the node as an entity
    entity = _extract_entity(node)
    result.update(entity)
    return result


def _extract_count(node: Any) -> Dict[str, Any]:
    """Extract ``{amount, unit}`` from a ``(count (unit X)(number Y))`` node."""
    result: Dict[str, Any] = {"amount": None, "unit": None}
    if not isinstance(node, list):
        return result
    for child in node:
        if not isinstance(child, list) or not child:
            continue
        if child[0] == "unit" and len(child) > 1:
            result["unit"] = child[1]
        elif child[0] == "number" and len(child) > 1:
            try:
                result["amount"] = float(child[1])
            except (ValueError, TypeError):
                result["amount"] = child[1]
    return result


def _fill_role(roles: Dict[str, Any], role_key: str, role_val: List[Any]) -> None:
    """Write a single role into the *roles* dict (in-place).

    Shared by ``extract_roles`` and the nested-inside-object pass so that
    both the top-level and LLM-generated nested layouts are handled uniformly.
    """
    if role_key == "type":
        roles["action_type"] = role_val[0] if role_val else None

    elif role_key in ("object", "theme", "patient"):
        # The entity descriptor is the FIRST child; any additional list children
        # may be sub-roles nested inside the object clause (LLM style).
        entity_node = role_val[0] if role_val else None
        roles["object"] = _extract_entity(entity_node)
        # Walk remaining children for nested sub-roles (source, goal, etc.)
        for extra in role_val[1:]:
            if isinstance(extra, list) and extra:
                _fill_role(roles, extra[0], extra[1:])

    elif role_key == "source":
        inner = role_val[0] if len(role_val) == 1 else role_val
        if isinstance(inner, list) and inner and inner[0] in ("a", "an"):
            roles["source"] = _extract_location(inner)
        else:
            roles["source"] = _extract_entity(inner)

    elif role_key in ("goal", "destination", "target"):
        inner = role_val[0] if len(role_val) == 1 else role_val
        if isinstance(inner, list) and inner and inner[0] in ("a", "an"):
            roles["goal"] = _extract_location(inner)
        else:
            roles["goal"] = _extract_entity(inner)

    elif role_key in ("utensil", "instrument"):
        inner = role_val[0] if len(role_val) == 1 else role_val
        roles["utensil"] = _extract_entity(inner)

    elif role_key in ("content", "stuff"):
        inner = role_val[0] if len(role_val) == 1 else role_val
        roles["content"] = _extract_entity(inner)

    elif role_key == "count":
        # Reconstruct the full node for _extract_count (it expects the head)
        roles["count"] = _extract_count([role_key] + list(role_val))

    elif role_key == "technique":
        roles["technique"] = role_val[0] if role_val else None


def extract_roles(tree: Any) -> Dict[str, Any]:
    """Walk a parsed CRAM tree and return a flat dict of semantic roles.

    Handles both standard sibling-level roles and the LLM-generated style
    where sub-roles (e.g. ``source``) are nested inside the ``object`` clause.

    Returned dict keys:
    ``action_type``, ``object``, ``source``, ``goal``, ``utensil``,
    ``content``, ``count``, ``technique``.
    """
    roles: Dict[str, Any] = {
        "action_type": None,
        "object":  {"tag": None, "type": None, "props": {}},
        "source":  {"tag": None, "type": None, "props": {}, "relation": None},
        "goal":    {"tag": None, "type": None, "props": {}, "relation": None},
        "utensil": {"tag": None, "type": None, "props": {}},
        "content": {"tag": None, "type": None, "props": {}},
        "count":   {"amount": None, "unit": None},
        "technique": None,
    }

    if not isinstance(tree, list) or not tree:
        return roles

    head = tree[0]

    # Unwrap (perform (an action ...))
    if head == "perform" and len(tree) > 1:
        return extract_roles(tree[1])

    if head in ("an", "a") and len(tree) >= 2 and tree[1] == "action":
        for child in tree[2:]:
            if not isinstance(child, list) or not child:
                continue
            role_key = child[0]
            # Old-style CRAM: inline (an object ...) / (a location ...) as direct
            # children of the action node — no explicit role wrapper.
            if role_key in ("an", "a") and len(child) >= 2:
                descriptor_kind = child[1]
                if descriptor_kind == "object":
                    # Only fill if object slot is still empty
                    if not roles["object"]["tag"] and not roles["object"]["type"]:
                        roles["object"] = _extract_entity(child)
                elif descriptor_kind == "location":
                    if not roles["goal"]["tag"] and not roles["goal"]["type"]:
                        roles["goal"] = _extract_location(child)
            else:
                _fill_role(roles, role_key, child[1:])

    return roles


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 – Intermediate Representation Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CRAMEntityInfo:
    """Symbolic description of a single entity extracted from a CRAM plan.

    Attributes
    ----------
    tag:
        The ``:tag`` label from the CRAM expression (e.g. ``"cup"``).
        This is the most human-readable identifier and is passed first to the
        body resolver.
    semantic_type:
        The ``(type ...)`` value (e.g. ``"Artifact"``).
        Used as fallback when *tag* is ``None``.
    properties:
        Additional key-value properties found in the CRAM expression
        (e.g. ``{"color": "white", "material": "ceramic"}``).
    """

    tag: Optional[str] = None
    semantic_type: Optional[str] = None
    properties: Dict[str, str] = field(default_factory=dict)

    def best_name(self) -> Optional[str]:
        """Return the most specific identifier available."""
        return self.tag or self.semantic_type

    def __str__(self) -> str:
        return self.best_name() or "unknown"

    def __bool__(self) -> bool:
        return bool(self.tag or self.semantic_type)


@dataclass
class CRAMLocationInfo:
    """Symbolic description of a location entity in a CRAM plan.

    Attributes
    ----------
    tag:
        The ``:tag`` label (e.g. ``"table"``).
    semantic_type:
        The ``(type ...)`` value (e.g. ``"Surface"``).
    properties:
        Additional key-value properties.
    relation:
        Spatial relation keyword (``"on"``, ``"in"``, ``"at"``).
    """

    tag: Optional[str] = None
    semantic_type: Optional[str] = None
    properties: Dict[str, str] = field(default_factory=dict)
    relation: Optional[str] = None

    def best_name(self) -> Optional[str]:
        return self.tag or self.semantic_type

    def __str__(self) -> str:
        base = self.best_name() or "unknown"
        return f"{self.relation}:{base}" if self.relation else base

    def __bool__(self) -> bool:
        return bool(self.tag or self.semantic_type)


@dataclass
class CRAMActionPlan:
    """Intermediate symbolic representation of a single parsed CRAM plan.

    All fields are symbolic strings/dicts extracted from the LISP syntax.
    No PyCRAM objects are referenced here — this dataclass is fully
    standalone and serialisable.

    Attributes
    ----------
    action_type:
        CRAM action type string (e.g. ``"PickingUp"``, ``"Cutting"``).
    raw_cram:
        The original CRAM expression string (for debugging / round-trips).
    object:
        The primary object to act on.
    source:
        The location where the object currently resides (pick-up context).
    goal:
        The target/destination location (place context).
    utensil:
        A tool or instrument involved in the action (e.g. knife for cutting).
    content:
        Substance/ingredient being moved (for filling, pouring, etc.).
    amount:
        Numeric quantity (from ``(count (number ...))``.
    unit:
        Unit of measurement (from ``(count (unit ...)``).
    technique:
        Manner/technique string (e.g. ``"Slicing"``, ``"Spiral Mixing"``).
    """

    action_type: str
    raw_cram: str = ""

    object: CRAMEntityInfo = field(default_factory=CRAMEntityInfo)
    source: CRAMLocationInfo = field(default_factory=CRAMLocationInfo)
    goal: CRAMLocationInfo = field(default_factory=CRAMLocationInfo)
    utensil: CRAMEntityInfo = field(default_factory=CRAMEntityInfo)
    content: CRAMEntityInfo = field(default_factory=CRAMEntityInfo)

    amount: Optional[float] = None
    unit: Optional[str] = None
    technique: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 – CRAM Parser (Section 1 + 2 wired together)
# ─────────────────────────────────────────────────────────────────────────────

class CRAMParser:
    """Parses a CRAM S-expression string into a ``CRAMActionPlan``.

    This is a pure-Python parser — no PyCRAM dependency.
    """

    def parse(self, cram_string: str) -> CRAMActionPlan:
        """Parse *cram_string* and return an intermediate ``CRAMActionPlan``.

        Parameters
        ----------
        cram_string:
            A LISP-style CRAM plan string such as::

                (an action (type PickingUp)
                  (object (:tag cup (an object (type Artifact color white))))
                  (source (a location (on (:tag table (an object (type Surface)))))))

        Raises
        ------
        ValueError
            If the top-level action type cannot be determined.
        """
        tree = parse_cram(cram_string)
        roles = extract_roles(tree)

        action_type = roles.get("action_type")
        if not action_type:
            raise ValueError(
                f"Could not extract action_type from CRAM expression: {cram_string!r}"
            )

        def _to_entity(d: Dict[str, Any]) -> CRAMEntityInfo:
            return CRAMEntityInfo(
                tag=d.get("tag"),
                semantic_type=d.get("type"),
                properties=d.get("props", {}),
            )

        def _to_location(d: Dict[str, Any]) -> CRAMLocationInfo:
            return CRAMLocationInfo(
                tag=d.get("tag"),
                semantic_type=d.get("type"),
                properties=d.get("props", {}),
                relation=d.get("relation"),
            )

        return CRAMActionPlan(
            action_type=action_type,
            raw_cram=cram_string,
            object=_to_entity(roles["object"]),
            source=_to_location(roles["source"]),
            goal=_to_location(roles["goal"]),
            utensil=_to_entity(roles["utensil"]),
            content=_to_entity(roles["content"]),
            amount=roles["count"].get("amount"),
            unit=roles["count"].get("unit"),
            technique=roles.get("technique"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 – PyCRAM Mapper
# ─────────────────────────────────────────────────────────────────────────────

# Type alias: a callable that receives a CRAMEntityInfo and returns a PyCRAM Body
BodyResolver = Callable[[CRAMEntityInfo], Any]


class CRAMToPyCRAMMapper:
    """Maps a ``CRAMActionPlan`` to a PyCRAM ``PartialDesignator``.

    Requires a *body_resolver* callable to bind symbolic entity info to
    live PyCRAM ``Body`` objects at call time — the mapper itself remains
    decoupled from the PyCRAM world.

    Notes
    -----
    PyCRAM classes are imported lazily inside ``map()`` so that this module
    can be imported even without a PyCRAM installation (for parse-only use).
    """

    # Normalised CRAM action_type string → PyCRAM class name.
    # Keys must be pre-normalised (lowercase, no spaces / hyphens / underscores)
    # because _normalise_action_type() strips all of those before the lookup.
    _ACTION_MAP: Dict[str, str] = {
        # ── Core: pick-up / lift ───────────────────────────────────────────
        "pickingup":           "PickUpAction",
        "pickup":              "PickUpAction",
        "lifting":             "PickUpAction",   # best available core match
        "liftobject":          "PickUpAction",
        "takeobject":          "PickUpAction",   # take = pick up from location
        "removeobject":        "PickUpAction",   # remove from surface
        # ── Core: place ───────────────────────────────────────────────────
        "placing":             "PlaceAction",
        "place":               "PlaceAction",
        "putobject":           "PlaceAction",
        # ── Core: open / close ────────────────────────────────────────────
        "opening":             "OpenAction",
        "openobject":          "OpenAction",
        "shutting":            "CloseAction",
        "closing":             "CloseAction",
        "closeobject":         "CloseAction",
        # ── Core: navigation / perception ─────────────────────────────────
        "navigating":          "NavigateAction",
        "navigate":            "NavigateAction",
        "lookat":              "LookAtAction",
        "look":                "LookAtAction",
        "detecting":           "DetectAction",
        "detect":              "DetectAction",
        # ── Core: robot body ──────────────────────────────────────────────
        "parkarms":            "ParkArmsAction",
        "park":                "ParkArmsAction",
        # ── Composite: tool-based ─────────────────────────────────────────
        "cutting":             "CuttingAction",
        "cutobject":           "CuttingAction",
        "mixing":              "MixingAction",
        "mix":                 "MixingAction",
        "usewhisk":            "MixingAction",   # stirring with a whisk
        "usespoon":            "MixingAction",   # spooning maps to mixing
        "stir":                "MixingAction",
        "stirring":            "MixingAction",
        "pouring":             "PouringAction",
        "pour":                "PouringAction",
        "pourfromplicejar":    "PouringAction",
        "pourfromspicejar":    "PouringAction",
        "operateatap":         "PouringAction",  # tap → controlled pour
        # ── Composite: transport ──────────────────────────────────────────
        "transporting":        "TransportAction",
        "transport":           "TransportAction",
        "pickandplace":        "PickAndPlaceAction",
        "moveandpickup":       "MoveAndPickUpAction",
        "moveandplace":        "MoveAndPlaceAction",
        "efficienttransport":  "EfficientTransportAction",
        # ── Composite: search / face ──────────────────────────────────────
        "search":              "SearchAction",
        "searching":           "SearchAction",
        "faceat":              "FaceAtAction",
        "face":                "FaceAtAction",
    }

    # Maps PyCRAM class name → import path (relative to the pycram package)
    _CLASS_MODULE: Dict[str, str] = {
        # Core
        "PickUpAction":             "pycram.robot_plans.actions.core.pick_up",
        "PlaceAction":              "pycram.robot_plans.actions.core.placing",
        "OpenAction":               "pycram.robot_plans.actions.core.container",
        "CloseAction":              "pycram.robot_plans.actions.core.container",
        "NavigateAction":           "pycram.robot_plans.actions.core.navigation",
        "LookAtAction":             "pycram.robot_plans.actions.core.navigation",
        "DetectAction":             "pycram.robot_plans.actions.core.misc",
        "ParkArmsAction":           "pycram.robot_plans.actions.core.robot_body",
        # Composite – tool-based
        "CuttingAction":            "pycram.robot_plans.actions.composite.tool_based",
        "MixingAction":             "pycram.robot_plans.actions.composite.tool_based",
        "PouringAction":            "pycram.robot_plans.actions.composite.tool_based",
        # Composite – transport
        "TransportAction":          "pycram.robot_plans.actions.composite.transporting",
        "PickAndPlaceAction":       "pycram.robot_plans.actions.composite.transporting",
        "MoveAndPickUpAction":      "pycram.robot_plans.actions.composite.transporting",
        "MoveAndPlaceAction":       "pycram.robot_plans.actions.composite.transporting",
        "EfficientTransportAction": "pycram.robot_plans.actions.composite.transporting",
        # Composite – search / face
        "SearchAction":             "pycram.robot_plans.actions.composite.searching",
        "FaceAtAction":             "pycram.robot_plans.actions.composite.facing",
    }

    def _normalise_action_type(self, action_type: str) -> str:
        return action_type.lower().replace(" ", "").replace("-", "").replace("_", "")

    def _load_pycram_class(self, class_name: str) -> type:
        """Lazily import and return the PyCRAM action class by name."""
        module_path = self._CLASS_MODULE.get(class_name)
        if not module_path:
            raise ImportError(f"No module path registered for PyCRAM class '{class_name}'")
        import importlib
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name, None)
        if cls is None:
            raise ImportError(
                f"Class '{class_name}' not found in module '{module_path}'"
            )
        return cls

    def map(
        self,
        plan: CRAMActionPlan,
        body_resolver: BodyResolver,
        arm: Any = None,
        grasp_description: Any = None,
        approach_from: Any = None,
    ) -> Any:
        """Convert *plan* to a PyCRAM ``PartialDesignator``.

        Parameters
        ----------
        plan:
            The intermediate ``CRAMActionPlan`` produced by ``CRAMParser``.
        body_resolver:
            Callable ``(CRAMEntityInfo) -> Body``.  Maps a symbolic entity
            description to a live PyCRAM ``Body`` world object.
            Returning ``None`` is allowed for optional roles.
        arm:
            PyCRAM ``Arms`` enum value to use for arm-based actions.
            Defaults to ``Arms.RIGHT`` if ``None`` and pycram is available.
        grasp_description:
            PyCRAM ``GraspDescription`` to use.  When ``None`` the action's
            own defaults are used (via the ``PartialDesignator`` mechanism).

        Returns
        -------
        PartialDesignator
            A PyCRAM ``PartialDesignator[ActionDescription]`` ready for
            ``.resolve()`` and ``.perform()``.

        Raises
        ------
        ValueError
            If the CRAM ``action_type`` has no registered PyCRAM mapping.
        ImportError
            If the required PyCRAM module cannot be imported.
        """
        key = self._normalise_action_type(plan.action_type)
        class_name = self._ACTION_MAP.get(key)
        if class_name is None:
            raise ValueError(
                f"No PyCRAM mapping for CRAM action_type '{plan.action_type}'. "
                f"Registered types: {sorted(self._ACTION_MAP.keys())}"
            )

        pycram_cls = self._load_pycram_class(class_name)

        # Resolve default arm if not provided
        if arm is None:
            try:
                from pycram.datastructures.enums import Arms
                arm = Arms.RIGHT
            except ImportError:
                pass  # arm stays None; user must provide it

        # Resolve Body objects for roles that require them.
        # source_entity is used by pour/tap-style plans where the primary actor
        # lives in the source role rather than the object role.
        source_entity = CRAMEntityInfo(
            tag=plan.source.tag,
            semantic_type=plan.source.semantic_type,
            properties=plan.source.properties,
        )
        obj_body    = body_resolver(plan.object)    if plan.object    else None
        tool_body   = body_resolver(plan.utensil)   if plan.utensil   else None
        source_body = body_resolver(source_entity)  if source_entity  else None
        goal_body   = body_resolver(
            CRAMEntityInfo(
                tag=plan.goal.tag,
                semantic_type=plan.goal.semantic_type,
                properties=plan.goal.properties,
            )
        ) if plan.goal else None

        return self._build_partial(
            class_name, pycram_cls, plan,
            obj_body, tool_body, source_body, goal_body,
            arm, grasp_description,
            approach_from=approach_from,
        )

    @staticmethod
    def _body_to_pose(body: Any) -> Any:
        """Convert a Body to a PoseStamped using the body's global frame origin."""
        try:
            from pycram.datastructures.pose import PoseStamped
            return PoseStamped.from_spatial_type(body.global_pose)
        except Exception:
            return body

    @staticmethod
    def _surface_pose_for_placement(
        surface_body: Any,
        obj_body: Any = None,
        approach_from: Any = None,
    ) -> Any:
        """Compute the placement target pose on surface_body's top face + obj half-height.

        For PlaceAction, the target_location must be the pose where the placed
        object's *center* should end up.  ``_body_to_pose`` returns the surface
        body's frame origin (often at floor/mesh-center level), which is wrong.
        Here we use the world-frame AABB max_z as the surface height, then add
        half the placed object's local height so the object rests on the surface.

        When *approach_from* (a PoseStamped of the robot's nav pose) is provided,
        the x,y target is the nearest reachable point on the surface to the robot
        rather than the surface center.  This avoids IK failures when the robot
        approaches from one side of a large table.
        """
        try:
            from pycram.datastructures.pose import PoseStamped

            world = surface_body._world
            world_root = world.root

            # Bounding box of the surface in world frame
            surface_bb = (
                surface_body.collision
                .as_bounding_box_collection_in_frame(world_root)
                .bounding_box()
            )
            surface_z = surface_bb.max_z  # top face z in world frame

            # x, y: when approach_from is given, use the robot's position clamped
            # to the surface BB (with a small inward margin) so the placement target
            # is on the near edge reachable by the arm.  Otherwise fall back to the
            # surface body's own reference point (table centre).
            _EDGE_MARGIN = 0.05  # stay 5 cm inside the BB edge
            if approach_from is not None:
                try:
                    robot_x = approach_from.pose.position.x
                    robot_y = approach_from.pose.position.y
                    surface_x = max(
                        surface_bb.min_x + _EDGE_MARGIN,
                        min(surface_bb.max_x - _EDGE_MARGIN, robot_x),
                    )
                    surface_y = max(
                        surface_bb.min_y + _EDGE_MARGIN,
                        min(surface_bb.max_y - _EDGE_MARGIN, robot_y),
                    )
                except Exception:
                    approach_from = None  # fall through to centre logic

            if approach_from is None:
                try:
                    gp = PoseStamped.from_spatial_type(surface_body.global_pose)
                    raw_x = gp.pose.position.x
                    raw_y = gp.pose.position.y
                except Exception:
                    raw_x = (surface_bb.min_x + surface_bb.max_x) / 2.0
                    raw_y = (surface_bb.min_y + surface_bb.max_y) / 2.0

                surface_x = max(surface_bb.min_x, min(surface_bb.max_x, raw_x))
                surface_y = max(surface_bb.min_y, min(surface_bb.max_y, raw_y))

            # Half-height of the object being placed (local frame, attachment-invariant)
            obj_half_z = 0.0
            if obj_body is not None:
                try:
                    obj_bb = (
                        obj_body.collision
                        .as_bounding_box_collection_in_frame(obj_body)
                        .bounding_box()
                    )
                    obj_half_z = (obj_bb.max_z - obj_bb.min_z) / 2.0
                except Exception:
                    pass

            placement_z = surface_z + obj_half_z

            logger.info(
                "_surface_pose_for_placement: surface=%r top_z=%.3f obj_half_z=%.3f → "
                "pose=(%.3f, %.3f, %.3f) [%s]",
                getattr(surface_body, "name", surface_body),
                surface_z, obj_half_z,
                surface_x, surface_y, placement_z,
                "near-edge" if approach_from is not None else "centre",
            )

            return PoseStamped.from_list(
                [surface_x, surface_y, placement_z],
                [0.0, 0.0, 0.0, 1.0],
                world_root,
            )
        except Exception as exc:
            logger.warning(
                "_surface_pose_for_placement failed (%s); falling back to global_pose", exc
            )
            return CRAMToPyCRAMMapper._body_to_pose(surface_body)

    def _build_partial(
        self,
        class_name: str,
        pycram_cls: type,
        plan: "CRAMActionPlan",
        obj_body: Any,
        tool_body: Any,
        source_body: Any,
        goal_body: Any,
        arm: Any,
        grasp_description: Any,
        approach_from: Any = None,
    ) -> Any:
        """Dispatch to the correct ``.description()`` factory for each action class."""

        # ── Core: PickUpAction ─────────────────────────────────────────────
        if class_name == "PickUpAction":
            return pycram_cls.description(
                object_designator=obj_body,
                arm=arm,
                grasp_description=grasp_description,
            )

        # ── Core: PlaceAction ──────────────────────────────────────────────
        if class_name == "PlaceAction":
            if goal_body is None:
                raise ValueError(
                    f"PlaceAction: could not resolve the target/destination location — "
                    f"no matching body found in the world for goal entity {plan.goal!r}. "
                    f"Check that the CRAM plan contains a 'target', 'goal', or 'destination' "
                    f"role with a name that matches an object in the simulation world."
                )
            # Use surface AABB top face + obj half-height so the arm reaches
            # the correct z (not the surface body's frame origin).  Pass
            # approach_from so placement targets the near edge, not the centre.
            target_location = self._surface_pose_for_placement(
                goal_body, obj_body, approach_from=approach_from
            )
            return pycram_cls.description(
                object_designator=obj_body,
                target_location=target_location,
                arm=arm,
            )

        # ── Core: OpenAction / CloseAction ────────────────────────────────
        if class_name in ("OpenAction", "CloseAction"):
            return pycram_cls.description(
                object_designator_description=obj_body,
                arm=arm,
            )

        # ── Core: NavigateAction ──────────────────────────────────────────
        if class_name == "NavigateAction":
            target_body = goal_body or obj_body
            target_location = self._body_to_pose(target_body) if target_body else None
            return pycram_cls.description(target_location=target_location)

        # ── Core: LookAtAction ────────────────────────────────────────────
        if class_name == "LookAtAction":
            target = self._body_to_pose(obj_body or goal_body) if (obj_body or goal_body) else None
            return pycram_cls.description(target=target)

        # ── Core: DetectAction ────────────────────────────────────────────
        if class_name == "DetectAction":
            try:
                from pycram.datastructures.enums import DetectionTechnique
                technique = DetectionTechnique.TYPES
            except ImportError:
                technique = None
            return pycram_cls.description(technique=technique)

        # ── Core: ParkArmsAction ──────────────────────────────────────────
        if class_name == "ParkArmsAction":
            return pycram_cls.description(arm=arm)

        # ── Composite: CuttingAction ──────────────────────────────────────
        if class_name == "CuttingAction":
            kwargs: Dict[str, Any] = dict(
                object_=obj_body,
                tool=tool_body,
                arm=arm,
                technique=plan.technique,
            )
            if plan.amount is not None:
                kwargs["slice_thickness"] = plan.amount
            return pycram_cls.description(**kwargs)

        # ── Composite: MixingAction ───────────────────────────────────────
        if class_name == "MixingAction":
            return pycram_cls.description(
                object_=obj_body,
                tool=tool_body,
                arm=arm,
                technique=plan.technique,
            )

        # ── Composite: PouringAction ──────────────────────────────────────
        # Pouring plans use (source ...) for the liquid container; fall back
        # to source_body when object_ is not set.
        if class_name == "PouringAction":
            actor = obj_body or source_body
            return pycram_cls.description(
                object_=actor,
                tool=tool_body,
                arm=arm,
                technique=plan.technique,
            )

        # ── Composite: TransportAction ────────────────────────────────────
        if class_name in ("TransportAction", "EfficientTransportAction"):
            target_location = self._body_to_pose(goal_body) if goal_body else None
            return pycram_cls.description(
                object_designator=obj_body,
                target_location=target_location,
                arm=arm,
            )

        # ── Composite: PickAndPlaceAction ─────────────────────────────────
        if class_name == "PickAndPlaceAction":
            target_location = self._body_to_pose(goal_body) if goal_body else None
            return pycram_cls.description(
                object_designator=obj_body,
                target_location=target_location,
                arm=arm,
                grasp_description=grasp_description,
            )

        # ── Composite: MoveAndPickUpAction ────────────────────────────────
        if class_name == "MoveAndPickUpAction":
            standing_position = self._body_to_pose(source_body) if source_body else None
            return pycram_cls.description(
                standing_position=standing_position,
                object_designator=obj_body,
                arm=arm,
                grasp_description=grasp_description,
            )

        # ── Composite: MoveAndPlaceAction ─────────────────────────────────
        if class_name == "MoveAndPlaceAction":
            standing_position = self._body_to_pose(source_body) if source_body else None
            target_location   = self._body_to_pose(goal_body)   if goal_body   else None
            return pycram_cls.description(
                standing_position=standing_position,
                object_designator=obj_body,
                target_location=target_location,
                arm=arm,
            )

        # ── Composite: SearchAction ───────────────────────────────────────
        if class_name == "SearchAction":
            target_location = self._body_to_pose(goal_body or obj_body) if (goal_body or obj_body) else None
            return pycram_cls.description(
                target_location=target_location,
                object_type=None,   # caller can inject a SemanticAnnotation type
            )

        # ── Composite: FaceAtAction ───────────────────────────────────────
        if class_name == "FaceAtAction":
            pose = self._body_to_pose(obj_body or goal_body) if (obj_body or goal_body) else None
            return pycram_cls.description(pose=pose)

        raise NotImplementedError(
            f"_build_partial not implemented for PyCRAM class '{class_name}'"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 – Top-level Serializer (convenience API)
# ─────────────────────────────────────────────────────────────────────────────

class CRAMToPyCRAMSerializer:
    """Converts CRAM S-expression strings to PyCRAM action objects.

    Wraps ``CRAMParser`` and ``CRAMToPyCRAMMapper`` into a single API.

    Typical workflow
    ----------------
    ::

        ser = CRAMToPyCRAMSerializer()

        # Parse only (no PyCRAM needed)
        plan: CRAMActionPlan = ser.parse(cram_string)

        # Map to PyCRAM PartialDesignator (requires PyCRAM + body_resolver)
        partial = ser.to_partial_designator(plan, body_resolver=my_resolver)

        # One-shot
        partial = ser.serialize(cram_string, body_resolver=my_resolver)
    """

    def __init__(self) -> None:
        self._parser = CRAMParser()
        self._mapper = CRAMToPyCRAMMapper()

    # ── Public API ──────────────────────────────────────────────────────────

    def parse(self, cram_string: str) -> CRAMActionPlan:
        """Parse *cram_string* → ``CRAMActionPlan`` (no PyCRAM required).

        Parameters
        ----------
        cram_string:
            A LISP-style CRAM plan string produced by the llmr
            ``ad_graph`` workflow.

        Returns
        -------
        CRAMActionPlan
            Symbolic intermediate representation.
        """
        return self._parser.parse(cram_string)

    def to_partial_designator(
        self,
        plan: CRAMActionPlan,
        body_resolver: BodyResolver,
        arm: Any = None,
        grasp_description: Any = None,
        approach_from: Any = None,
    ) -> Any:
        """Map *plan* → PyCRAM ``PartialDesignator`` using *body_resolver*.

        Parameters
        ----------
        plan:
            A ``CRAMActionPlan`` previously produced by ``parse()``.
        body_resolver:
            ``Callable[[CRAMEntityInfo], Body]`` — maps a symbolic entity to
            the corresponding live PyCRAM ``Body`` world object.
        arm:
            PyCRAM ``Arms`` enum value.  Defaults to ``Arms.RIGHT``.
        grasp_description:
            PyCRAM ``GraspDescription``.  ``None`` uses action defaults.

        Returns
        -------
        PartialDesignator
            Ready for ``.resolve()`` / ``.perform()``.
        """
        return self._mapper.map(plan, body_resolver, arm, grasp_description, approach_from=approach_from)

    def serialize(
        self,
        cram_string: str,
        body_resolver: BodyResolver,
        arm: Any = None,
        grasp_description: Any = None,
    ) -> Any:
        """One-shot: parse *cram_string* and map to a PyCRAM ``PartialDesignator``.

        Parameters
        ----------
        cram_string:
            CRAM S-expression string.
        body_resolver:
            ``Callable[[CRAMEntityInfo], Body]``.
        arm:
            PyCRAM ``Arms`` enum value.
        grasp_description:
            PyCRAM ``GraspDescription``.

        Returns
        -------
        PartialDesignator
        """
        plan = self.parse(cram_string)
        return self.to_partial_designator(plan, body_resolver, arm, grasp_description)

    def serialize_many(
        self,
        cram_strings: List[str],
        body_resolver: BodyResolver,
        arm: Any = None,
        grasp_description: Any = None,
    ) -> List[Any]:
        """Serialize a list of CRAM strings, returning one PartialDesignator per string.

        Errors in individual plans are caught and re-raised with context.
        """
        results = []
        for i, cram_str in enumerate(cram_strings):
            try:
                results.append(
                    self.serialize(cram_str, body_resolver, arm, grasp_description)
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to serialize CRAM plan at index {i}: {cram_str!r}"
                ) from exc
        return results

    # ── Utility: parse without mapping (safe, no pycram import) ────────────

    @staticmethod
    def parse_raw(cram_string: str) -> Any:
        """Return the raw nested-list tree for *cram_string* (debugging aid)."""
        return parse_cram(cram_string)

    @staticmethod
    def extract_roles_from(cram_string: str) -> Dict[str, Any]:
        """Return the flat roles dict extracted from *cram_string* (debugging aid)."""
        return extract_roles(parse_cram(cram_string))