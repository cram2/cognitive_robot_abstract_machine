import logging

from krrood.ripple_down_rules.datastructures.case import Case
from krrood.ripple_down_rules.datastructures.dataclasses import CaseQuery
from krrood.ripple_down_rules.utils import SubclassJSONSerializer

try:
    import graphviz
except ImportError:
    graphviz = None
    logging.debug("Graphviz is not installed")


def is_simple(obj):
    return isinstance(obj, (int, float, str, bool, type(None)))


def get_colored_value(value):
    if isinstance(value, str):
        color = "#A31515"  # red for strings
        val = repr(value)
    elif isinstance(value, (int, float)):
        color = "#098658"  # green for numbers
        val = str(value)
    elif isinstance(value, bool):
        color = "#0000FF"  # blue for booleans
        val = str(value)
    elif value is None:
        color = "#808080"  # gray for None
        val = "None"
    else:
        color = "#000000"  # fallback
        val = str(value)
    val = (
        val.replace("class ", "")
        .replace("'", "")
        .replace("<", "")
        .replace(">", "")
        .replace("?", "")
        .replace("\n", " ")
    )
    val = val[:50] + "..." if len(val) > 50 else val
    return f'<FONT COLOR="{color}">{val}</FONT>'


def generate_object_graph(
    obj,
    name="root",
    seen=None,
    graph=None,
    current_depth=0,
    max_depth=3,
    chain_name=None,
    included_attrs=None,
    iterable_limit=3,
):
    if seen is None:
        seen = set()
    if graph is None:
        graph = graphviz.Digraph(format="svg")
        graph.attr("node", shape="plaintext")

    obj_id = id(obj)
    if obj_id in seen or current_depth > max_depth:
        return graph
    seen.add(obj_id)

    # Build HTML table label for this object node
    type_name = type(obj).__name__ if not isinstance(obj, type) else obj.__name__
    rows = [f"<TR><TD><B>{name}</B></TD><TD><I>{type_name}</I></TD></TR>"]

    # We'll keep track of non-simple attrs to add edges later
    non_simple_attrs = []

    if isinstance(obj, (list, tuple, set, dict)):
        items = obj.items() if isinstance(obj, dict) else enumerate(obj)
        i = 0
        for idx, item in items:
            if idx == "scope":
                continue
            if i >= iterable_limit:
                rows.append(
                    f'<TR><TD ALIGN="LEFT" PORT="{idx}">...</TD><TD ALIGN="LEFT">...</TD></TR>'
                )
                break
            # Represent items as attr = index + type (for the label)
            if is_simple(item):
                val_colored = get_colored_value(item)
                rows.append(
                    f'<TR><TD ALIGN="LEFT" PORT="{idx}">[{idx}]</TD><TD ALIGN="LEFT">{val_colored}</TD></TR>'
                )
            else:
                type_name = (
                    type(item).__name__ if not isinstance(item, type) else item.__name__
                )
                rows.append(
                    f'<TR><TD ALIGN="LEFT" PORT="{idx}">[{idx}]</TD><TD ALIGN="LEFT"><I>{type_name}</I></TD></TR>'
                )
                non_simple_attrs.append((str(idx), item))
            i += 1

    else:
        for attr in dir(obj):
            if attr.startswith("_"):
                continue
            if isinstance(obj, CaseQuery) and attr == "scope":
                continue
            if isinstance(obj, Case) and attr in ["data"]:
                continue
            if isinstance(obj, SubclassJSONSerializer) and attr == "data_class_refs":
                continue
            value = getattr(obj, attr)
            if callable(value):
                continue
            if is_simple(value):
                val_colored = get_colored_value(value)
                rows.append(
                    f'<TR><TD ALIGN="LEFT" PORT="{attr}">{attr}</TD><TD ALIGN="LEFT">{val_colored}</TD></TR>'
                )
            else:
                type_name = (
                    type(value).__name__
                    if not isinstance(value, type)
                    else value.__name__
                )
                # Show just name and type inside the node
                rows.append(
                    f'<TR><TD ALIGN="LEFT" PORT="{attr}">{attr}</TD><TD ALIGN="LEFT"><I>{type_name}</I></TD></TR>'
                )
                non_simple_attrs.append((attr, value))

    label = f"""<
        <TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
        {''.join(rows)}
        </TABLE>
    >"""

    graph.node(str(obj_id), label)
    chain_name = chain_name if chain_name is not None else name

    # Add edges from this node to non-simple attribute nodes
    for attr, value in non_simple_attrs:
        if attr.startswith("_"):
            continue
        if attr == "scope":
            continue
        if callable(value):
            continue
        attr_chain_name = f"{chain_name}.{attr}"
        if included_attrs is not None and attr_chain_name not in included_attrs:
            continue
        generate_object_graph(
            value,
            attr,
            seen,
            graph,
            current_depth + 1,
            max_depth,
            chain_name=f"{chain_name}.{attr}",
            included_attrs=included_attrs,
        )
        # Edge from this object's attribute port to nested object's node
        graph.edge(f"{obj_id}:{attr}", str(id(value)), label=attr)

    return graph
