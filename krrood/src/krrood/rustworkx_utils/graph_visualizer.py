from __future__ import annotations

from collections import defaultdict
from textwrap import fill
from typing_extensions import Optional, List, Dict, Tuple, TYPE_CHECKING, Any

import igraph as ig
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path

from krrood.rustworkx_utils.utils import ColorLegend

if TYPE_CHECKING:
    from krrood.rustworkx_utils.rxnode import RWXNode


class GraphVisualizer:
    """Responsible for rendering an RWXNode graph with a layered top-to-bottom layout.

    The class encapsulates the visualization pipeline in small reusable steps while
    preserving the previous behavior.
    """

    def __init__(
        self,
        node: RWXNode,
        figsize=(80, 80),
        node_size=7000,
        font_size=25,
        spacing_x: float = 4,
        spacing_y: float = 4,
        curve_scale: float = 0.5,
        layout: str = "tidy",
        edge_style: str = "orthogonal",
        label_max_chars_per_line: Optional[int] = 13,
        orthogonal_spline_ratio_threshold: float = 3,
        orthogonal_crossings_threshold: int = 4,
        orthogonal_min_length_threshold: Optional[float] = 4,
        filename: str = "pdf_graph.pdf",
        title: str = "Directed Query Graph (Top to Bottom)",
    ):
        self.node = node
        self.params = dict(
            figsize=figsize,
            node_size=node_size,
            font_size=font_size,
            spacing_x=spacing_x,
            spacing_y=spacing_y,
            curve_scale=curve_scale,
            layout=layout,
            edge_style=edge_style,
            label_max_chars_per_line=label_max_chars_per_line,
            orthogonal_spline_ratio_threshold=orthogonal_spline_ratio_threshold,
            orthogonal_crossings_threshold=orthogonal_crossings_threshold,
            orthogonal_min_length_threshold=orthogonal_min_length_threshold,
            filename=filename,
            title=title,
        )
        self.ctx: Dict[str, Any] = {}

    def render(self):
        self._check_dependencies()
        self._build_rooted_subgraph()
        self._layer_nodes()
        self._order_layers_barycentric()
        self._assign_coordinates()
        if self.ctx.get("empty"):
            return self.ctx["fig"], self.ctx["ax"]
        self._finalize_positions_and_edges()
        self._auto_figsize()
        fig, ax = self._setup_figure()
        self._wrap_labels_and_sizes()
        self._compute_pixel_and_obstacles(fig, ax)
        self._adjust_limits_and_center(ax)
        self._draw_wrap_boxes(ax)
        self._draw_edges(ax)
        self._draw_nodes_and_labels(ax)
        self._draw_legend(fig, ax)
        self._style_and_save(fig, ax)
        return fig, ax

    # ---- Steps ----
    def _check_dependencies(self):
        if not ig or not mpl or not fill:
            raise RuntimeError(
                "igraph, matplotlib, textwrap must be installed to visualize the graph."
            )
        # Ensure a non-interactive backend for headless environments
        try:
            mpl.use("Agg")
        except Exception:
            pass

    def _build_rooted_subgraph(self):
        root = self.node.root
        sub_nodes = [root] + root.descendants
        sub_id_set = {n.id for n in sub_nodes}
        id_to_node = {n.id: n for n in sub_nodes}
        rx_edges_all = list(self.node.graph.edge_list())
        rx_edges = [
            (u, v) for (u, v) in rx_edges_all if u in sub_id_set and v in sub_id_set
        ]
        preds: Dict[int, List[int]] = defaultdict(list)
        succs: Dict[int, List[int]] = defaultdict(list)
        for u, v in rx_edges:
            preds[v].append(u)
            succs[u].append(v)
        for nid in sub_id_set:
            preds.setdefault(nid, [])
            succs.setdefault(nid, [])
        self.ctx.update(
            dict(
                root=root,
                sub_nodes=sub_nodes,
                sub_id_set=sub_id_set,
                id_to_node=id_to_node,
                rx_edges=rx_edges,
                preds=preds,
                succs=succs,
            )
        )

    def _layer_nodes(self):
        root = self.ctx["root"]
        succs = self.ctx["succs"]
        sub_id_set = self.ctx["sub_id_set"]
        layer: Dict[int, int] = {root.id: 0}
        from collections import deque

        dq = deque([root.id])
        while dq:
            u = dq.popleft()
            lu = layer.get(u, 0)
            for v in succs[u]:
                new_l = lu + 1
                if layer.get(v, -1) < new_l:
                    layer[v] = new_l
                    dq.append(v)
        for nid in sub_id_set:
            layer.setdefault(nid, 0)
        max_layer = max(layer.values()) if layer else 0
        layers: List[List[int]] = (
            [[] for _ in range(max_layer + 1)] if max_layer >= 0 else []
        )
        for nid, l in layer.items():
            layers[l].append(nid)
        for l in range(len(layers)):
            layers[l].sort()
        self.ctx.update(dict(layer=layer, layers=layers, max_layer=max_layer))

    def _order_layers_barycentric(self):
        layers = self.ctx["layers"]
        preds = self.ctx["preds"]
        succs = self.ctx["succs"]

        def compute_order_index(layer_nodes: List[int]) -> Dict[int, int]:
            return {nid: idx for idx, nid in enumerate(layer_nodes)}

        def sort_by_barycenter(
            current_layer: List[int], reference_layer: List[int], use_preds: bool
        ) -> List[int]:
            ref_pos = compute_order_index(reference_layer)

            def bary(nid: int) -> float:
                neighbors = preds[nid] if use_preds else succs[nid]
                neighbors = [w for w in neighbors if w in ref_pos]
                if not neighbors:
                    return float(ref_pos.get(nid, 0))
                return float(np.mean([ref_pos[w] for w in neighbors]))

            return sorted(current_layer, key=lambda nid: (bary(nid), nid))

        for _ in range(3):
            for l in range(1, len(layers)):
                layers[l] = sort_by_barycenter(layers[l], layers[l - 1], use_preds=True)
            for l in range(len(layers) - 2, -1, -1):
                layers[l] = sort_by_barycenter(
                    layers[l], layers[l + 1], use_preds=False
                )
        self.ctx["layers"] = layers

    def _assign_coordinates(self):
        layout = self.params["layout"]
        layer = self.ctx["layer"]
        layers = self.ctx["layers"]
        sub_nodes = self.ctx["sub_nodes"]
        id_to_node = self.ctx["id_to_node"]
        root = self.ctx["root"]
        max_layer = self.ctx["max_layer"]

        margin_x = 0.08
        margin_y = 0.08
        usable_w = 1.0 - 2 * margin_x
        usable_h = 1.0 - 2 * margin_y
        depth = max_layer + 1 if max_layer >= 0 else 1
        ordered_nodes: List[RWXNode] = []
        coords: List[Tuple[float, float]] = []

        if layout == "tidy":
            primary_children: Dict[int, List[int]] = defaultdict(list)
            sub_id_set = self.ctx["sub_id_set"]
            for n in sub_nodes:
                pid = n._primary_parent_id
                if pid is not None and pid in sub_id_set:
                    primary_children[pid].append(n.id)
            for pid, ch in primary_children.items():
                ch.sort()
            x_pos: Dict[int, float] = {}
            seen: set[int] = set()
            x_cursor = 0.0

            def assign_x(nid: int):
                nonlocal x_cursor
                children = primary_children.get(nid, [])
                if not children:
                    x_pos[nid] = x_cursor
                    x_cursor += 1.0
                else:
                    for c in children:
                        assign_x(c)
                    x_pos[nid] = float(np.mean([x_pos[c] for c in children]))
                seen.add(nid)

            assign_x(root.id)
            for n in sorted(sub_nodes, key=lambda t: (layer[t.id], t.id)):
                if n.id not in seen:
                    assign_x(n.id)
            xs = list(x_pos.values())
            xmin = min(xs) if xs else 0.0
            xmax = max(xs) if xs else 1.0
            span = (xmax - xmin) if (xmax - xmin) > 1e-9 else 1.0
            x_norm: Dict[int, float] = {}
            for nid, xv in x_pos.items():
                x_norm[nid] = margin_x + ((xv - xmin) / span) * usable_w
            for l, layer_nodes in enumerate(layers):
                if not layer_nodes:
                    continue
                y_raw = 1.0 if depth == 1 else 1.0 - (l / float(max(depth - 1, 1)))
                y = margin_y + y_raw * usable_h
                layer_nodes_sorted = sorted(
                    layer_nodes, key=lambda nid: (x_norm.get(nid, 0.0), nid)
                )
                for nid in layer_nodes_sorted:
                    x = x_norm.get(nid, margin_x + 0.5 * usable_w)
                    ordered_nodes.append(id_to_node[nid])
                    coords.append((x, y))
        else:
            for l, layer_nodes in enumerate(layers):
                k = max(1, len(layer_nodes))
                y_raw = 1.0 if depth == 1 else 1.0 - (l / float(max(depth - 1, 1)))
                y = margin_y + y_raw * usable_h
                for i, nid in enumerate(layer_nodes):
                    x_rel = (i + 1) / float(k + 1)
                    x = margin_x + x_rel * usable_w
                    ordered_nodes.append(id_to_node[nid])
                    coords.append((x, y))
        if not ordered_nodes:
            fig, ax = plt.subplots(figsize=(self.params["figsize"] or (6, 4)))
            self.ctx.update(dict(empty=True, fig=fig, ax=ax))
            return
        norm_pos = np.array(coords, dtype=float)
        x_extent, y_extent = 1.0, 1.0
        if layout == "tidy":
            sx_eff = float(self.params["spacing_x"])
            sy_eff = float(self.params["spacing_y"])
            norm_pos[:, 0] = margin_x + (norm_pos[:, 0] - margin_x) * sx_eff
            norm_pos[:, 1] = margin_y + (norm_pos[:, 1] - margin_y) * sy_eff
            x_extent = 2.0 * margin_x + (1.0 - 2 * margin_x) * sx_eff
            y_extent = 2.0 * margin_y + (1.0 - 2 * margin_y) * sy_eff
        self.ctx.update(
            dict(
                ordered_nodes=ordered_nodes,
                coords=coords,
                norm_pos=norm_pos,
                x_extent=x_extent,
                y_extent=y_extent,
                max_width=max((len(L) for L in layers), default=1),
                depth=(self.ctx["max_layer"] + 1 if self.ctx["max_layer"] >= 0 else 1),
            )
        )

    def _finalize_positions_and_edges(self):
        ordered_nodes = self.ctx["ordered_nodes"]
        rx_edges = self.ctx["rx_edges"]
        id_map: Dict[int, int] = {n.id: i for i, n in enumerate(ordered_nodes)}
        edges = [
            (id_map[u], id_map[v]) for (u, v) in rx_edges if u in id_map and v in id_map
        ]
        self.ctx.update(dict(id_map=id_map, edges=edges))

    def _auto_figsize(self):
        figsize = self.params["figsize"]
        if figsize is not None:
            self.ctx["figsize"] = figsize
            return
        ordered_nodes = self.ctx["ordered_nodes"]
        norm_pos = self.ctx["norm_pos"]
        edges = self.ctx["edges"]
        max_width = self.ctx["max_width"]
        depth = self.ctx["depth"]
        label_lengths = [len(getattr(n, "name", "")) for n in ordered_nodes]
        avg_label_len = float(np.mean(label_lengths)) if label_lengths else 0.0
        label_factor_w = 1.0 + min(2.0, avg_label_len / 30.0)

        def _dist_pt_seg_l(px, py, ax_, ay_, bx_, by_):
            vx, vy = bx_ - ax_, by_ - ay_
            wx, wy = px - ax_, py - ay_
            vv = vx * vx + vy * vy
            if vv <= 1e-12:
                return float(np.hypot(wx, wy))
            t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
            cx, cy = ax_ + t * vx, ay_ + t * vy
            return float(np.hypot(px - cx, py - cy))

        near_hits = 0
        thr = 0.06
        for u_idx, v_idx in edges:
            ax_, ay_ = norm_pos[u_idx, 0], norm_pos[u_idx, 1]
            bx_, by_ = norm_pos[v_idx, 0], norm_pos[v_idx, 1]
            xmin, xmax = min(ax_, bx_), max(ax_, bx_)
            ymin, ymax = min(ay_, by_), max(ay_, by_)
            for k, (xn, yn) in enumerate(norm_pos):
                if k == u_idx or k == v_idx:
                    continue
                if (xmin <= xn <= xmax) and (ymin <= yn <= ymax):
                    if _dist_pt_seg_l(xn, yn, ax_, ay_, bx_, by_) < thr:
                        near_hits += 1
        cong_factor = (
            1.0 + min(1.5, near_hits / float(max(1, len(ordered_nodes)))) * 0.35
        )
        base_w, base_h = 12.0, 9.0
        w_scale = (
            max(1.0, (max_width / 2.0) * label_factor_w)
            * float(self.params["spacing_x"])
            * cong_factor
        )
        h_scale = (
            max(1.0, (depth / 3.0)) * float(self.params["spacing_y"]) * cong_factor
        )
        self.ctx["figsize"] = (base_w * w_scale, base_h * h_scale)

    def _setup_figure(self):
        figsize = self.ctx.get("figsize", self.params["figsize"])
        fig, ax = plt.subplots(figsize=figsize)
        xlim = self.ctx.get("xlim", (0.0, self.ctx["x_extent"]))
        ylim = self.ctx.get("ylim", (0.0, self.ctx["y_extent"]))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return fig, ax

    def _wrap_labels_and_sizes(self):
        ordered_nodes = self.ctx["ordered_nodes"]
        node_size = float(self.params["node_size"])
        labels = [n.name for n in ordered_nodes]
        figsize = self.ctx.get("figsize", self.params["figsize"])
        wrap_width = 24
        if figsize is not None:
            dynamic_width = max(16, int(24 * (figsize[0] / 12.0)))
            wrap_width = dynamic_width
        if self.params["label_max_chars_per_line"] is not None:
            wrap_width = int(self.params["label_max_chars_per_line"])
        wrapped_labels = [
            fill(lbl, width=wrap_width, break_long_words=True, break_on_hyphens=True)
            for lbl in labels
        ]
        line_counts = [wl.count("\n") + 1 for wl in wrapped_labels]
        n_total = len(ordered_nodes)
        if n_total > 18:
            base_size = node_size * max(0.72, min(1.0, float(np.sqrt(18.0 / n_total))))
        else:
            base_size = node_size
        size_per_node = []
        for lines in line_counts:
            scale = 1.0 + 0.70 * (lines - 1)
            size_per_node.append(base_size * scale)
        size_per_node = np.array(size_per_node, dtype=float)
        self.ctx.update(
            dict(
                labels=labels,
                wrapped_labels=wrapped_labels,
                size_per_node=size_per_node,
            )
        )

    def _compute_pixel_and_obstacles(self, fig, ax):
        norm_pos = self.ctx["norm_pos"]
        size_per_node = self.ctx["size_per_node"]
        radii_pt = np.sqrt(size_per_node / np.pi)
        fig_w_px, fig_h_px = (
            fig.get_size_inches()[0] * fig.dpi,
            fig.get_size_inches()[1] * fig.dpi,
        )
        bbox = ax.get_position()
        ax_w_px = max(1.0, bbox.width * fig_w_px)
        ax_h_px = max(1.0, bbox.height * fig_h_px)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        dx_per_px = (xlim[1] - xlim[0]) / ax_w_px
        dy_per_px = (ylim[1] - ylim[0]) / ax_h_px
        radii_px = radii_pt * (fig.dpi / 72.0)
        radii_data_x = radii_px * dx_per_px
        radii_data_y = radii_px * dy_per_px
        obstacles_rects: List[Tuple[float, float, float, float]] = []
        inflate_px_x = 3.0
        inflate_px_y = 3.0
        inflate_dx = dx_per_px * inflate_px_x
        inflate_dy = dy_per_px * inflate_px_y
        for k, (xn, yn) in enumerate(norm_pos):
            rx_k = float(abs(radii_data_x[k])) + inflate_dx
            ry_k = float(abs(radii_data_y[k])) + inflate_dy
            obstacles_rects.append((xn - rx_k, xn + rx_k, yn - ry_k, yn + ry_k))
        avg_rx = (
            float(np.mean([abs(v) for v in radii_data_x]))
            if len(radii_data_x)
            else 0.01
        )
        avg_ry = (
            float(np.mean([abs(v) for v in radii_data_y]))
            if len(radii_data_y)
            else 0.01
        )
        step_x = max(0.02, 2.5 * avg_rx)
        step_y = max(0.02, 2.5 * avg_ry)
        self.ctx.update(
            dict(
                radii_data_x=radii_data_x,
                radii_data_y=radii_data_y,
                dx_per_px=dx_per_px,
                dy_per_px=dy_per_px,
                obstacles_rects=obstacles_rects,
                step_x=step_x,
                step_y=step_y,
                prev_ortho_segments=[],
                bump_registry=set(),
            )
        )

    def _adjust_limits_and_center(self, ax):
        # Compute padded content bounds from obstacles and center horizontally around the root when possible.
        norm_pos = self.ctx["norm_pos"]
        obstacles_rects = self.ctx["obstacles_rects"]
        dx_per_px = self.ctx["dx_per_px"]
        dy_per_px = self.ctx["dy_per_px"]
        x_extent = self.ctx["x_extent"]
        y_extent = self.ctx["y_extent"]
        root = self.ctx["root"]
        id_map = self.ctx["id_map"]
        # Content bounds
        xs_min = min(r[0] for r in obstacles_rects) if obstacles_rects else 0.0
        xs_max = max(r[1] for r in obstacles_rects) if obstacles_rects else x_extent
        ys_min = min(r[2] for r in obstacles_rects) if obstacles_rects else 0.0
        ys_max = max(r[3] for r in obstacles_rects) if obstacles_rects else y_extent
        # Padding in pixels converted to data units
        pad_x = dx_per_px * 8.0
        pad_y = dy_per_px * 8.0
        # Horizontal centering around root if feasible
        root_idx = id_map.get(root.id)
        xs_min_pad = xs_min - pad_x
        xs_max_pad = xs_max + pad_x
        if root_idx is not None:
            root_x = float(norm_pos[root_idx, 0])
            # Choose symmetric window around root wide enough to include all content with padding
            half_width = max(root_x - xs_min_pad, xs_max_pad - root_x, 0.05)
            x_left = root_x - half_width
            x_right = root_x + half_width
            # If window exceeds extents, shift without shrinking
            if x_left < 0.0:
                shift = -x_left
                x_left += shift
                x_right += shift
            if x_right > x_extent:
                shift = x_right - x_extent
                x_left -= shift
                x_right -= shift
            # If still outside due to too large content, clamp to full extent
            if x_left < 0.0 or x_right > x_extent or (x_right - x_left) > x_extent:
                x_left = 0.0
                x_right = x_extent
        else:
            x_left = max(0.0, xs_min_pad)
            x_right = min(x_extent, xs_max_pad)
        y_bottom = max(0.0, ys_min - pad_y)
        y_top = min(y_extent, ys_max + pad_y)
        # Store and apply
        self.ctx["xlim"] = (x_left, x_right)
        self.ctx["ylim"] = (y_bottom, y_top)
        ax.set_xlim(self.ctx["xlim"])
        ax.set_ylim(self.ctx["ylim"])

    def _draw_wrap_boxes(self, ax):
        ordered_nodes = self.ctx["ordered_nodes"]
        id_map = self.ctx["id_map"]
        obstacles_rects = self.ctx["obstacles_rects"]
        x_extent = self.ctx["x_extent"]
        y_extent = self.ctx["y_extent"]
        dx_per_px = self.ctx["dx_per_px"]
        dy_per_px = self.ctx["dy_per_px"]
        wrap_roots = [n for n in ordered_nodes if n.wrap_subtree and (n is not n.root)]
        if not wrap_roots:
            return
        box_pad_px = 12.0
        pad_dx = dx_per_px * box_pad_px
        pad_dy = dy_per_px * box_pad_px
        for r in wrap_roots:
            ids_in_subtree = {r.id}

            def collect(node):
                for child in node.children:
                    if child.wrap_subtree:
                        continue
                    ids_in_subtree.add(child.id)
                    if child.children:
                        collect(child)

            collect(r)
            idxs = [id_map[nid] for nid in ids_in_subtree if nid in id_map]
            if not idxs:
                continue
            xs_min = min(obstacles_rects[k][0] for k in idxs) - pad_dx
            xs_max = max(obstacles_rects[k][1] for k in idxs) + pad_dx
            ys_min = min(obstacles_rects[k][2] for k in idxs) - pad_dy
            ys_max = max(obstacles_rects[k][3] for k in idxs) + pad_dy
            xs_min = max(0.0, xs_min)
            ys_min = max(0.0, ys_min)
            xs_max = min(x_extent, xs_max)
            ys_max = min(y_extent, ys_max)
            width = max(1e-6, xs_max - xs_min)
            height = max(1e-6, ys_max - ys_min)
            fc = (
                r.wrap_facecolor
                if r.wrap_facecolor
                else (r.color.color if r.color else "#cccccc")
            )
            ec = (
                r.wrap_edgecolor
                if r.wrap_edgecolor
                else (r.color.color if r.color else "#666666")
            )
            alpha_box = float(getattr(r, "wrap_alpha", 0.08))
            box = FancyBboxPatch(
                (xs_min, ys_min),
                width,
                height,
                boxstyle="round,pad=0.01",
                linewidth=2.0,
                edgecolor=ec,
                facecolor=fc,
                alpha=alpha_box,
                zorder=1.5,
            )
            ax.add_patch(box)

    @staticmethod
    def _dist_pt_seg(px, py, ax_, ay_, bx_, by_):
        vx, vy = bx_ - ax_, by_ - ay_
        wx, wy = px - ax_, py - ay_
        vv = vx * vx + vy * vy
        if vv <= 1e-12:
            return float(np.hypot(wx, wy))
        t = max(0.0, min(1.0, (wx * vx + wy * wy) / vv))
        cx, cy = ax_ + t * vx, ay_ + t * vy
        return float(np.hypot(px - cx, py - cy))

    def _seg_blocked(
        self,
        p_start: Tuple[float, float],
        p_end: Tuple[float, float],
        ignore: Tuple[int, int],
    ):
        """Return True if the segment p_start->p_end intersects any obstacle rectangle.
        Uses a robust Cohen–Sutherland style clipping check to detect any intersection,
        not only corner proximity or endpoint containment.
        """
        obstacles_rects = self.ctx["obstacles_rects"]
        x0, y0 = p_start
        x1, y1 = p_end

        # Cohen–Sutherland outcodes
        INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

        def outcode(x, y, xmin, xmax, ymin, ymax):
            code = INSIDE
            if x < xmin:
                code |= LEFT
            elif x > xmax:
                code |= RIGHT
            if y < ymin:
                code |= BOTTOM
            elif y > ymax:
                code |= TOP
            return code

        for k, (xmin, xmax, ymin, ymax) in enumerate(obstacles_rects):
            if k in ignore:
                continue
            code0 = outcode(x0, y0, xmin, xmax, ymin, ymax)
            code1 = outcode(x1, y1, xmin, xmax, ymin, ymax)
            while True:
                if (code0 | code1) == 0:
                    # Trivially accept: at least partially inside -> intersects
                    return True
                if (code0 & code1) != 0:
                    # Trivially reject: both endpoints share an outside zone
                    break
                # Calculate line segment intersection with rectangle boundary
                code_out = code0 or code1
                if code_out & TOP:
                    x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0) if (y1 != y0) else x0
                    y = ymax
                elif code_out & BOTTOM:
                    x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0) if (y1 != y0) else x0
                    y = ymin
                elif code_out & RIGHT:
                    y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0) if (x1 != x0) else y0
                    x = xmax
                else:  # LEFT
                    y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0) if (x1 != x0) else y0
                    x = xmin
                if code_out == code0:
                    x0, y0 = x, y
                    code0 = outcode(x0, y0, xmin, xmax, ymin, ymax)
                else:
                    x1, y1 = x, y
                    code1 = outcode(x1, y1, xmin, xmax, ymin, ymax)
        return False

    def _draw_edges(self, ax):
        edge_style = self.params["edge_style"]
        curve_scale = float(self.params["curve_scale"])
        edges = self.ctx["edges"]
        norm_pos = self.ctx["norm_pos"]
        radii_data_x = self.ctx["radii_data_x"]
        radii_data_y = self.ctx["radii_data_y"]
        obstacles_rects = self.ctx["obstacles_rects"]
        prev_ortho_segments = self.ctx["prev_ortho_segments"]
        x_extent = self.ctx["x_extent"]
        y_extent = self.ctx["y_extent"]
        arrow_color = "#666666"

        def within_x(xp: float) -> bool:
            return 0.0 <= xp <= x_extent

        def within_y(yp: float) -> bool:
            return 0.0 <= yp <= y_extent

        for idx, (u, v) in enumerate(edges):
            x0, y0 = norm_pos[u]
            x1, y1 = norm_pos[v]
            # congestion probe (for arc/spline)
            near_count = 0
            nearest = None
            min_d = 1e9
            for xmin, xmax, ymin, ymax in obstacles_rects:
                cx = 0.5 * (xmin + xmax)
                cy = 0.5 * (ymin + ymax)
                d = float(np.hypot(cx - 0.5 * (x0 + x1), cy - 0.5 * (y0 + y1)))
                if d < min_d:
                    min_d = d
                    nearest = (cx, cy)
                if (min(x0, x1) <= cx <= max(x0, x1)) and (
                    min(y0, y1) <= cy <= max(y0, y1)
                ):
                    if self._dist_pt_seg(cx, cy, x0, y0, x1, y1) < 0.06:
                        near_count += 1

            if edge_style == "straight":
                # Adaptively render long straight edges as splines, and crop endpoints to node borders.
                dvec = np.array([x1 - x0, y1 - y0], dtype=float)
                dist = float(np.hypot(dvec[0], dvec[1]))
                if dist < 1e-9:
                    continue
                uvec = dvec / dist
                pvec = np.array([-uvec[1], uvec[0]])
                # Crop endpoints to just outside node borders along the edge direction
                sA = 1.15 * float(
                    np.hypot(radii_data_x[u] * uvec[0], radii_data_y[u] * uvec[1])
                )
                sB = 1.15 * float(
                    np.hypot(radii_data_x[v] * uvec[0], radii_data_y[v] * uvec[1])
                )
                start = np.array([x0, y0]) + uvec * sA
                end = np.array([x1, y1]) - uvec * sB
                d2 = end - start
                L = float(np.hypot(d2[0], d2[1]))
                if L < 1e-9:
                    continue
                # Threshold for switching to spline (in normalized axis units)
                long_thr = 0.18  # about 18% of axis span
                if L > long_thr:
                    # Use spline with gentle lateral offset
                    offset = 0.12 * L
                    if near_count:
                        offset *= min(1.0 + 0.22 * near_count, 2.5)
                    if min_d < 0.08:
                        offset *= 1.0 + min(1.2, (0.08 - min_d) / 0.08 * 1.2)
                    offset *= curve_scale
                    sign = 1.0 if (idx % 2 == 0) else -1.0
                    if nearest is not None:
                        vec_to_near = np.array(nearest) - start
                        if float(np.dot(pvec, vec_to_near)) > 0:
                            sign = -sign
                    c1 = start + d2 * (1.0 / 3.0) + pvec * (offset * sign)
                    c2 = start + d2 * (2.0 / 3.0) + pvec * (offset * sign)
                    path = Path(
                        [tuple(start), tuple(c1), tuple(c2), tuple(end)],
                        [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
                    )
                    arrow = FancyArrowPatch(
                        path=path,
                        arrowstyle="-|>",
                        mutation_scale=16,
                        linewidth=4.0,
                        color=arrow_color,
                        alpha=0.5,
                        zorder=4,
                        clip_on=False,
                    )
                    ax.add_patch(arrow)
                else:
                    # Short straight segment: draw straight with cropped endpoints
                    from matplotlib.path import Path as MPath

                    verts = [tuple(start), tuple(end)]
                    codes = [MPath.MOVETO, MPath.LINETO]
                    path = MPath(verts, codes)
                    arrow = FancyArrowPatch(
                        path=path,
                        arrowstyle="-|>",
                        mutation_scale=16,
                        linewidth=4.0,
                        color=arrow_color,
                        alpha=0.5,
                        zorder=4,
                        clip_on=False,
                    )
                    ax.add_patch(arrow)
                continue

            if edge_style == "orthogonal":

                def try_hv():
                    xm = x1
                    p1 = (x0, y0)
                    p2 = (xm, y0)
                    p3 = (xm, y1)
                    if (
                        within_x(xm)
                        and not self._seg_blocked(p1, p2, ignore=(u, v))
                        and not self._seg_blocked(p2, p3, ignore=(u, v))
                    ):
                        return [p1, p2, p3]
                    return None

                def try_vh():
                    ym = y1
                    p1 = (x0, y0)
                    p2 = (x0, ym)
                    p3 = (x1, ym)
                    if (
                        within_y(ym)
                        and not self._seg_blocked(p1, p2, ignore=(u, v))
                        and not self._seg_blocked(p2, p3, ignore=(u, v))
                    ):
                        return [p1, p2, p3]
                    return None

                def try_vhv():
                    ym = 0.5 * (y0 + y1)
                    p1 = (x0, y0)
                    p2 = (x0, ym)
                    p3 = (x1, ym)
                    p4 = (x1, y1)
                    if (
                        within_y(ym)
                        and (not self._seg_blocked(p1, p2, ignore=(u, v)))
                        and (not self._seg_blocked(p2, p3, ignore=(u, v)))
                        and (not self._seg_blocked(p3, p4, ignore=(u, v)))
                    ):
                        return [p1, p2, p3, p4]
                    return None

                candidates: List[List[Tuple[float, float]]] = []
                for fn in (try_hv, try_vh, try_vhv):
                    pts = fn()
                    if pts is not None:
                        candidates.append(pts)

                def count_crosses(points: List[Tuple[float, float]]):
                    crosses = 0
                    for i in range(len(points) - 1):
                        a0, a1 = points[i], points[i + 1]
                        ax0, ay0 = a0
                        ax1, ay1 = a1
                        if abs(ax0 - ax1) < 1e-12:  # vertical
                            xv = ax0
                            for (
                                ori,
                                x0s,
                                y0s,
                                x1s,
                                y1s,
                                u_prev,
                                v_prev,
                            ) in prev_ortho_segments:
                                if (
                                    ori == "v"
                                    and min(y0s, y1s) <= ay0 <= max(y0s, y1s)
                                    and abs(x0s - xv) < 1e-9
                                ):
                                    crosses += 1
                        else:  # horizontal
                            yh = ay0
                            for (
                                ori,
                                x0s,
                                y0s,
                                x1s,
                                y1s,
                                u_prev,
                                v_prev,
                            ) in prev_ortho_segments:
                                if (
                                    ori == "h"
                                    and min(x0s, x1s) <= ax0 <= max(x0s, x1s)
                                    and abs(y0s - yh) < 1e-9
                                ):
                                    crosses += 1
                    return crosses

                best = None
                best_score = None
                for pts in candidates:
                    score = (count_crosses(pts), len(pts))
                    if best_score is None or score < best_score:
                        best_score, best = score, pts
                if best is None:
                    best = [(x0, y0), (x1, y1)]

                # Evaluate path quality: length ratio and crossings
                def total_len(points: List[Tuple[float, float]]):
                    acc = 0.0
                    for i in range(1, len(points)):
                        ax0, ay0 = points[i - 1]
                        ax1, ay1 = points[i]
                        acc += float(np.hypot(ax1 - ax0, ay1 - ay0))
                    return acc

                straight_len = float(np.hypot(x1 - x0, y1 - y0)) + 1e-9
                path_len = total_len(best)
                crosses = count_crosses(best)
                length_ratio = path_len / straight_len
                # Thresholds for fallback
                ratio_thr = float(
                    self.params.get("orthogonal_spline_ratio_threshold", 2.2)
                )
                cross_thr = int(self.params.get("orthogonal_crossings_threshold", 2))
                min_len_thr = self.params.get("orthogonal_min_length_threshold", None)
                should_min_len = (min_len_thr is not None) and (
                    straight_len > float(min_len_thr)
                )
                if (
                    (length_ratio > ratio_thr)
                    or (crosses > cross_thr)
                    or should_min_len
                ):
                    # Fallback to spline for problematic or long orthogonal path
                    dvec = np.array([x1 - x0, y1 - y0], dtype=float)
                    dist = float(np.hypot(dvec[0], dvec[1]))
                    if dist >= 1e-9:
                        uvec = dvec / dist
                        pvec = np.array([-uvec[1], uvec[0]])
                        sA = 1.15 * float(
                            np.hypot(
                                radii_data_x[u] * uvec[0], radii_data_y[u] * uvec[1]
                            )
                        )
                        sB = 1.15 * float(
                            np.hypot(
                                radii_data_x[v] * uvec[0], radii_data_y[v] * uvec[1]
                            )
                        )
                        start = np.array([x0, y0]) + uvec * sA
                        end = np.array([x1, y1]) - uvec * sB
                        d2 = end - start
                        L = float(np.hypot(d2[0], d2[1]))
                        if L >= 1e-9:
                            offset = 0.12 * L
                            if near_count:
                                offset *= min(1.0 + 0.22 * near_count, 2.5)
                            if min_d < 0.08:
                                offset *= 1.0 + min(1.2, (0.08 - min_d) / 0.08 * 1.2)
                            offset *= curve_scale
                            sign = 1.0 if (idx % 2 == 0) else -1.0
                            if nearest is not None:
                                vec_to_near = np.array(nearest) - start
                                if float(np.dot(pvec, vec_to_near)) > 0:
                                    sign = -sign
                            c1 = start + d2 * (1.0 / 3.0) + pvec * (offset * sign)
                            c2 = start + d2 * (2.0 / 3.0) + pvec * (offset * sign)
                            path = Path(
                                [tuple(start), tuple(c1), tuple(c2), tuple(end)],
                                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
                            )
                            arrow = FancyArrowPatch(
                                path=path,
                                arrowstyle="-|>",
                                mutation_scale=16,
                                linewidth=4.0,
                                color=arrow_color,
                                alpha=0.5,
                                zorder=4,
                                clip_on=False,
                            )
                            ax.add_patch(arrow)
                            continue
                from matplotlib.path import Path as MPath

                # Crop endpoints of the orthogonal path so it meets node borders
                pts = list(best)
                # Pre-cropping deduplication to ensure non-zero first/last segments
                pre_pts: List[Tuple[float, float]] = []
                for p in pts:
                    if not pre_pts:
                        pre_pts.append(p)
                    else:
                        if (
                            float(
                                np.hypot(p[0] - pre_pts[-1][0], p[1] - pre_pts[-1][1])
                            )
                            > 1e-12
                        ):
                            pre_pts.append(p)
                pts = pre_pts if len(pre_pts) >= 2 else pts
                clamp_frac = 0.8
                eps = 1e-9
                # First segment direction
                if len(pts) >= 2:
                    a0x, a0y = pts[0]
                    a1x, a1y = pts[1]
                    dir0 = np.array([a1x - a0x, a1y - a0y], dtype=float)
                    seg_len0 = float(np.hypot(dir0[0], dir0[1]))
                    if seg_len0 > eps:
                        # Compute effective radius in that direction for source node u
                        uvec0 = dir0 / seg_len0
                        sA = 1.15 * float(
                            np.hypot(
                                radii_data_x[u] * uvec0[0], radii_data_y[u] * uvec0[1]
                            )
                        )
                        # Clamp to avoid over-cropping short segments
                        sA = min(sA, clamp_frac * seg_len0)
                        pts[0] = (a0x + uvec0[0] * sA, a0y + uvec0[1] * sA)
                # Last segment direction
                if len(pts) >= 2:
                    b0x, b0y = pts[-2]
                    b1x, b1y = pts[-1]
                    dirL = np.array([b1x - b0x, b1y - b0y], dtype=float)
                    seg_lenL = float(np.hypot(dirL[0], dirL[1]))
                    if seg_lenL > eps:
                        uvecL = dirL / seg_lenL
                        sB = 1.15 * float(
                            np.hypot(
                                radii_data_x[v] * uvecL[0], radii_data_y[v] * uvecL[1]
                            )
                        )
                        # Clamp to avoid over-cropping short segments
                        sB = min(sB, clamp_frac * seg_lenL)
                        pts[-1] = (b1x - uvecL[0] * sB, b1y - uvecL[1] * sB)
                # Remove near-duplicate consecutive points after cropping
                dedup_pts: List[Tuple[float, float]] = []
                for p in pts:
                    if not dedup_pts:
                        dedup_pts.append(p)
                    else:
                        px, py = dedup_pts[-1]
                        if float(np.hypot(p[0] - px, p[1] - py)) > 1e-8:
                            dedup_pts.append(p)
                pts = dedup_pts if len(dedup_pts) >= 2 else pts
                verts: List[Tuple[float, float]] = []
                codes: List[int] = []

                def add_moveto(pt):
                    verts.append(pt)
                    codes.append(MPath.MOVETO)

                def add_lineto(pt):
                    verts.append(pt)
                    codes.append(MPath.LINETO)

                add_moveto(pts[0])
                for i in range(1, len(pts)):
                    add_lineto(pts[i])
                for i in range(1, len(pts)):
                    a = pts[i - 1]
                    b = pts[i]
                    ori = "v" if abs(a[0] - b[0]) < 1e-12 else "h"
                    prev_ortho_segments.append((ori, a[0], a[1], b[0], b[1], u, v))
                path = MPath(verts, codes)
                arrow = FancyArrowPatch(
                    path=path,
                    arrowstyle="-|>",
                    mutation_scale=16,
                    linewidth=4.0,
                    color=arrow_color,
                    alpha=0.5,
                    zorder=4,
                    clip_on=False,
                )
                ax.add_patch(arrow)
                continue

            # arc or spline routing
            dvec = np.array([x1 - x0, y1 - y0], dtype=float)
            dist = float(np.hypot(dvec[0], dvec[1]))
            if dist < 1e-9:
                continue
            uvec = dvec / dist
            pvec = np.array([-uvec[1], uvec[0]])
            sA = 1.15 * float(
                np.hypot(radii_data_x[u] * uvec[0], radii_data_y[u] * uvec[1])
            )
            sB = 1.15 * float(
                np.hypot(radii_data_x[v] * uvec[0], radii_data_y[v] * uvec[1])
            )
            start = np.array([x0, y0]) + uvec * sA
            end = np.array([x1, y1]) - uvec * sB
            d2 = end - start
            L = float(np.hypot(d2[0], d2[1]))
            if L < 1e-9:
                continue
            offset = 0.12 * L
            if near_count:
                offset *= min(1.0 + 0.22 * near_count, 2.5)
            if min_d < 0.08:
                offset *= 1.0 + min(1.2, (0.08 - min_d) / 0.08 * 1.2)
            offset *= curve_scale
            sign = 1.0 if (idx % 2 == 0) else -1.0
            if nearest is not None:
                vec_to_near = np.array(nearest) - start
                if float(np.dot(pvec, vec_to_near)) > 0:
                    sign = -sign
            c1 = start + d2 * (1.0 / 3.0) + pvec * (offset * sign)
            c2 = start + d2 * (2.0 / 3.0) + pvec * (offset * sign)
            path = Path(
                [tuple(start), tuple(c1), tuple(c2), tuple(end)],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
            )
            arrow = FancyArrowPatch(
                path=path,
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=4.0,
                color=arrow_color,
                alpha=0.5,
                zorder=4,
                clip_on=False,
            )
            ax.add_patch(arrow)

    def _draw_nodes_and_labels(self, ax):
        ordered_nodes = self.ctx["ordered_nodes"]
        norm_pos = self.ctx["norm_pos"]
        size_per_node = self.ctx["size_per_node"]
        wrapped_labels = self.ctx["wrapped_labels"]
        font_size = self.params["font_size"]
        colors = [
            n.color.color if n.color else ColorLegend().color for n in ordered_nodes
        ]
        # Base node markers
        ax.scatter(
            norm_pos[:, 0],
            norm_pos[:, 1],
            s=size_per_node,
            c=colors,
            edgecolors="black",
            linewidths=2.0,
            alpha=0.95,
            zorder=2,
        )
        # Optional enclosing circles for emphasis
        enclosed_mask = np.array([int(n.enclosed) for n in ordered_nodes], dtype=bool)
        any_enclosed = bool(enclosed_mask.any())
        if any_enclosed:
            enc_pos = norm_pos[enclosed_mask]
            enc_sizes = size_per_node[enclosed_mask] * 1.5
            ax.scatter(
                enc_pos[:, 0],
                enc_pos[:, 1],
                s=enc_sizes,
                facecolors="none",
                edgecolors="black",
                linewidths=4.0,
                alpha=0.95,
                zorder=2.8,
            )
        # Labels
        for (x, y), text in zip(norm_pos, wrapped_labels):
            # Avoid obscuring empty-label nodes with a white bbox; skip bbox if label is empty/whitespace
            if text.strip() == "":
                # Draw no text to keep the node fully visible
                continue
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=font_size,
                fontweight="medium",
                wrap=True,
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white", alpha=0.85),
                zorder=3,
            )
        self.ctx["colors"] = colors
        self.ctx["any_enclosed"] = any_enclosed

    def _draw_legend(self, fig, ax):
        from collections import OrderedDict
        from matplotlib.lines import Line2D

        ordered_nodes = self.ctx["ordered_nodes"]
        colors = self.ctx["colors"]
        fam_to_color = OrderedDict()
        for n, col in zip(ordered_nodes, colors):
            label = n.color.name if n.color else ColorLegend().name
            if label not in fam_to_color:
                fam_to_color[label] = col
        if not fam_to_color:
            return
        handles = []
        for lbl, col in fam_to_color.items():
            patch = mpl.patches.Patch(facecolor=col, edgecolor="black", label=lbl)
            handles.append(patch)
        # Add enclosed marker legend if applicable
        if self.ctx.get("any_enclosed", False):
            enclosed_handle = Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                label=self.node.enclosed_name,
                markerfacecolor="none",
                markeredgecolor="black",
                markeredgewidth=2.5,
            )
            handles.append(enclosed_handle)
        fw, fh = fig.get_size_inches()
        scale = 0.7 * (max(0.5, fw / 12.0) + max(0.5, fh / 9.0))
        legend_fs = float(np.clip(10.0 * scale, 8.0, 28.0))
        title_fs = float(np.clip(legend_fs * 1.1, 9.0, 32.0))
        ax.legend(
            handles=handles,
            title="Node types",
            loc="upper left",
            framealpha=0.9,
            fontsize=legend_fs,
            title_fontsize=title_fs,
        )

    def _style_and_save(self, fig, ax):
        # White backgrounds
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        # Optional grid can remain subtle
        ax.grid(True, alpha=0.25)
        # Keep title if desired
        ax.set_title(self.params["title"], fontsize=14, pad=20)
        ax.set_aspect("auto", adjustable="box")
        xlim = self.ctx.get("xlim", (0.0, self.ctx["x_extent"]))
        ylim = self.ctx.get("ylim", (0.0, self.ctx["y_extent"]))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # Remove tick marks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        # Hide axis spines (borders) and frame
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_frame_on(False)
        # Compact layout then save with a small white margin around content
        plt.tight_layout()
        filename = self.params["filename"]
        dpi = 300
        plt.savefig(
            filename,
            format="pdf",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.2,
            facecolor=fig.get_facecolor(),
        )
