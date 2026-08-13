/* ============================================================================
 * panels/graph/graph.js — vis-network rendering for the graph panel.
 *
 * A thin wrapper around vis-network used by panel.js: force layout for entity
 * graphs, hierarchical layout for trees (plan / statechart), execution status
 * drawn as node rings (see STATUS_STYLE), and in-place status re-colouring so
 * live updates never re-run the layout. Exposed as window.Graph.
 * ==========================================================================*/
(function () {
  'use strict';

  // the panel hands its own elements over on mount: a panel owns its DOM subtree,
  // so this renderer must never look them up on the document itself
  let el = null, legendEl = null;
  function attach(canvasEl, legendContainer) {
    el = canvasEl;
    legendEl = legendContainer;
  }

  const GROUP_STYLE = {
    // %% TBox
    root:    { color: '#e8eefb', ring: '#ffffff', size: 24, label: 'Root concept' },
    subpackage:   { color: '#5b8cff', ring: '#a9c2ff', size: 15, label: 'Subpackage' },
    python_class: { color: '#ffb648', ring: '#ffd89a', size: 13, label: 'Python class' },
    external_class:   { color: '#8c9bbd', ring: '#c3ccdf', size: 14, label: 'External base class' },
    // %% ABox individuals, bucketed by their asserted type
    robot:   { color: '#ff7a9c', ring: '#ffb3c6', size: 20, label: 'Robot / body' },
    object:  { color: '#39d5c8', ring: '#8ff0e7', size: 15, label: 'Object / substance' },
    event:   { color: '#b98cff', ring: '#d9c2ff', size: 16, label: 'Event / episode' },
    plan:    { color: '#ffb648', ring: '#ffd89a', size: 15, label: 'Executed plan' },
    package: { color: '#4bd38a', ring: '#a6ecc6', size: 14, label: 'Package' },
    other:   { color: '#7f8db0', ring: '#b6c0d8', size: 13, label: 'Other' },

    // %% kinematic-chain links, bucketed by the robot part they belong to
    base:       { color: '#4bd38a', ring: '#a6ecc6', size: 14, label: 'Base / torso' },
    left_arm:   { color: '#ff7a9c', ring: '#ffb3c6', size: 15, label: 'Left arm' },
    right_arm:  { color: '#b98cff', ring: '#d9c2ff', size: 15, label: 'Right arm' },
    gripper:    { color: '#39d5c8', ring: '#8ff0e7', size: 15, label: 'Grippers' },
    sensor:     { color: '#ffb648', ring: '#ffd89a', size: 15, label: 'Head / sensors' },

    // %% plan-tree nodes, bucketed by the kind of plan node
    action:           { color: '#b98cff', ring: '#d9c2ff', size: 16, label: 'Action' },
    motion:           { color: '#ff7a9c', ring: '#ffb3c6', size: 15, label: 'Motion' },
    condition:        { color: '#ffb648', ring: '#ffd89a', size: 15, label: 'Condition' },
    attachment:       { color: '#39d5c8', ring: '#8ff0e7', size: 15, label: 'Attach / detach' },
    other_plan_node:  { color: '#7f8db0', ring: '#b6c0d8', size: 13, label: 'Other plan node' },

    // %% motion-statechart nodes, bucketed by the kind giskardpy compiled
    task:        { color: '#ff7a9c', ring: '#ffb3c6', size: 20, label: 'Task (motion constraint)' },
    monitor:     { color: '#4bd38a', ring: '#a6ecc6', size: 14, label: 'Monitor / observation' },
    motion_goal: { color: '#5b8cff', ring: '#a9c2ff', size: 15, label: 'Goal (contains nodes)' },
    motion_end:  { color: '#b98cff', ring: '#d9c2ff', size: 16, label: 'End / cancel motion' },
  };

  // %% execution status → node ring
  // One palette for both status vocabularies: coraplex TaskStatus on plan nodes
  // (CREATED/RUNNING/SUCCEEDED/FAILED/INTERRUPTED/PAUSE) and giskardpy
  // LifeCycleValues on statechart nodes (NOT_STARTED/RUNNING/PAUSED/DONE/FAILED).
  const STATUS_STYLE = {
    RUNNING:     { c: '#ffb648', w: 20, text: 'running', legend: 'running' },
    SUCCEEDED:   { c: '#4bd38a', w: 16, text: 'succeeded', legend: 'succeeded / done' },
    DONE:        { c: '#4bd38a', w: 16, text: 'done', legend: 'succeeded / done' },
    FAILED:      { c: '#ff6b8b', w: 20, text: 'failed', legend: 'failed' },
    INTERRUPTED: { c: '#ff9d6b', w: 16, d: [9, 7], text: 'interrupted', legend: 'interrupted' },
    PAUSE:       { c: '#5b8cff', w: 16, d: [9, 7], text: 'paused', legend: 'paused' },
    PAUSED:      { c: '#5b8cff', w: 16, d: [9, 7], text: 'paused', legend: 'paused' },
    CREATED:     { c: '#46557a', w: 7, d: [4, 5], legend: 'not started' },
    NOT_STARTED: { c: '#46557a', w: 7, d: [4, 5], legend: 'not started' },
  };
  const STATUS_LEGEND_ORDER = ['RUNNING', 'SUCCEEDED', 'FAILED', 'PAUSE', 'INTERRUPTED', 'CREATED'];

  // vis node patch that draws `status` as the node's border. The colour object is
  // written in full (background included): a per-node colour replaces the group's,
  // so a partial one would drop the group's fill.
  function statusPatch(id, baseLabel, status, group) {
    const gs = GROUP_STYLE[group] || GROUP_STYLE.ind;
    const st = STATUS_STYLE[status];
    const patch = { id: id };
    if (!st) {
      patch.label = baseLabel;
      patch.borderWidth = 2;
      patch.borderWidthSelected = 2;
      patch.color = { background: gs.color, border: gs.ring,
                      highlight: { background: gs.ring, border: '#ffffff' } };
      patch.shapeProperties = { borderDashes: false };
      return patch;
    }
    patch.label = st.text ? baseLabel + '\n' + st.text : baseLabel;
    patch.borderWidth = st.w;
    patch.borderWidthSelected = st.w;   // vis would otherwise thin the ring to 2 on click
    patch.color = { background: gs.color, border: st.c,
                    highlight: { background: gs.ring, border: '#ffffff' } };
    patch.shapeProperties = { borderDashes: st.d || false };
    return patch;
  }

  const EDGE_STYLE = {
    // motion-statechart transitions (giskardpy TransitionKind)
    START:   { c: '#4bd38a', o: 0.75, w: 1.8 },
    END:     { c: '#ff6b8b', o: 0.7,  w: 1.6 },
    PAUSE:   { c: '#5b8cff', o: 0.6,  w: 1.4, d: [5, 4] },
    RESET:   { c: '#8c9bbd', o: 0.5,  w: 1.2, d: [2, 4] },
    isa:         { c: '#39d5c8', o: 0.5,  w: 1.6 },                          // subClassOf
    type:        { c: '#7f9cc9', o: 0.5,  w: 1.1, d: [3, 3] },               // rdf:type (instanceOf)
    prop:        { c: '#b98cff', o: 0.55, w: 1.4 },                          // object-property triple
    restriction: { c: '#ffb648', o: 0.5,  w: 1.3, d: [4, 3] },               // OWL restriction
    disjoint:    { c: '#ff6b8b', o: 0.4,  w: 1.2, d: [2, 3], noArrow: true },// owl:disjointWith (symmetric)
    default:     { c: '#3a4c6e', o: 0.4,  w: 1 },
  };

  let network = null, nodes = null, edges = null, allNodeIds = [];
  let gestures = null;      // the wheel handler installed on the container
  let selectCb = function () {};
  let dblCb = function () {};
  let freezeTimer = null;   // re-freeze physics a moment after a node drag ends
  let baseLabels = {};      // node id -> label without the status line
  let baseGroups = {};      // node id -> group (statusPatch needs its fill colour)
  const POS = {};           // view key -> settled node positions (instant re-open)
  let viewKey = null;
  let zoomFloor = 0;        // see fitView()

  // Fit the whole graph — but on status views never zoom out past a floor. vis
  // scales borders with the zoom, so a big statechart squeezed into the panel
  // would draw its status rings hairline-thin, which is the one thing they must
  // not be. Below the floor we show it at the floor and let the user pan.
  const MIN_STATUS_SCALE = 0.75;
  function fitView(floor) {
    if (!network) return;
    network.fit({ animation: false });
    if (floor && network.getScale() < floor) {
      network.moveTo({ scale: floor, position: network.getViewPosition(), animation: false });
    }
  }

  function build(data) {
    if (!el) throw new Error('Graph.attach(container, legend) must run before build()');
    const hier = data.layout === 'hier' || data.layout === 'hier-lr';
    viewKey = data.key || null;
    const cached = viewKey && !hier ? POS[viewKey] : null;
    baseLabels = {}; baseGroups = {};
    const visNodes = data.nodes.map(function (n) {
      const st = GROUP_STYLE[n.group] || GROUP_STYLE.ind;
      baseLabels[n.id] = n.label;
      baseGroups[n.id] = n.group;
      const node = { id: n.id, label: n.label, group: n.group,
                     title: n.title || n.label, value: st.size };
      if (n.status) Object.assign(node, statusPatch(n.id, n.label, n.status, n.group));
      if (cached && cached[n.id]) { node.x = cached[n.id].x; node.y = cached[n.id].y; }
      return node;
    });
    const seen = {};
    const rawEdges = data.edges.filter(function (e) {
      const k = e.from + '>' + e.to + '>' + (e.label || ''); if (seen[k]) return false; seen[k] = 1; return true;
    });
    // plain lines: no edge labels, no arrowheads — the relation name only shows
    // as a hover tooltip, and in the answer panel when a node is clicked
    const visEdges = rawEdges.map(function (e) {
      const s = EDGE_STYLE[e.kind] || EDGE_STYLE.default;
      return {
        from: e.from, to: e.to,
        title: e.label || undefined,
        color: { color: s.c, opacity: s.o }, width: s.w, dashes: s.d || false,
      };
    });

    nodes = new vis.DataSet(visNodes);
    edges = new vis.DataSet(visEdges);
    allNodeIds = visNodes.map(function (n) { return n.id; });
    const bigGraph = visNodes.length > 170;

    // a status ring only reads as a ring if the dot underneath it is big enough,
    // so status views draw larger nodes (and labels) — and need more room between them
    const hasStatus = data.nodes.some(function (n) { return !!n.status; });

    const groups = {};
    Object.keys(GROUP_STYLE).forEach(function (g) {
      const st = GROUP_STYLE[g];
      groups[g] = {
        shape: 'dot',
        color: { background: st.color, border: st.ring, highlight: { background: st.ring, border: '#fff' } },
        font: { color: '#dfe8fb', size: hasStatus ? 17 : 13, face: 'Inter',
                strokeWidth: 3, strokeColor: '#0b1220' },
        borderWidth: 2,
      };
    });

    // trees (plan / statechart) read far better laid out top-down (or
    // left-to-right) than as a force hairball
    const layout = hier
      ? { hierarchical: { enabled: true, direction: data.layout === 'hier-lr' ? 'LR' : 'UD',
                          sortMethod: 'directed',
                          levelSeparation: data.layout === 'hier-lr' ? 190 : (hasStatus ? 195 : 110),
                          nodeSpacing: hasStatus ? 250 : 130,
                          treeSpacing: hasStatus ? 280 : 160, shakeTowards: 'roots' } }
      : { improvedLayout: !bigGraph };   // improvedLayout pre-solve is expensive

    network = new vis.Network(el, { nodes: nodes, edges: edges }, {
      groups: groups,
      nodes: { scaling: hasStatus ? { min: 34, max: 56 } : { min: 12, max: 30 } },
      // straight edges render far cheaper than smooth curves at hundreds of edges
      edges: {
        smooth: hier ? { type: 'cubicBezier', forceDirection: data.layout === 'hier-lr' ? 'horizontal' : 'vertical', roundness: 0.5 } : false,
        arrows: { to: { enabled: !!data.arrows, scaleFactor: 0.6 } },
      },
      physics: (hier || cached) ? false : {
        // fewer settle iterations; physics is switched off once stable (below)
        stabilization: { iterations: bigGraph ? 120 : 200, updateInterval: 25 },
        barnesHut: { gravitationalConstant: -7000, springLength: 120, springConstant: 0.03, damping: 0.5, avoidOverlap: 0 },
      },
      // zoomView off: core/graph-gestures.js reads the wheel instead, so a touchpad
      // swipe pans rather than zooming a flat 10% per event
      interaction: { hover: !bigGraph, tooltipDelay: 150, navigationButtons: false,
                     dragNodes: true, zoomView: false },
      layout: layout,
    });

    if (gestures) gestures.destroy();
    gestures = GraphGestures.install(network, el);

    // freeze the simulation once it has settled — this is the key win: without it
    // vis keeps simulating forever, so every hover/redraw stays laggy.
    // Then fit the whole (now expanded) layout into the viewport: the initial
    // zoom is from before the physics spread the nodes out, so without this
    // the outer nodes end up outside the visible area.
    zoomFloor = hasStatus ? MIN_STATUS_SCALE : 0;
    network.once('stabilizationIterationsDone', function () {
      network.setOptions({ physics: false });
      if (viewKey) POS[viewKey] = network.getPositions();
      fitView(zoomFloor);
    });
    if (hier || cached) {
      // nothing to stabilize: fit straight away
      network.once('afterDrawing', function () { fitView(zoomFloor); });
    }

    // …but re-animate while a node is being dragged, so neighbours spring along
    // (that's the fun bit), then freeze again shortly after the drag ends.
    if (freezeTimer) { clearTimeout(freezeTimer); freezeTimer = null; }
    network.on('dragStart', function (params) {
      if (hier) return;                                    // keep the tree in place
      if (!params.nodes || !params.nodes.length) return;   // node drag, not a pan
      if (freezeTimer) { clearTimeout(freezeTimer); freezeTimer = null; }
      network.setOptions({ physics: true });
    });
    network.on('dragEnd', function (params) {
      if (hier) return;
      if (!params.nodes || !params.nodes.length) return;
      if (freezeTimer) clearTimeout(freezeTimer);
      freezeTimer = setTimeout(function () { if (network) network.setOptions({ physics: false }); }, 1000);
    });

    network.on('click', function (params) {
      if (params.nodes.length) selectCb(params.nodes[0]);
    });
    // double-click: on a node → drill into it; on empty space → fit the view
    network.on('doubleClick', function (params) {
      if (params.nodes.length) dblCb(params.nodes[0]);
      else if (!params.edges.length) fitView(zoomFloor);
    });

    buildLegend(data);
  }

  function statusLegend() {
    // rings, not filled dots — that is exactly how a status reads on a node
    STATUS_LEGEND_ORDER.forEach(function (key) {
      const st = STATUS_STYLE[key];
      const d = document.createElement('div');
      d.className = 'li';
      d.innerHTML = '<span class="ring" style="border-color:' + st.c +
        (st.d ? ';border-style:dashed' : '') + '"></span>' + st.legend;
      legendEl.appendChild(d);
    });
  }

  function buildLegend(data) {
    legendEl.innerHTML = '';
    // a view can ship its own legend rows (the URDF tree names the kinematic-chain
    // groups: base / left arm / right arm / gripper / sensor)
    if (data.legend) {
      data.legend.forEach(function (row) {
        const st = GROUP_STYLE[row.group];
        if (!st) return;
        const d = document.createElement('div');
        d.className = 'li';
        d.innerHTML = '<span class="dot" style="background:' + st.color + '"></span>' + row.label;
        legendEl.appendChild(d);
      });
      if (data.statusLegend) statusLegend();
      return;
    }
    const present = {};
    data.nodes.forEach(function (n) { present[n.group] = 1; });
    ['root', 'package', 'subpackage', 'python_class', 'external_class', 'robot', 'object', 'event', 'plan', 'other']
      .filter(function (g) { return present[g]; })
      .forEach(function (g) {
        const st = GROUP_STYLE[g];
        const d = document.createElement('div');
        d.className = 'li';
        d.innerHTML = '<span class="dot" style="background:' + st.color + '"></span>' + st.label;
        legendEl.appendChild(d);
      });
    if (data.statusLegend) statusLegend();
  }

  // Re-colour node rings in place — used by the live status poll, which must not
  // rebuild the graph (that would re-run the layout every second).
  // `map`: {nodeId: status}. Returns false if an id is unknown, i.e. the caller
  // has a structurally different graph and needs a real rebuild.
  function setStatuses(map) {
    if (!nodes) return false;
    const patch = [];
    for (const id in map) {
      if (baseLabels[id] === undefined) return false;
      patch.push(statusPatch(id, baseLabels[id], map[id], baseGroups[id]));
    }
    if (patch.length) nodes.update(patch);
    return true;
  }

  // dim everything, then spotlight the given ids
  function highlight(ids) {
    if (!nodes) return;
    const set = {}; ids.forEach(function (i) { set[i] = 1; });
    nodes.update(allNodeIds.map(function (id) {
      const on = set[id];
      return { id: id, opacity: on ? 1 : 0.16, font: { color: on ? '#ffffff' : '#5f7196' } };
    }));
    const present = ids.filter(function (i) { return set[i] && allNodeIds.indexOf(i) >= 0; });
    if (present.length) {
      network.selectNodes(present);
      network.fit({ nodes: present, animation: { duration: 600, easingFunction: 'easeInOutQuad' } });
    }
  }

  function reset() {
    if (!nodes) return;
    nodes.update(allNodeIds.map(function (id) { return { id: id, opacity: 1, font: { color: '#dfe8fb' } }; }));
    network.unselectAll();
    network.fit({ animation: { duration: 500 } });
  }

  function focus(id) {
    if (network && allNodeIds.indexOf(id) >= 0) network.focus(id, { scale: 1.1, animation: { duration: 500 } });
  }

  // re-fit after the container changes size (e.g. graph maximised to fullscreen)
  function resize() {
    if (!network) return;
    network.redraw();
    fitView(zoomFloor);
  }

  // %% zoom controls, for when a gesture is not the easiest way to get there
  function zoomBy(factor) {
    if (network) GraphGestures.zoomBy(network, factor);
  }

  function fit() {
    fitView(zoomFloor);
  }

  window.Graph = {
    attach: attach,
    build: build,
    zoomBy: zoomBy,
    fit: fit,
    setStatuses: setStatuses,
    highlight: highlight,
    reset: reset,
    focus: focus,
    resize: resize,
    onSelect: function (cb) { selectCb = cb; },
    onDoubleSelect: function (cb) { dblCb = cb; },
  };
})();
