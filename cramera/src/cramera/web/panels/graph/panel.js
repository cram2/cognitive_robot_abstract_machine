/* ============================================================================
 * panels/graph/panel.js — the graph view with five tabs.
 *
 *   Knowledge   the entity graph + CRAM architecture (double-click drills in)
 *   Kinematics  the robot's URDF tree (links as nodes, joints as edges)
 *   Plan        the executed plan tree, node border = execution status
 *   Statechart  the giskardpy motion statechart of the running motion group
 *   Transforms  the executing world's connection graph, node border = freshness
 *
 * The tabs in LIVE_ENDPOINT take their node status from the cramera-live bridge
 * while it is attached: structure changes rebuild the graph, pure status
 * changes only re-colour the rings (no layout jumps).
 *
 * Bus events:
 *   emits    entity:select {id, detail, relations}   node clicked
 *   listens  entity:highlight {ids, focus?}          spotlight matching nodes
 *   listens  scene:step {step}                       highlight the running episode
 *   listens  live:changed {on, url}                  start/stop the status poll
 *
 * Rendering is delegated to graph.js (window.Graph, vis-network wrapper).
 * ==========================================================================*/
Panels.define('graph', function (root, bus) {
  root.innerHTML =
    '<div class="graph-wrap">' +
    '  <div class="graph-tabs" id="graph-tabs">' +
    '    <button data-view="knowledge" class="active" title="the entity graph (EQL / knowledge base)">Knowledge</button>' +
    '    <button data-view="kinematics" title="the robot\'s kinematic structure — URDF links &amp; joints">Kinematics</button>' +
    '    <button data-view="plan" title="the plan tree, with the execution status of every node">Plan</button>' +
    '    <button data-view="chart" title="the giskardpy motion statechart of the running motion group">Statechart</button>' +
    '    <button data-view="transforms" title="the world\'s connection graph — which frame hangs from which, and how recently it moved">Transforms</button>' +
    '    <span class="gt-live" id="gt-live" title="node status is streaming from the running demo">◉ live status</span>' +
    '  </div>' +
    '  <div id="graph"></div>' +
    '  <div class="graph-zoom">' +
    '    <button id="graph-zoom-in" title="Zoom in — or pinch on a touchpad">+</button>' +
    '    <button id="graph-zoom-out" title="Zoom out — or pinch on a touchpad">−</button>' +
    '    <button id="graph-zoom-fit" title="Fit the whole graph">⤡</button>' +
    '  </div>' +
    '  <div id="graph-empty" class="graph-empty" style="display:none"></div>' +
    '  <div id="graph-nav" class="graph-nav" style="display:none">' +
    '    <button id="gnav-home" title="back to the overview">⌂</button>' +
    '    <button id="gnav-up" title="one level up">↑ back</button>' +
    '    <span id="gnav-path"></span>' +
    '  </div>' +
    '  <div class="legend" id="legend"></div>' +
    '</div>';

  const emptyEl = root.querySelector('#graph-empty');
  const navEl = root.querySelector('#graph-nav');
  const navUp = root.querySelector('#gnav-up');
  const navHome = root.querySelector('#gnav-home');
  const navPath = root.querySelector('#gnav-path');
  const tabsEl = root.querySelector('#graph-tabs');
  const liveBadge = root.querySelector('#gt-live');
  Graph.attach(root.querySelector('#graph'), root.querySelector('#legend'));

  // %% zoom controls
  // one step in and its exact inverse out, so clicking + then − lands where it started
  const ZOOM_STEP = 1.3;
  root.querySelector('#graph-zoom-in').addEventListener('click', function () { Graph.zoomBy(ZOOM_STEP); });
  root.querySelector('#graph-zoom-out').addEventListener('click', function () { Graph.zoomBy(1 / ZOOM_STEP); });
  root.querySelector('#graph-zoom-fit').addEventListener('click', function () { Graph.fit(); });

  // %% tabs
  const TABS = {
    knowledge:  { url: '/api/knowledge' },
    kinematics: { url: '/api/knowledge/view?name=kinematics' },
    plan:       { url: '/api/knowledge/view?name=plan' },
    chart:      { url: '/api/knowledge/view?name=chart' },
    transforms: { url: '/api/knowledge/view?name=transforms' },
  };
  // the bridge endpoint each live view polls
  const LIVE_ENDPOINT = { plan: '/plan', chart: '/chart', transforms: '/transforms' };
  let tab = 'knowledge';
  let view = null;            // the currently rendered payload
  const base = {};            // tab -> payload as loaded from the server
  const shown = {};           // tab -> payload currently rendered (drill-downs)
  const stacks = {};          // tab -> parent payloads for the back button
  Object.keys(TABS).forEach(function (t) { stacks[t] = []; });
  let inGraphSet = {};

  function setView(payload) {
    view = payload;
    shown[tab] = payload;
    inGraphSet = {};
    payload.nodes.forEach(function (n) { inGraphSet[n.id] = 1; });
    if (emptyEl) {
      const empty = !payload.nodes.length;
      emptyEl.style.display = empty ? '' : 'none';
      emptyEl.textContent = empty ? (payload.empty || 'Nothing to show in this view.') : '';
    }
    Graph.build({
      nodes: payload.nodes, edges: payload.edges, legend: payload.legend,
      layout: payload.layout, arrows: !!payload.arrows,
      // a view may name the statuses its legend lists, instead of taking the default
      statusLegend: payload.statusLegend || false,
      key: (payload.key || tab) + '#' + stacks[tab].length,
    });
    updateNav();
  }
  function updateNav() {
    const inside = stacks[tab].length > 0;
    navEl.style.display = inside ? '' : 'none';
    if (inside) {
      const path = stacks[tab].slice(1).map(function (v) { return v.breadcrumb; }).concat([view.breadcrumb]);
      navPath.textContent = path.join(' / ');
    }
  }
  async function drill(id) {
    if (!view.details[id]) return;
    try {
      const r = await fetch(SceneContext.withScene('/api/knowledge/expand?node=' + encodeURIComponent(id)));
      const p = await ResponseUtil.parseJson(r);
      if (!p.ok) return;                       // node has no inside view
      stacks[tab].push(view);
      setView(p);
      select(id);
    } catch (err) { /* server unreachable — stay where we are */ }
  }
  function goBack() { if (stacks[tab].length) setView(stacks[tab].pop()); }
  function goHome() {
    if (!stacks[tab].length) return;
    stacks[tab] = [];
    setView(base[tab]);
  }
  navUp.addEventListener('click', goBack);
  navHome.addEventListener('click', goHome);

  async function showTab(name) {
    if (!TABS[name]) return;
    tab = name;
    tabsEl.querySelectorAll('button').forEach(function (b) {
      b.classList.toggle('active', b.dataset.view === name);
    });
    if (!base[name]) {
      emptyEl.style.display = '';
      emptyEl.textContent = 'loading…';
      try {
        const r = await fetch(SceneContext.withScene(TABS[name].url));
        if (r.status === 404) throw new Error('this build needs the /api/knowledge/view route — restart the server');
        const p = await ResponseUtil.parseJson(r);
        if (!p.ok) {
          emptyEl.textContent = p.error || 'view unavailable';
          return;
        }
        p.key = name;
        base[name] = p;
      } catch (err) {
        emptyEl.textContent = 'Could not load this view: ' + ((err && err.message) || err);
        return;
      }
    }
    setView(shown[name] || base[name]);
    liveRefresh(true);            // a live tab picks the bridge status up at once
  }
  tabsEl.querySelectorAll('button').forEach(function (b) {
    b.addEventListener('click', function () { showTab(b.dataset.view); });
  });

  // %% node click → describe in whatever panel listens
  function select(id) {
    const d = view && view.details && view.details[id];
    if (!d) return;
    const relations = (view.edges || [])
      .filter(function (e) { return e.from === id || e.to === id; })
      .map(function (e) {
        return { s: labelOf(e.from), p: e.label || e.kind, o: labelOf(e.to) };
      });
    bus.emit('entity:select', { id: id, detail: d, relations: relations });
    spotlight({ ids: [id], focus: id });
  }
  function labelOf(id) { return (view.details[id] && view.details[id].label) || id; }
  Graph.onSelect(select);
  Graph.onDoubleSelect(drill);

  // %% highlights (from EQL results or our own selection)
  function spotlight(p) {
    const ids = (p && p.ids) || [];
    let hi = ids.filter(function (id) { return inGraphSet[id]; });
    if (p && p.focus && inGraphSet[p.focus]) {
      const neighbours = (view.edges || [])
        .filter(function (e) { return e.from === p.focus || e.to === p.focus; })
        .map(function (e) { return e.from === p.focus ? e.to : e.from; });
      hi = hi.concat(neighbours.filter(function (id) { return inGraphSet[id]; }));
    }
    if (hi.length) Graph.highlight(hi); else Graph.reset();
  }
  bus.on('entity:highlight', spotlight);
  bus.on('scene:step', function (p) {
    if (p.step === '__done__') { Graph.reset(); return; }
    if (tab === 'knowledge' && !stacks[tab].length && inGraphSet[p.step]) select(p.step);
  });

  // %% live status overlay (Plan / Statechart tabs)
  // The bridge publishes the plan tree and the executing motion statechart with
  // per-node status. Structure changes (the plan grows as actions expand, a new
  // statechart is compiled per motion group) rebuild the graph; a pure status
  // change only re-colours the rings, so the layout never jumps.
  const CHART_LEGEND = [
    { group: 'task', label: 'Task (motion constraint)' },
    { group: 'monitor', label: 'Monitor / observation' },
    { group: 'motion_goal', label: 'Goal (contains nodes)' },
    { group: 'motion_end', label: 'End / cancel motion' },
  ];
  const TRANSFORM_LEGEND = [
    { group: 'world_frame', label: 'World root' },
    { group: 'actuated_frame', label: 'Actuated joint' },
    { group: 'free_frame', label: 'Free-floating' },
    { group: 'fixed_frame', label: 'Fixed to its parent' },
  ];
  const FRESHNESS_LEGEND = ['MOVING', 'SETTLED', 'STALE', 'STATIC'];
  const FRAME_GROUP = { actuated: 'actuated_frame', free: 'free_frame', fixed: 'fixed_frame' };
  const liveSig = { plan: '', chart: '', transforms: '' };
  let liveTimer = null;
  let liveState = { on: false, url: '' };

  function liveSource() {
    const p = shown[tab] || base[tab];
    return (p && p.live) || null;               // 'plan' | 'chart' | null
  }

  // drop the redundant 'Action' suffix only — a label that merely contains the word,
  // such as 'ActionNode', must survive intact. Mirrors
  // PlanViewPayload._shorten_action_label: the bridge sends the raw designator name,
  // so the live path shortens it here.
  function shortenActionLabel(label) {
    const shortened = label.replace(/Action$/, '');
    return shortened || label;
  }

  function planPayload(live) {
    const nodes = [], edges = [], details = {};
    (live.nodes || []).forEach(function (n) {
      const label = shortenActionLabel(n.label || '?');
      const lines = ['a ' + n.kind,
                     'status: ' + n.status + (n.derived ? ' (derived from the motion statechart)' : '')];
      if (n.arm) lines.push('arm: ' + n.arm);
      if (n.target) lines.push('target: ' + n.target);
      nodes.push({ id: n.id, label: label, group: n.group,
                   title: [label].concat(lines).join('\n'), status: n.status });
      details[n.id] = { label: label, group: n.group, lines: lines };
      if (n.parent) edges.push({ from: n.parent, to: n.id, kind: 'property', label: 'has step' });
    });
    return { ok: true, breadcrumb: 'live plan', nodes: nodes, edges: edges, details: details,
             legend: live.legend || [], layout: 'hier', arrows: true, statusLegend: true,
             live: 'plan', key: 'plan-live',
             empty: 'The bridge is attached but the demo has not started its plan yet.' };
  }

  function chartPayload(live) {
    const nodes = [], edges = [], details = {}, isParent = {};
    (live.nodes || []).forEach(function (n) { if (n.parent) isParent[n.parent] = 1; });
    (live.nodes || []).forEach(function (n) {
      const group = isParent[n.id] ? 'motion_goal'
        : /EndMotion|CancelMotion/.test(n.class_name) ? 'motion_end'
        : /Monitor|Reached|Observation|Condition/.test(n.class_name + n.name) ? 'monitor' : 'task';
      const lines = ['a ' + n.class_name, 'life cycle: ' + n.life_cycle, 'observation: ' + n.observation];
      nodes.push({ id: n.id, label: n.name, group: group,
                   title: [n.name].concat(lines).join('\n'), status: n.life_cycle });
      details[n.id] = { label: n.name, group: group, lines: lines };
      if (n.parent) edges.push({ from: n.parent, to: n.id, kind: 'type', label: 'contains' });
    });
    (live.edges || []).forEach(function (e) {
      edges.push({ from: e.from, to: e.to, kind: e.kind, label: (e.kind || '').toLowerCase() + ' transition' });
    });
    return { ok: true, breadcrumb: 'statechart' + (live.title ? ' · ' + live.title : ''),
             nodes: nodes, edges: edges, details: details, legend: CHART_LEGEND,
             layout: 'hier', arrows: true, statusLegend: true, live: 'chart', key: 'chart-live',
             empty: 'Attached, but no motion statechart is executing right now.' };
  }

  // The world's connection graph: one node per frame, one edge per connection. A
  // frame's ring is the freshness of the connection that carries it — the root frame,
  // which no connection carries, cannot go stale.
  function transformsPayload(live) {
    const nodes = [], edges = [], details = {}, carried = {};
    const connections = live.connections || [];
    connections.forEach(function (c) { carried[c.child] = c; });
    const frames = {};
    connections.forEach(function (c) { frames[c.parent] = 1; frames[c.child] = 1; });
    Object.keys(frames).forEach(function (frame) {
      const connection = carried[frame];
      const group = connection ? (FRAME_GROUP[connection.kind] || 'fixed_frame') : 'world_frame';
      const status = connection ? connection.freshness : 'STATIC';
      const lines = connection
        ? ['a frame carried by ' + connection.name,
           'connection: ' + connection.kind,
           'last written by: ' + connection.writer,
           'last changed: ' + (connection.ageSeconds === null || connection.ageSeconds === undefined
             ? 'never, since the bridge attached'
             : connection.ageSeconds + ' s ago')]
        : ['the world root frame'];
      nodes.push({ id: frame, label: frame, group: group,
                   title: [frame].concat(lines).join('\n'), status: status });
      details[frame] = { label: frame, group: group, lines: lines };
    });
    connections.forEach(function (c) {
      edges.push({ from: c.parent, to: c.child, kind: c.kind, label: c.name });
    });
    return { ok: true, breadcrumb: 'transform graph', nodes: nodes, edges: edges,
             details: details, legend: TRANSFORM_LEGEND, layout: 'hier', arrows: true,
             statusLegend: FRESHNESS_LEGEND, live: 'transforms', key: 'transforms-live',
             empty: 'Attached, but the demo has not published a world yet.' };
  }

  function livePayload(source, live) {
    if (source === 'plan') return planPayload(live);
    if (source === 'chart') return chartPayload(live);
    return transformsPayload(live);
  }

  async function liveRefresh(force) {
    const src = liveSource();
    const active = !!src && liveState.on;
    liveBadge.classList.toggle('on', active);
    if (!active) return;
    if (stacks[tab].length) return;              // inside a drill-down: leave it alone
    let live;
    try {
      live = await fetch(liveState.url + LIVE_ENDPOINT[src]).then(ResponseUtil.parseJson);
    } catch (err) { return; }                    // bridge gone — the 3D side handles it
    if (!live || !(live.nodes || live.connections)) return;
    const payload = livePayload(src, live);
    if (force || live.signature !== liveSig[src]) {    // structure changed → rebuild
      liveSig[src] = live.signature;
      base[tab] = payload;
      setView(payload);
      return;
    }
    // same structure: only re-colour, and keep the detail lines in sync
    const map = {};
    payload.nodes.forEach(function (n) { map[n.id] = n.status; });
    if (!Graph.setStatuses(map)) { base[tab] = payload; setView(payload); return; }
    base[tab] = payload;
    if (view && view.details) view.details = payload.details;
  }

  bus.on('live:changed', function (p) {
    liveState = { on: !!p.on, url: p.url || '' };
    if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
    if (liveState.on) {
      liveTimer = setInterval(function () { liveRefresh(false); }, 700);
      liveRefresh(true);
    } else {
      liveBadge.classList.remove('on');
      liveSig.plan = liveSig.chart = liveSig.transforms = '';
      // drop the live payloads so the live tabs fall back to the recorded bundle
      const liveTabs = Object.keys(LIVE_ENDPOINT);
      liveTabs.forEach(function (t) { delete base[t]; delete shown[t]; stacks[t] = []; });
      if (liveTabs.indexOf(tab) >= 0) showTab(tab);
    }
  });

  // %% boot
  showTab('knowledge');

  return {
    destroy: function () {
      if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
    },
  };
});
