/* ============================================================================
 * panels/graph/panel.js — the graph view with four tabs.
 *
 *   Knowledge   the entity graph + CRAM architecture (double-click drills in)
 *   Kinematics  the robot's URDF tree (links as nodes, joints as edges)
 *   Plan        the executed plan tree, node border = execution status
 *   Statechart  the giskardpy motion statechart of the running motion group
 *
 * Plan and Statechart additionally take LIVE node status from the cram-viz-live
 * bridge while it is attached: structure changes rebuild the graph, pure
 * status changes only re-colour the rings (no layout jumps).
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
    '    <span class="gt-live" id="gt-live" title="node status is streaming from the running demo">◉ live status</span>' +
    '  </div>' +
    '  <div id="graph"></div>' +
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

  // %% tabs
  const TABS = {
    knowledge:  { url: '/api/kb' },
    kinematics: { url: '/api/kb/view?name=kinematics' },
    plan:       { url: '/api/kb/view?name=plan' },
    chart:      { url: '/api/kb/view?name=chart' },
  };
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
      layout: payload.layout, arrows: !!payload.arrows, statusLegend: !!payload.statusLegend,
      key: (payload.key || tab) + '#' + stacks[tab].length,
    });
    updateNav();
  }
  function updateNav() {
    const inside = stacks[tab].length > 0;
    navEl.style.display = inside ? '' : 'none';
    if (inside) {
      const path = stacks[tab].slice(1).map(function (v) { return v.crumb; }).concat([view.crumb]);
      navPath.textContent = path.join(' / ');
    }
  }
  async function drill(id) {
    if (!view.details[id]) return;
    try {
      const r = await fetch('/api/kb/expand?node=' + encodeURIComponent(id));
      const p = await r.json();
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
        const r = await fetch(TABS[name].url);
        if (r.status === 404) throw new Error('this build needs the /api/kb/view route — restart the server');
        const p = await r.json();
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
  const PLAN_GROUP = {
    ActionNode: 'event', MotionNode: 'robot', ConditionNode: 'goal',
    AttachNode: 'object', DetachNode: 'object',
  };
  const PLAN_LEGEND = [
    { group: 'event', label: 'Action' }, { group: 'robot', label: 'Motion' },
    { group: 'goal', label: 'Condition' }, { group: 'object', label: 'Attach / detach' },
    { group: 'ind', label: 'Other plan node' },
  ];
  const CHART_LEGEND = [
    { group: 'robot', label: 'Task (motion constraint)' },
    { group: 'concept', label: 'Monitor / observation' },
    { group: 'klass', label: 'Goal (contains nodes)' },
    { group: 'event', label: 'End / cancel motion' },
  ];
  const liveSig = { plan: '', chart: '' };
  let liveTimer = null;
  let liveState = { on: false, url: '' };

  function liveSource() {
    const p = shown[tab] || base[tab];
    return (p && p.live) || null;               // 'plan' | 'chart' | null
  }

  // drop the redundant 'Action' suffix only — a label that merely contains the
  // word, such as 'ActionNode', must survive intact (mirrors kb.shorten_action_label)
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
      nodes.push({ id: n.id, label: label, group: PLAN_GROUP[n.kind] || 'ind',
                   title: [label].concat(lines).join('\n'), status: n.status });
      details[n.id] = { label: label, group: PLAN_GROUP[n.kind] || 'ind', lines: lines };
      if (n.parent) edges.push({ from: n.parent, to: n.id, kind: 'prop', label: 'has step' });
    });
    return { ok: true, crumb: 'live plan', nodes: nodes, edges: edges, details: details,
             legend: PLAN_LEGEND, layout: 'hier', arrows: true, statusLegend: true,
             live: 'plan', key: 'plan-live',
             empty: 'The bridge is attached but the demo has not started its plan yet.' };
  }

  function chartPayload(live) {
    const nodes = [], edges = [], details = {}, isParent = {};
    (live.nodes || []).forEach(function (n) { if (n.parent) isParent[n.parent] = 1; });
    (live.nodes || []).forEach(function (n) {
      const group = isParent[n.id] ? 'klass'
        : /EndMotion|CancelMotion/.test(n.cls) ? 'event'
        : /Monitor|Reached|Observation|Condition/.test(n.cls + n.name) ? 'concept' : 'robot';
      const lines = ['a ' + n.cls, 'life cycle: ' + n.life, 'observation: ' + n.obs];
      nodes.push({ id: n.id, label: n.name, group: group,
                   title: [n.name].concat(lines).join('\n'), status: n.life });
      details[n.id] = { label: n.name, group: group, lines: lines };
      if (n.parent) edges.push({ from: n.parent, to: n.id, kind: 'type', label: 'contains' });
    });
    (live.edges || []).forEach(function (e) {
      edges.push({ from: e.from, to: e.to, kind: e.kind, label: (e.kind || '').toLowerCase() + ' transition' });
    });
    return { ok: true, crumb: 'statechart' + (live.title ? ' · ' + live.title : ''),
             nodes: nodes, edges: edges, details: details, legend: CHART_LEGEND,
             layout: 'hier', arrows: true, statusLegend: true, live: 'chart', key: 'chart-live',
             empty: 'Attached, but no motion statechart is executing right now.' };
  }

  async function liveRefresh(force) {
    const src = liveSource();
    const active = !!src && liveState.on;
    liveBadge.classList.toggle('on', active);
    if (!active) return;
    if (stacks[tab].length) return;              // inside a drill-down: leave it alone
    let live;
    try {
      live = await fetch(liveState.url + (src === 'plan' ? '/plan' : '/chart'))
        .then(function (r) { return r.json(); });
    } catch (err) { return; }                    // bridge gone — the 3D side handles it
    if (!live || !live.nodes) return;
    const payload = src === 'plan' ? planPayload(live) : chartPayload(live);
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
      liveSig.plan = liveSig.chart = '';
      // drop the live payloads so both tabs fall back to the recorded bundle
      ['plan', 'chart'].forEach(function (t) { delete base[t]; delete shown[t]; stacks[t] = []; });
      if (tab === 'plan' || tab === 'chart') showTab(tab);
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
