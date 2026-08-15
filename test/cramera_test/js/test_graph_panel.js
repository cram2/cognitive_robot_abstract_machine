// Unit tests for panels/graph/panel.js (node:test): the live-plan colour-group mapping.
//
// panel.js is loaded with its free variables (Panels, Graph, fetch, ResponseUtil)
// bound as explicit function parameters rather than through global/window stubs, since
// the file itself never touches `window` or `document` directly (it only reaches DOM
// elements handed to it via its own `root` parameter). ResponseUtil is the real
// core/response.js, so the panel's error handling is exercised, not a stub of it.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');
const SOURCE = fs.readFileSync(path.join(WEB, 'panels/graph/panel.js'), 'utf8');

function loadResponseUtil() {
  const scope = {};
  new Function('window', fs.readFileSync(path.join(WEB, 'core/response.js'), 'utf8'))(scope);
  return scope.ResponseUtil;
}

// the real core/scene.js too: it decides whether a ?scene= parameter is appended to
// every API url the panel requests, so stubbing it would hide a wrong url
function loadSceneContext(search) {
  const scope = { location: { search: search || '' } };
  new Function('window', fs.readFileSync(path.join(WEB, 'core/scene.js'), 'utf8'))(scope);
  return scope.SceneContext;
}

function flush() {
  return new Promise(function (resolve) { setTimeout(resolve, 0); });
}

// %% stubs of the interfaces panel.js reads
function makeElement() {
  return {
    style: {},
    textContent: '',
    classList: { toggle() {}, add() {}, remove() {} },
    addEventListener() {},
    querySelectorAll() { return []; },
  };
}

function makeButton(view) {
  let onClick = null;
  return {
    dataset: { view: view },
    classList: { toggle() {} },
    addEventListener(event, cb) { if (event === 'click') onClick = cb; },
    click() { if (onClick) onClick(); },
  };
}

function makeRoot() {
  const byId = {
    '#graph-empty': makeElement(),
    '#graph-nav': makeElement(),
    '#gnav-up': makeElement(),
    '#gnav-home': makeElement(),
    '#gnav-path': makeElement(),
    '#gt-live': makeElement(),
    '#graph': makeElement(),
    '#legend': makeElement(),
    '#graph-zoom-in': makeButton(),
    '#graph-zoom-out': makeButton(),
    '#graph-zoom-fit': makeButton(),
  };
  const buttons = ['knowledge', 'kinematics', 'plan', 'chart', 'transforms'].map(makeButton);
  byId['#graph-tabs'] = { querySelectorAll() { return buttons; } };
  return {
    innerHTML: '',
    querySelector(selector) { return byId[selector]; },
    buttons: buttons,
    control(selector) { return byId[selector]; },
  };
}

function makeBus() {
  const handlers = {};
  return {
    on(event, cb) { (handlers[event] = handlers[event] || []).push(cb); },
    emit(event, payload) { (handlers[event] || []).forEach(function (cb) { cb(payload); }); },
  };
}

function makeFetch(responses, requested) {
  return async function fetch(url) {
    if (requested) requested.push(url);
    const body = responses[url];
    if (!body) throw new Error('unexpected fetch: ' + url);
    if (typeof body === 'number') return errorPage(body);
    return { ok: true, status: 200, json: async function () { return body; } };
  };
}

// what a host with no matching backend route answers: an HTML page, not JSON
function errorPage(status) {
  return {
    ok: false,
    status: status,
    json: async function () {
      throw new SyntaxError('JSON.parse: unexpected character at line 1 column 1');
    },
  };
}

function loadPanel(responses, search) {
  let factory = null;
  let lastBuild = null;
  const requested = [];
  const Panels = { define(id, f) { factory = f; } };
  const zooms = [];
  const Graph = {
    attach() {}, build(payload) { lastBuild = payload; },
    onSelect() {}, onDoubleSelect() {}, highlight() {}, reset() {},
    setStatuses() { return false; },
    zoomBy(factor) { zooms.push(factor); }, fit() { zooms.push('fit'); },
  };
  new Function('Panels', 'Graph', 'fetch', 'ResponseUtil', 'SceneContext', SOURCE)(
    Panels, Graph, makeFetch(responses, requested), loadResponseUtil(), loadSceneContext(search)
  );
  return {
    factory: factory,
    lastBuild: function () { return lastBuild; },
    requested: requested,
    zooms: zooms,
  };
}

// %% live plan colour groups
// the bridge classifies plan nodes now (knowledge/enums.py's PlanNodeGroup); the panel
// only has to pass the group through, legend included
test('a live plan is drawn with the groups and legend the bridge sent', async function () {
  const panel = loadPanel({
    '/api/knowledge': { ok: true, nodes: [], edges: [], details: {} },
    '/api/knowledge/view?name=plan': { ok: true, nodes: [], edges: [], details: {}, live: 'plan' },
    'http://bridge/plan': {
      signature: 's1',
      nodes: [
        { id: 'a1', kind: 'AttachNode', label: 'AttachNode', status: 'CREATED', group: 'attachment' },
        { id: 'm1', kind: 'MotionNode', label: 'MotionNode', status: 'CREATED', group: 'motion' },
      ],
      legend: [{ group: 'attachment', label: 'Attach / detach' }],
    },
  });
  const root = makeRoot();
  const bus = makeBus();
  const instance = panel.factory(root, bus);
  try {
    await flush();

    root.buttons.find(function (b) { return b.dataset.view === 'plan'; }).click();
    await flush();

    bus.emit('live:changed', { on: true, url: 'http://bridge' });
    await flush();

    const byId = {};
    panel.lastBuild().nodes.forEach(function (n) { byId[n.id] = n; });
    assert.strictEqual(byId.a1.group, 'attachment');
    assert.strictEqual(byId.m1.group, 'motion');
    assert.deepStrictEqual(panel.lastBuild().legend, [
      { group: 'attachment', label: 'Attach / detach' },
    ]);
  } finally {
    instance.destroy();       // clears the live-poll setInterval even if an assertion above throws
  }
});


// %% live statechart colour groups
test('statechart nodes are grouped by the kind of node giskardpy compiled', async function () {
  const panel = loadPanel({
    '/api/knowledge': { ok: true, nodes: [], edges: [], details: {} },
    '/api/knowledge/view?name=chart': { ok: true, nodes: [], edges: [], details: {}, live: 'chart' },
    'http://bridge/chart': {
      signature: 'c1',
      title: 'reach',
      nodes: [
        { id: 'g0', name: 'ReachGoal', class_name: 'Goal', life_cycle: 'RUNNING', observation: '1' },
        { id: 't1', parent: 'g0', name: 'CartesianPose', class_name: 'CartesianPose', life_cycle: 'RUNNING', observation: '1' },
        { id: 'm1', parent: 'g0', name: 'PoseReached', class_name: 'PoseReached', life_cycle: 'RUNNING', observation: '0' },
        { id: 'e1', parent: 'g0', name: 'EndMotion', class_name: 'EndMotion', life_cycle: 'CREATED', observation: '0' },
      ],
      edges: [],
    },
  });
  const root = makeRoot();
  const bus = makeBus();
  const instance = panel.factory(root, bus);
  try {
    await flush();

    root.buttons.find(function (b) { return b.dataset.view === 'chart'; }).click();
    await flush();

    bus.emit('live:changed', { on: true, url: 'http://bridge' });
    await flush();

    const byId = {};
    panel.lastBuild().nodes.forEach(function (n) { byId[n.id] = n; });
    assert.strictEqual(byId.g0.group, 'motion_goal');   // has children
    assert.strictEqual(byId.t1.group, 'task');
    assert.strictEqual(byId.m1.group, 'monitor');       // name matches Reached
    assert.strictEqual(byId.e1.group, 'motion_end');
  } finally {
    instance.destroy();
  }
});


// %% live transform graph
// the bridge sends connections; the panel turns them into frames, and rings each
// frame with the freshness of the connection that carries it
function loadTransformsPanel() {
  return loadPanel({
    '/api/knowledge': { ok: true, nodes: [], edges: [], details: {} },
    '/api/knowledge/view?name=transforms': { ok: true, nodes: [], edges: [], details: {}, live: 'transforms' },
    'http://bridge/transforms': {
      signature: 't1',
      connections: [
        { name: 'root_T_drawer', parent: 'kitchen/root', child: 'kitchen/drawer',
          kind: 'actuated', writer: 'demo', freshness: 'MOVING', ageSeconds: 0.1 },
        { name: 'root_T_shelf', parent: 'kitchen/root', child: 'kitchen/shelf',
          kind: 'fixed', writer: 'nobody', freshness: 'STATIC', ageSeconds: null },
        { name: 'root_T_milk', parent: 'kitchen/root', child: 'milk.stl',
          kind: 'free', writer: 'viewer', freshness: 'SETTLED', ageSeconds: 1.5 },
      ],
    },
  });
}

async function showTransforms(panel) {
  const root = makeRoot();
  const bus = makeBus();
  const instance = panel.factory(root, bus);
  await flush();
  root.buttons.find(function (b) { return b.dataset.view === 'transforms'; }).click();
  await flush();
  bus.emit('live:changed', { on: true, url: 'http://bridge' });
  await flush();
  return instance;
}

test('each frame is grouped by the kind of connection that carries it', async function () {
  const panel = loadTransformsPanel();
  const instance = await showTransforms(panel);
  try {
    const byId = {};
    panel.lastBuild().nodes.forEach(function (n) { byId[n.id] = n; });
    assert.strictEqual(byId['kitchen/drawer'].group, 'actuated_frame');
    assert.strictEqual(byId['kitchen/shelf'].group, 'fixed_frame');
    assert.strictEqual(byId['milk.stl'].group, 'free_frame');
  } finally {
    instance.destroy();
  }
});

test('a frame is ringed with the freshness of its connection', async function () {
  const panel = loadTransformsPanel();
  const instance = await showTransforms(panel);
  try {
    const byId = {};
    panel.lastBuild().nodes.forEach(function (n) { byId[n.id] = n; });
    assert.strictEqual(byId['kitchen/drawer'].status, 'MOVING');
    assert.strictEqual(byId['milk.stl'].status, 'SETTLED');
  } finally {
    instance.destroy();
  }
});

test('the root frame no connection carries cannot go stale', async function () {
  const panel = loadTransformsPanel();
  const instance = await showTransforms(panel);
  try {
    const root = panel.lastBuild().nodes.find(function (n) { return n.id === 'kitchen/root'; });
    assert.strictEqual(root.group, 'world_frame');
    assert.strictEqual(root.status, 'STATIC');
  } finally {
    instance.destroy();
  }
});

test('every connection is drawn as an edge from its parent frame to its child', async function () {
  const panel = loadTransformsPanel();
  const instance = await showTransforms(panel);
  try {
    assert.deepStrictEqual(panel.lastBuild().edges, [
      { from: 'kitchen/root', to: 'kitchen/drawer', kind: 'actuated', label: 'root_T_drawer' },
      { from: 'kitchen/root', to: 'kitchen/shelf', kind: 'fixed', label: 'root_T_shelf' },
      { from: 'kitchen/root', to: 'milk.stl', kind: 'free', label: 'root_T_milk' },
    ]);
  } finally {
    instance.destroy();
  }
});

test('the transform view names the statuses its legend may list', async function () {
  const panel = loadTransformsPanel();
  const instance = await showTransforms(panel);
  try {
    assert.deepStrictEqual(panel.lastBuild().statusLegend, ['MOVING', 'SETTLED', 'STALE', 'STATIC']);
  } finally {
    instance.destroy();
  }
});

test('a frame reports who last wrote it and how long ago', async function () {
  const panel = loadTransformsPanel();
  const instance = await showTransforms(panel);
  try {
    const milk = panel.lastBuild().nodes.find(function (n) { return n.id === 'milk.stl'; });
    assert.match(milk.title, /last written by: viewer/);
    assert.match(milk.title, /last changed: 1.5 s ago/);
  } finally {
    instance.destroy();
  }
});

test('a frame nothing has written yet says so instead of reporting an age', async function () {
  const panel = loadTransformsPanel();
  const instance = await showTransforms(panel);
  try {
    const shelf = panel.lastBuild().nodes.find(function (n) { return n.id === 'kitchen/shelf'; });
    assert.match(shelf.title, /never, since the bridge attached/);
  } finally {
    instance.destroy();
  }
});

// %% a route with no backend
test('a view whose route has no backend reports the status, not a JSON.parse error', async function () {
  const panel = loadPanel({
    '/api/knowledge': { ok: true, nodes: [], edges: [], details: {} },
    '/api/knowledge/view?name=kinematics': 502,
  });
  const root = makeRoot();
  const instance = panel.factory(root, makeBus());
  try {
    await flush();

    root.buttons.find(function (b) { return b.dataset.view === 'kinematics'; }).click();
    await flush();

    const reported = root.querySelector('#graph-empty').textContent;
    assert.match(reported, /HTTP 502/);
    assert.doesNotMatch(reported, /JSON\.parse/);
  } finally {
    instance.destroy();
  }
});

// %% the active scene reaches the api
test('every api request carries the scene the url names', async function () {
  const requested = [];
  const panel = loadPanel({
    '/api/knowledge?scene=lab': { ok: true, nodes: [], edges: [], details: {} },
    '/api/knowledge/view?name=kinematics&scene=lab': { ok: true, nodes: [], edges: [], details: {} },
  }, '?scene=lab');
  const root = makeRoot();
  const instance = panel.factory(root, makeBus());
  try {
    await flush();
    root.buttons.find(function (b) { return b.dataset.view === 'kinematics'; }).click();
    await flush();

    assert.deepStrictEqual(panel.requested, [
      '/api/knowledge?scene=lab',
      '/api/knowledge/view?name=kinematics&scene=lab',
    ]);
  } finally {
    instance.destroy();
  }
});

// %% on-screen zoom controls
// a laptop touchpad is the awkward case for any gesture, so zooming must also be
// reachable without one
test('the zoom controls step the graph in, out and back to a full fit', async function () {
  const panel = loadPanel({ '/api/knowledge': { ok: true, nodes: [], edges: [], details: {} } });
  const root = makeRoot();
  const instance = panel.factory(root, makeBus());
  try {
    await flush();
    root.control('#graph-zoom-in').click();
    root.control('#graph-zoom-out').click();
    root.control('#graph-zoom-fit').click();

    assert.strictEqual(panel.zooms.length, 3);
    assert.ok(panel.zooms[0] > 1, 'zoom in must magnify, got ' + panel.zooms[0]);
    assert.ok(panel.zooms[1] < 1, 'zoom out must shrink, got ' + panel.zooms[1]);
    // stepping out undoes stepping in
    assert.ok(Math.abs(panel.zooms[0] * panel.zooms[1] - 1) < 1e-12);
    assert.strictEqual(panel.zooms[2], 'fit');
  } finally {
    instance.destroy();
  }
});
