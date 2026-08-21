// Unit tests for panels/graph/panel.js (node:test): the live-plan colour-group mapping.
//
// panel.js is loaded with its free variables (Panels, Graph, fetch) bound as explicit
// function parameters rather than through global/window stubs, since the file itself
// never touches `window` or `document` directly (it only reaches DOM elements handed to
// it via its own `root` parameter).
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');
const SOURCE = fs.readFileSync(path.join(WEB, 'panels/graph/panel.js'), 'utf8');

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
  };
  const buttons = ['knowledge', 'kinematics', 'plan', 'chart'].map(makeButton);
  byId['#graph-tabs'] = { querySelectorAll() { return buttons; } };
  return {
    innerHTML: '',
    querySelector(selector) { return byId[selector]; },
    buttons: buttons,
  };
}

function makeBus() {
  const handlers = {};
  return {
    on(event, cb) { (handlers[event] = handlers[event] || []).push(cb); },
    emit(event, payload) { (handlers[event] || []).forEach(function (cb) { cb(payload); }); },
  };
}

function makeFetch(responses) {
  return async function fetch(url) {
    const body = responses[url];
    if (!body) throw new Error('unexpected fetch: ' + url);
    return { status: 200, json: async function () { return body; } };
  };
}

function loadPanel(responses) {
  let factory = null;
  let lastBuild = null;
  const Panels = { define(id, f) { factory = f; } };
  const Graph = {
    attach() {}, build(payload) { lastBuild = payload; },
    onSelect() {}, onDoubleSelect() {}, highlight() {}, reset() {},
    setStatuses() { return false; },
  };
  new Function('Panels', 'Graph', 'fetch', SOURCE)(Panels, Graph, makeFetch(responses));
  return { factory: factory, lastBuild: function () { return lastBuild; } };
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
