const BrowserSource = require('./browser_source');
// Unit test for panels/graph/panel.js (node:test): the graph canvas must be visible
// before the renderer measures it.
//
// vis-network sizes its canvas from the container it is handed at build time, so a
// graph built into a hidden container comes out empty and stays empty until something
// resizes the window. The Plan tab hides the canvas to show its step list, so every
// graph tab reached from it is built into a hidden container unless the panel shows the
// canvas first.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');
const SOURCE = BrowserSource.read(path.join(WEB, 'panels/graph/panel.js'));

function loadModule(relativePath, scope) {
  new Function('window', BrowserSource.read(path.join(WEB, relativePath)))(scope);
  return scope;
}

function flush() {
  return new Promise(function (resolve) { setTimeout(resolve, 0); });
}

// %% stubs of the interfaces panel.js reads
function makeElement() {
  return {
    style: {},
    innerHTML: '',
    textContent: '',
    classList: { toggle() {}, add() {}, remove() {} },
    addEventListener() {},
    querySelector() { return undefined; },
    querySelectorAll() { return []; },
  };
}

function makeButton(view) {
  let onClick = null;
  return {
    dataset: { view: view },
    style: {},
    classList: { toggle() {} },
    addEventListener(event, callback) { if (event === 'click') onClick = callback; },
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
    '.graph-canvas': makeElement(),
    '#graph-steps': makeElement(),
    '#legend': makeElement(),
    '#graph-zoom-in': makeButton(),
    '#graph-zoom-out': makeButton(),
    '#graph-zoom-fit': makeButton(),
  };
  const buttons = ['knowledge', 'plan', 'chart', 'kinematics', 'transforms'].map(makeButton);
  byId['#graph-tabs'] = { querySelectorAll() { return buttons; } };
  return {
    innerHTML: '',
    querySelector(selector) { return byId[selector]; },
    buttons: buttons,
    control(selector) { return byId[selector]; },
  };
}

function makeBus() {
  return { on() {}, emit() {} };
}

function loadPanel(root, responses) {
  let factory = null;
  const canvasDisplayPerBuild = [];
  const canvas = root.control('.graph-canvas');
  const Panels = { define(id, f) { factory = f; } };
  const Graph = {
    attach() {},
    build() { canvasDisplayPerBuild.push(canvas.style.display); },
    onSelect() {}, onDoubleSelect() {}, highlight() {}, reset() {},
    setStatuses() { return true; },
    zoomBy() {}, fit() {}, resize() {},
  };
  const observed = [];
  let observerCallback = null;
  const window = {
    addEventListener() {},
    setTimeout(handler) { handler(); return 1; },
    clearTimeout() {},
    ResizeObserver: function (callback) {
      observerCallback = callback;
      return { observe(element) { observed.push(element); } };
    },
  };
  async function fetchStub(url) {
    const body = responses[url];
    if (!body) throw new Error('unexpected fetch: ' + url);
    return { ok: true, status: 200, json: async function () { return body; } };
  }
  new Function('Panels', 'Graph', 'fetch', 'ResponseUtil', 'SceneContext', 'window', SOURCE)(
    Panels, Graph, fetchStub,
    loadModule('core/response.js', {}).ResponseUtil,
    loadModule('core/scene.js', { location: { search: '' } }).SceneContext,
    window
  );
  const resizes = [];
  Graph.resize = function () { resizes.push(1); };
  return {
    instance: factory(root, makeBus()),
    canvasDisplayPerBuild: canvasDisplayPerBuild,
    observed: observed,
    resizes: resizes,
    reportSize: function (width, height) {
      observerCallback([{ contentRect: { width: width, height: height } }]);
    },
  };
}

const EMPTY_KNOWLEDGE = { ok: true, nodes: [], edges: [], details: {} };

// %% the canvas a graph is measured into
test('a graph tab reached from the step list is built into a visible canvas', async function () {
  const root = makeRoot();
  const panel = loadPanel(root, {
    '/api/knowledge': { ok: true, nodes: [{ id: 'e1', label: 'Milk', group: 'object' }], edges: [], details: {} },
    '/api/knowledge/view?name=plan': {
      ok: true, layout: 'hier', details: {}, edges: [],
      nodes: [{ id: 'p1', kind: 'ActionNode', label: 'TransportAction', status: 'CREATED', group: 'action' }],
    },
  });
  function click(view) {
    root.buttons.find(function (b) { return b.dataset.view === view; }).click();
    return flush();
  }
  try {
    await flush();
    await click('plan');                               // the step list hides the canvas
    assert.strictEqual(root.control('.graph-canvas').style.display, 'none');

    await click('knowledge');

    assert.strictEqual(panel.canvasDisplayPerBuild.at(-1), '');
  } finally {
    panel.instance.destroy();
  }
});


// %% the canvas is resized by more than the window
// hiding a panel re-columns the grid and dragging a divider re-shares it, neither of
// which fires a window resize — so the graph has to watch the canvas it draws into
test('the graph re-fits when its own canvas changes size', async function () {
  const root = makeRoot();
  const panel = loadPanel(root, { '/api/knowledge': EMPTY_KNOWLEDGE });
  try {
    await flush();
    assert.deepStrictEqual(panel.observed, [root.control('.graph-canvas')]);

    panel.reportSize(1566, 370);

    assert.strictEqual(panel.resizes.length, 1);
  } finally {
    panel.instance.destroy();
  }
});

test('a canvas with no size on screen is left alone', async function () {
  const root = makeRoot();
  const panel = loadPanel(root, { '/api/knowledge': EMPTY_KNOWLEDGE });
  try {
    await flush();

    panel.reportSize(0, 0);                   // the step list, or a hidden panel

    assert.strictEqual(panel.resizes.length, 0);
  } finally {
    panel.instance.destroy();
  }
});
