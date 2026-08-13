// Unit tests for panels/graph/graph.js (node:test) against stubbed vis + DOM:
// status rings, layouts per view kind, in-place status patching and the zoom
// floor that keeps rings readable on big graphs.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

// %% stubs of the vis-network interfaces the renderer drives
class ItemStore {
  constructor(items) {
    this.items = {};
    this.auto = 0;
    (items || []).forEach((item) => {
      const id = item.id !== undefined ? item.id : '_' + this.auto++;
      this.items[id] = item;
    });
  }
  update(list) {
    (Array.isArray(list) ? list : [list]).forEach((patch) => {
      this.items[patch.id] = Object.assign({}, this.items[patch.id], patch);
    });
  }
  get(id) { return this.items[id]; }
}

let lastOptions = null;
let lastData = null;
let lastRenderer = null;

class RecordingRenderer {
  constructor(container, data, options) {
    lastData = data;
    lastOptions = options;
    lastRenderer = this;
    this.container = container;
    this.handlers = {};
  }
  once(event, cb) { (this.handlers[event] = this.handlers[event] || []).push(cb); }
  on(event, cb) { (this.handlers[event] = this.handlers[event] || []).push(cb); }
  fire(event, payload) { (this.handlers[event] || []).forEach((cb) => cb(payload)); }
  setOptions() {}
  fit() { this.fitted = (this.fitted || 0) + 1; }
  getScale() { return this.scale === undefined ? 0.3 : this.scale; }
  getViewPosition() { return { x: 0, y: 0 }; }
  moveTo(options) { this.moved = options; }
  getPositions() { return {}; }
  selectNodes() {}
  unselectAll() {}
  focus() {}
  redraw() {}
}

// the container the panel hands over, recording the input handlers installed on it
function makeCanvasStub() {
  const listeners = [];
  return {
    appendChild() {}, innerHTML: '',
    addEventListener(type, callback) { listeners.push({ type: type, callback: callback }); },
    removeEventListener(type, callback) {
      const at = listeners.findIndex((l) => l.type === type && l.callback === callback);
      if (at >= 0) listeners.splice(at, 1);
    },
    getBoundingClientRect() { return { left: 0, top: 0, width: 800, height: 600 }; },
    handlerCount(type) { return listeners.filter((l) => l.type === type).length; },
    handles(type) { return listeners.some((l) => l.type === type); },
    wheel(event) { listeners.filter((l) => l.type === 'wheel').forEach((l) => l.callback(event)); },
  };
}

function evaluateGraphJs() {
  global.document = {
    createElement() { return { className: '', innerHTML: '' }; },
  };
  global.window = {};
  // the real gesture module, so the wheel handling under test is the one that ships
  new Function('window', fs.readFileSync(path.join(WEB, 'core/graph-gestures.js'), 'utf8'))(global.window);
  global.GraphGestures = global.window.GraphGestures;
  global.vis = { DataSet: ItemStore, Network: RecordingRenderer };
  new Function(fs.readFileSync(path.join(WEB, 'panels/graph/graph.js'), 'utf8'))();
  return global.window.Graph;
}

// the renderer must receive its elements from the panel, never look them up itself
function loadGraphJs(canvas) {
  const graph = evaluateGraphJs();
  graph.attach(canvas || makeCanvasStub(), { appendChild() {}, innerHTML: '' });
  return graph;
}

function planFixture(Graph) {
  Graph.build({
    key: 'plan',
    layout: 'hier',
    arrows: true,
    statusLegend: true,
    nodes: [
      { id: 'p0', label: 'Sequential', group: 'other', status: 'SUCCEEDED' },
      { id: 'p1', label: 'Transport', group: 'event', status: 'RUNNING' },
      { id: 'p2', label: 'MoveTCP', group: 'robot', status: 'CREATED' },
      { id: 'p3', label: 'Place', group: 'event', status: 'FAILED' },
      { id: 'p4', label: 'plain', group: 'plan' },
    ],
    edges: [
      { from: 'p0', to: 'p1', kind: 'property' },
      { from: 'p1', to: 'p2', kind: 'property' },
      { from: 'p0', to: 'p3', kind: 'property' },
    ],
  });
}

const node = (id) => lastData.nodes.get(id);

// %% the panel owns the DOM
test('building before attach is refused', function () {
  const Graph = evaluateGraphJs();
  assert.throws(function () {
    Graph.build({ nodes: [], edges: [] });
  }, /attach/);
});

// %% status rings
test('status renders as a coloured ring + status word', function () {
  const Graph = loadGraphJs();
  planFixture(Graph);
  assert.strictEqual(node('p1').color.border, '#ffb648');            // running: amber
  assert.strictEqual(node('p1').label, 'Transport\nrunning');
  assert.strictEqual(node('p0').color.border, '#4bd38a');            // succeeded: green
  assert.strictEqual(node('p3').color.border, '#ff6b8b');            // failed: red
  assert.ok(node('p1').borderWidth > node('p2').borderWidth);        // active > created
  assert.strictEqual(node('p1').borderWidthSelected, node('p1').borderWidth);
  assert.ok(Array.isArray(node('p2').shapeProperties.borderDashes)); // created: dashed
  assert.strictEqual(node('p4').color, undefined);                   // no status: group style
});

test('group fill survives the status ring patch', function () {
  const Graph = loadGraphJs();
  planFixture(Graph);
  assert.strictEqual(node('p1').color.background, '#b98cff');        // event group fill
});

// %% live patching
test('setStatuses re-colours in place and rebuilds labels from the base', function () {
  const Graph = loadGraphJs();
  planFixture(Graph);
  assert.strictEqual(Graph.setStatuses({ p2: 'RUNNING', p1: 'SUCCEEDED' }), true);
  assert.strictEqual(node('p2').label, 'MoveTCP\nrunning');
  assert.strictEqual(node('p1').label, 'Transport\nsucceeded');      // not appended twice
});

test('setStatuses reports unknown ids so the caller rebuilds', function () {
  const Graph = loadGraphJs();
  planFixture(Graph);
  assert.strictEqual(Graph.setStatuses({ ghost: 'RUNNING' }), false);
});

// %% layouts
test('trees are hierarchical with physics off; entity graphs keep the force layout', function () {
  const Graph = loadGraphJs();
  planFixture(Graph);
  assert.strictEqual(lastOptions.layout.hierarchical.enabled, true);
  assert.strictEqual(lastOptions.physics, false);
  assert.strictEqual(lastOptions.edges.arrows.to.enabled, true);

  Graph.build({ key: 'knowledge', nodes: [{ id: 'k', label: 'k', group: 'robot' }], edges: [] });
  assert.strictEqual(lastOptions.layout.improvedLayout, true);
  assert.strictEqual(typeof lastOptions.physics, 'object');
  assert.strictEqual(lastOptions.edges.arrows.to.enabled, false);
});

test('status views scale nodes, labels and spacing up so rings stay readable', function () {
  const Graph = loadGraphJs();
  planFixture(Graph);
  const statusScaling = lastOptions.nodes.scaling.min;
  const statusFont = lastOptions.groups.event.font.size;
  Graph.build({ key: 'knowledge', nodes: [{ id: 'k', label: 'k', group: 'robot' }], edges: [] });
  assert.ok(statusScaling > lastOptions.nodes.scaling.min);
  assert.ok(statusFont > lastOptions.groups.event.font.size);
});

// %% zoom floor
test('a big status graph is never fitted below the zoom floor', function () {
  const Graph = loadGraphJs();
  planFixture(Graph);                       // stub getScale() reports 0.3 after fit
  lastRenderer.fire('afterDrawing');
  assert.ok(lastRenderer.fitted > 0);
  assert.strictEqual(lastRenderer.moved.scale, 0.75);
});

test('plain entity graphs fit freely (no zoom floor)', function () {
  const Graph = loadGraphJs();
  Graph.build({ key: 'knowledge', nodes: [{ id: 'k', label: 'k', group: 'robot' }], edges: [] });
  lastRenderer.fire('stabilizationIterationsDone');
  assert.strictEqual(lastRenderer.moved, undefined);
});

// %% touchpad input
test("vis-network's own wheel zoom is off, so the gestures are the only wheel handler", function () {
  const canvas = makeCanvasStub();
  const Graph = loadGraphJs(canvas);
  planFixture(Graph);
  assert.strictEqual(lastOptions.interaction.zoomView, false);
  assert.ok(canvas.handles('wheel'), 'no wheel handler installed on the graph container');
});

test('a two-finger scroll over the graph pans it instead of zooming', function () {
  const canvas = makeCanvasStub();
  const Graph = loadGraphJs(canvas);
  planFixture(Graph);
  let prevented = 0;
  canvas.wheel({ deltaX: 0, deltaY: 12, deltaMode: 0, ctrlKey: false, metaKey: false,
                 clientX: 400, clientY: 300, preventDefault() { prevented += 1; } });
  assert.strictEqual(prevented, 1, 'the page would scroll instead');
  assert.strictEqual(lastRenderer.moved.scale, undefined, 'a scroll must not rescale');
  // 12 pixels at the stub's scale of 0.3
  assert.strictEqual(lastRenderer.moved.position.y, 40);
});

test('rebuilding for another tab leaves one wheel handler, not one per build', function () {
  const canvas = makeCanvasStub();
  const Graph = loadGraphJs(canvas);
  planFixture(Graph);
  Graph.build({ key: 'knowledge', nodes: [{ id: 'k', label: 'k', group: 'robot' }], edges: [] });
  Graph.build({ key: 'kinematics', nodes: [{ id: 'l', label: 'l', group: 'base' }], edges: [] });
  assert.strictEqual(canvas.handlerCount('wheel'), 1);
});

test('the zoom buttons step the scale and the fit button refits', function () {
  const Graph = loadGraphJs();
  planFixture(Graph);
  Graph.zoomBy(2);
  assert.strictEqual(lastRenderer.moved.scale, 0.6);      // stub scale 0.3
  const fits = lastRenderer.fitted || 0;
  Graph.fit();
  assert.strictEqual(lastRenderer.fitted, fits + 1);
});

// %% statechart transitions
test('statechart transition kinds get distinct edge styles', function () {
  const Graph = loadGraphJs();
  Graph.build({
    key: 'chart', layout: 'hier', arrows: true,
    nodes: [
      { id: 's0', label: 'Goal', group: 'motion_goal', status: 'RUNNING' },
      { id: 's1', label: 'Move', group: 'robot', status: 'DONE' },
    ],
    edges: [
      { from: 's0', to: 's1', kind: 'START' },
      { from: 's1', to: 's0', kind: 'PAUSE' },
    ],
  });
  const edges = Object.values(lastData.edges.items);
  assert.strictEqual(edges[0].color.color, '#4bd38a');               // START: green
  assert.ok(Array.isArray(edges[1].dashes));                         // PAUSE: dashed
});
