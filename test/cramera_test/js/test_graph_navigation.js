'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const BrowserSource = require('./browser_source');

// %% controllable graph requests
const WEB = path.join(__dirname, '../../../cramera/src/cramera/web');

/** A response released by the test after another navigation has finished. */
class PendingResponse {
  constructor() {
    this.promise = new Promise(resolve => { this.resolve = resolve; });
  }
}

/** Minimal graph controls with observable navigation and rendering. */
class GraphNavigation {
  constructor() {
    this.responses = {
      '/api/knowledge': GraphNavigation.payload('knowledge'),
      '/api/knowledge/view?name=plan': GraphNavigation.payload('recorded', 'plan'),
    };
    this.handlers = new Map();
    this.elements = new Map();
    this.buttons = ['knowledge', 'plan', 'chart', 'kinematics', 'transforms'].map(name => {
      const button = this.element('tab-' + name);
      button.dataset = {view: name};
      return button;
    });
    this.element('#graph-tabs').querySelectorAll = () => this.buttons;
    this.root = {querySelector: selector => this.element(selector)};
    this.bus = {
      on: (name, handler) => this.handlers.set(name, handler),
      emit: (name, payload) => this.handlers.get(name)?.(payload),
    };
    const graph = {
      attach() {},
      build: payload => { this.rendered = payload; },
      onSelect() {},
      onDoubleSelect: handler => { this.drill = handler; },
      highlight() {}, reset() {}, resize() {}, zoomBy() {}, fit() {},
      setStatuses: statuses => { this.statuses = statuses; return true; },
    };
    const window = {location: {search: ''}, addEventListener() {}};
    for (const source of ['response', 'scene']) {
      new Function('window', BrowserSource.read(path.join(WEB, 'core', source + '.js')))(window);
    }
    const panels = {define: (name, factory) => { this.factory = factory; }};
    const fetch = async url => {
      const response = this.responses[url];
      assert.notEqual(response, undefined, 'unexpected request: ' + url);
      const payload = await (typeof response === 'function' ? response() : response);
      return {ok: true, status: 200, json: async () => payload};
    };
    new Function('Panels', 'Graph', 'fetch', 'ResponseUtil', 'SceneContext', 'window',
      'setInterval', 'clearInterval', BrowserSource.read(path.join(WEB, 'panels/graph/panel.js')))(
      panels, graph, fetch, window.ResponseUtil, window.SceneContext, window,
      handler => { this.poll = handler; return 1; }, () => { this.poll = null; }
    );
    this.instance = this.factory(this.root, this.bus);
  }

  /** Return a control whose click listener can be exercised without a browser. */
  element(selector) {
    if (!this.elements.has(selector)) {
      this.elements.set(selector, {
        style: {}, classList: {toggle() {}, add() {}, remove() {}},
        querySelector() {}, querySelectorAll() { return []; },
        addEventListener(name, handler) { this[name] = handler; },
      });
    }
    return this.elements.get(selector);
  }

  /** Construct a recorded view with an expandable entity. */
  static payload(name, live) {
    return {
      ok: true, nodes: [{id: name, label: name}], edges: [],
      details: {[name]: {label: name}}, live,
    };
  }

  /** Construct a bridge response whose identity and status distinguish updates. */
  static live(name) {
    return {signature: name, nodes: [{id: name, kind: 'MotionNode', label: name, status: name}]};
  }

  /** Select a tab and finish immediately available response continuations. */
  async show(name) {
    this.element('tab-' + name).click();
    await this.flush();
  }

  /** Wait for the production async callbacks to settle. */
  async flush() {
    await new Promise(resolve => setImmediate(resolve));
  }

  /** Attach or detach the streaming bridge through the production event handler. */
  live(on, url = 'http://bridge') {
    this.bus.emit('live:changed', {on, url});
  }

  /** Return the identities shown in the most recently built graph. */
  identities() {
    return this.rendered.nodes.map(node => node.id);
  }
}

// %% live response ownership
test('a pending live plan cannot replace another tab', async context => {
  const panel = new GraphNavigation();
  context.after(() => panel.instance.destroy());
  const pending = new PendingResponse();
  panel.responses['http://bridge/plan'] = pending.promise;
  await panel.show('plan');
  panel.live(true);
  await panel.show('knowledge');
  pending.resolve(GraphNavigation.live('late'));
  await panel.flush();
  assert.deepEqual(panel.identities(), ['knowledge']);
});

test('a detached bridge cannot overwrite the restored recording', async context => {
  const panel = new GraphNavigation();
  context.after(() => panel.instance.destroy());
  const pending = new PendingResponse();
  panel.responses['http://bridge/plan'] = pending.promise;
  await panel.show('plan');
  panel.live(true);
  panel.live(false);
  await panel.flush();
  pending.resolve(GraphNavigation.live('late'));
  await panel.flush();
  assert.deepEqual(panel.identities(), ['recorded']);
});

test('an old bridge cannot overwrite the replacement bridge', async context => {
  const panel = new GraphNavigation();
  context.after(() => panel.instance.destroy());
  const pending = new PendingResponse();
  panel.responses['http://bridge/plan'] = pending.promise;
  panel.responses['http://replacement/plan'] = GraphNavigation.live('replacement');
  await panel.show('plan');
  panel.live(true);
  panel.live(true, 'http://replacement');
  await panel.flush();
  pending.resolve(GraphNavigation.live('late'));
  await panel.flush();
  assert.deepEqual(panel.identities(), ['replacement']);
});

test('an older poll cannot replace a more recent response', async context => {
  const panel = new GraphNavigation();
  context.after(() => panel.instance.destroy());
  const older = new PendingResponse();
  const newer = new PendingResponse();
  const responses = [older.promise, newer.promise];
  panel.responses['http://bridge/plan'] = () => responses.shift();
  await panel.show('plan');
  panel.live(true);
  panel.poll();
  newer.resolve(GraphNavigation.live('newer'));
  await panel.flush();
  older.resolve(GraphNavigation.live('older'));
  await panel.flush();
  assert.deepEqual(panel.identities(), ['newer']);
});

test('a slow response is still shown while the following poll is pending', async context => {
  const panel = new GraphNavigation();
  context.after(() => panel.instance.destroy());
  const older = new PendingResponse();
  const newer = new PendingResponse();
  const responses = [older.promise, newer.promise];
  panel.responses['http://bridge/plan'] = () => responses.shift();
  await panel.show('plan');
  panel.live(true);
  panel.poll();
  older.resolve(GraphNavigation.live('older'));
  await panel.flush();
  assert.deepEqual(panel.identities(), ['older']);
  newer.resolve(GraphNavigation.live('newer'));
  await panel.flush();
  assert.deepEqual(panel.identities(), ['newer']);
});

// %% drill response ownership
test('a pending drill cannot replace another tab', async context => {
  const panel = new GraphNavigation();
  context.after(() => panel.instance.destroy());
  const pending = new PendingResponse();
  panel.responses['/api/knowledge/expand?node=knowledge'] = pending.promise;
  await panel.flush();
  const drilling = panel.drill('knowledge');
  await panel.show('plan');
  pending.resolve(GraphNavigation.payload('inside'));
  await drilling;
  assert.deepEqual(panel.identities(), ['recorded']);
});

test('returning to the same tab does not revive a cancelled drill', async context => {
  const panel = new GraphNavigation();
  context.after(() => panel.instance.destroy());
  const pending = new PendingResponse();
  panel.responses['/api/knowledge/expand?node=knowledge'] = pending.promise;
  await panel.flush();
  const drilling = panel.drill('knowledge');
  await panel.show('plan');
  await panel.show('knowledge');
  pending.resolve(GraphNavigation.payload('inside'));
  await drilling;
  assert.deepEqual(panel.identities(), ['knowledge']);
});

test('a second drill supersedes the first without an extra breadcrumb', async context => {
  const panel = new GraphNavigation();
  context.after(() => panel.instance.destroy());
  const older = new PendingResponse();
  const newer = new PendingResponse();
  const responses = [older.promise, newer.promise];
  panel.responses['/api/knowledge/expand?node=knowledge'] = () => responses.shift();
  await panel.flush();
  const first = panel.drill('knowledge');
  const second = panel.drill('knowledge');
  newer.resolve(GraphNavigation.payload('newer'));
  await second;
  older.resolve(GraphNavigation.payload('older'));
  await first;
  assert.deepEqual(panel.identities(), ['newer']);
  panel.element('#gnav-up').click();
  assert.deepEqual(panel.identities(), ['knowledge']);
});

test('destroyed panels ignore pending bridge responses', async () => {
  const panel = new GraphNavigation();
  const pending = new PendingResponse();
  panel.responses['http://bridge/plan'] = pending.promise;
  await panel.show('plan');
  panel.live(true);
  panel.instance.destroy();
  pending.resolve(GraphNavigation.live('late'));
  await panel.flush();
  assert.deepEqual(panel.identities(), ['recorded']);
});
