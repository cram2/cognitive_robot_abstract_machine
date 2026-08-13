// Unit tests for web/core/graph-gestures.js (node:test): how wheel input is read.
//
// vis-network's own wheel handler zooms a flat 10% per event and ignores deltaMode,
// deltaX and ctrlKey, so on a laptop touchpad one two-finger swipe — dozens of tiny
// wheel events — multiplies the zoom by 1.1 per event and the graph shoots away.
// These tests pin the replacement: two-finger scrolling pans, a pinch or a mouse
// notch zooms, and the zoom step follows how far the wheel actually moved.
//
// The module is loaded with a bare scope object standing in for `window`, and driven
// against a pinhole camera model of vis-network's view API, so the anchored-zoom
// invariant (the graph point under the pointer does not move) is a real assertion
// rather than a recorded call.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  const scope = {};
  new Function('window', fs.readFileSync(path.join(WEB, 'core/graph-gestures.js'), 'utf8'))(scope);
  return scope.GraphGestures;
}

const WIDTH = 800;
const HEIGHT = 600;

// vis-network's view as the three calls the gestures use see it: a scale and the
// graph coordinate the container's centre is looking at
function makeNetwork() {
  const view = { scale: 1, centre: { x: 0, y: 0 } };
  return {
    view: view,
    getScale: function () { return view.scale; },
    getViewPosition: function () { return { x: view.centre.x, y: view.centre.y }; },
    DOMtoCanvas: function (point) {
      return { x: view.centre.x + (point.x - WIDTH / 2) / view.scale,
               y: view.centre.y + (point.y - HEIGHT / 2) / view.scale };
    },
    moveTo: function (options) {
      if (options.scale !== undefined) view.scale = options.scale;
      if (options.position) view.centre = { x: options.position.x, y: options.position.y };
    },
  };
}

function makeContainer() {
  const listeners = {};
  return {
    addEventListener: function (type, callback) { listeners[type] = callback; },
    removeEventListener: function (type) { delete listeners[type]; },
    getBoundingClientRect: function () { return { left: 0, top: 0, width: WIDTH, height: HEIGHT }; },
    wheel: function (event) { listeners.wheel(event); },
    handles: function (type) { return typeof listeners[type] === 'function'; },
  };
}

function wheelEvent(fields) {
  const event = Object.assign({ deltaX: 0, deltaY: 0, deltaMode: 0, ctrlKey: false, metaKey: false,
                                clientX: WIDTH / 2, clientY: HEIGHT / 2, prevented: 0 }, fields);
  event.preventDefault = function () { event.prevented += 1; };
  return event;
}

// a two-finger swipe as Chromium reports it: many small pixel deltas, both axes
function touchpadSwipe(count) {
  const events = [];
  for (let step = 0; step < count; step += 1) events.push(wheelEvent({ deltaX: -3, deltaY: 7.5 }));
  return events;
}

// a trackpad pinch: the browser reports it as a ctrl-held wheel
function pinch(count, deltaY) {
  const events = [];
  for (let step = 0; step < count; step += 1) events.push(wheelEvent({ deltaY: deltaY, ctrlKey: true }));
  return events;
}

// %% reading the intent off a wheel event
test('a plain fine-grained wheel is a two-finger scroll, so it pans', function () {
  const gestures = load();
  assert.strictEqual(gestures.intentOf(wheelEvent({ deltaY: 7.5 })), gestures.PAN);
  assert.strictEqual(gestures.intentOf(wheelEvent({ deltaY: -3 })), gestures.PAN);
});

test('a sideways wheel only comes from a touchpad, so it pans', function () {
  const gestures = load();
  assert.strictEqual(gestures.intentOf(wheelEvent({ deltaX: 120, deltaY: 0 })), gestures.PAN);
});

test('a ctrl-held wheel is a pinch, so it zooms', function () {
  const gestures = load();
  assert.strictEqual(gestures.intentOf(wheelEvent({ deltaY: 4, ctrlKey: true })), gestures.ZOOM);
  assert.strictEqual(gestures.intentOf(wheelEvent({ deltaY: 4, metaKey: true })), gestures.ZOOM);
});

test('a wheel reporting lines or pages is a stepped mouse wheel, so it zooms', function () {
  const gestures = load();
  assert.strictEqual(gestures.intentOf(wheelEvent({ deltaY: 3, deltaMode: 1 })), gestures.ZOOM);
  assert.strictEqual(gestures.intentOf(wheelEvent({ deltaY: 1, deltaMode: 2 })), gestures.ZOOM);
});

test('a whole-notch pixel wheel is a mouse wheel, so it zooms', function () {
  const gestures = load();
  assert.strictEqual(gestures.intentOf(wheelEvent({ deltaY: 100 })), gestures.ZOOM);
  assert.strictEqual(gestures.intentOf(wheelEvent({ deltaY: -300 })), gestures.ZOOM);
});

// %% delta normalisation
test('deltas are normalised to pixels whatever unit the wheel reports', function () {
  const gestures = load();
  assert.deepStrictEqual(gestures.pixelsOf(wheelEvent({ deltaX: 5, deltaY: -9 })), { x: 5, y: -9 });
  assert.deepStrictEqual(gestures.pixelsOf(wheelEvent({ deltaY: 2, deltaMode: 1 })),
    { x: 0, y: 2 * gestures.LINE_PIXELS });
  assert.deepStrictEqual(gestures.pixelsOf(wheelEvent({ deltaY: 1, deltaMode: 2 })),
    { x: 0, y: gestures.PAGE_PIXELS });
});

// %% the zoom step
test('the zoom step follows the distance the wheel moved', function () {
  const gestures = load();
  const notch = gestures.scaleAfter(1, wheelEvent({ deltaY: -100 }));
  const tick = gestures.scaleAfter(1, wheelEvent({ deltaY: -4, ctrlKey: true }));
  assert.ok(notch > 1.2 && notch < 1.4, 'a mouse notch is a visible step, got ' + notch);
  assert.ok(tick > 1 && tick < 1.02, 'a pinch tick is a fine step, got ' + tick);
});

test('zooming out then in by the same distance returns to the same scale', function () {
  const gestures = load();
  const out = gestures.scaleAfter(1, wheelEvent({ deltaY: 60, ctrlKey: true }));
  const back = gestures.scaleAfter(out, wheelEvent({ deltaY: -60, ctrlKey: true }));
  assert.ok(Math.abs(back - 1) < 1e-12, 'expected 1, got ' + back);
});

test('the scale stops at the bounds', function () {
  const gestures = load();
  const far = wheelEvent({ deltaY: 100000, ctrlKey: true });
  const near = wheelEvent({ deltaY: -100000, ctrlKey: true });
  assert.strictEqual(gestures.scaleAfter(1, far), gestures.MIN_SCALE);
  assert.strictEqual(gestures.scaleAfter(1, near), gestures.MAX_SCALE);
});

test('a view already fitted past a bound stays put rather than jumping back', function () {
  const gestures = load();
  const fitted = gestures.MIN_SCALE / 4;   // network.fit() may zoom out below the floor
  assert.strictEqual(gestures.scaleAfter(fitted, wheelEvent({ deltaY: 200, ctrlKey: true })), fitted);
  assert.ok(gestures.scaleAfter(fitted, wheelEvent({ deltaY: -200, ctrlKey: true })) > fitted,
    'zooming back in must still work');
});

// %% installed on a network
test('the installed handler takes the page scroll over', function () {
  const gestures = load();
  const container = makeContainer();
  gestures.install(makeNetwork(), container);
  assert.ok(container.handles('wheel'));
  const event = wheelEvent({ deltaY: 7.5 });
  container.wheel(event);
  assert.strictEqual(event.prevented, 1);
});

test('a two-finger swipe pans by the distance scrolled and never zooms', function () {
  const gestures = load();
  const network = makeNetwork();
  const container = makeContainer();
  gestures.install(network, container);
  container.wheel(wheelEvent({ deltaX: -30, deltaY: 60 }));
  assert.strictEqual(network.getScale(), 1);
  assert.deepStrictEqual(network.getViewPosition(), { x: -30, y: 60 });
});

test('panning follows the pointer by the same screen distance at any zoom', function () {
  const gestures = load();
  const network = makeNetwork();
  const container = makeContainer();
  network.moveTo({ scale: 4 });
  gestures.install(network, container);
  container.wheel(wheelEvent({ deltaX: -30, deltaY: 60 }));
  // 60 screen pixels at scale 4 is 15 graph units, so the graph slides 60 pixels
  assert.deepStrictEqual(network.getViewPosition(), { x: -7.5, y: 15 });
  assert.strictEqual(network.getScale(), 4);
});

test('a whole two-finger swipe leaves the zoom untouched', function () {
  const gestures = load();
  const network = makeNetwork();
  const container = makeContainer();
  gestures.install(network, container);
  touchpadSwipe(40).forEach(container.wheel);
  assert.strictEqual(network.getScale(), 1, 'a swipe must not zoom at all');
  assert.deepStrictEqual(network.getViewPosition(), { x: -120, y: 300 });
});

test('a pinch keeps the graph point under the pointer where it is', function () {
  const gestures = load();
  const network = makeNetwork();
  const container = makeContainer();
  gestures.install(network, container);
  const pointer = { x: 200, y: 100 };
  const before = network.DOMtoCanvas(pointer);
  container.wheel(wheelEvent({ deltaY: -50, ctrlKey: true, clientX: pointer.x, clientY: pointer.y }));
  const after = network.DOMtoCanvas(pointer);
  assert.ok(network.getScale() > 1, 'the pinch must zoom in');
  assert.ok(Math.abs(after.x - before.x) < 1e-9 && Math.abs(after.y - before.y) < 1e-9,
    'anchor moved from ' + JSON.stringify(before) + ' to ' + JSON.stringify(after));
});

test('a long pinch stays inside the zoom bounds', function () {
  const gestures = load();
  const network = makeNetwork();
  const container = makeContainer();
  gestures.install(network, container);
  pinch(200, -40).forEach(container.wheel);
  assert.strictEqual(network.getScale(), gestures.MAX_SCALE);
  pinch(400, 40).forEach(container.wheel);
  assert.strictEqual(network.getScale(), gestures.MIN_SCALE);
});

// %% the zoom buttons
test('zoomBy steps the scale about the centre and respects the bounds', function () {
  const gestures = load();
  const network = makeNetwork();
  gestures.zoomBy(network, 2);
  assert.strictEqual(network.getScale(), 2);
  assert.deepStrictEqual(network.getViewPosition(), { x: 0, y: 0 });
  gestures.zoomBy(network, 1000);
  assert.strictEqual(network.getScale(), gestures.MAX_SCALE);
  gestures.zoomBy(network, 0.00001);
  assert.strictEqual(network.getScale(), gestures.MIN_SCALE);
});
