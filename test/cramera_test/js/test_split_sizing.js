// Unit tests for web/core/split-sizing.js (node:test): the pane geometry both the
// column divider (scene vs. knowledge column) and the row divider (EQL vs. graph)
// share. The module is pure arithmetic and touches no DOM, so it is loaded with a
// bare scope object standing in for `window`.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  const scope = {};
  new Function('window', fs.readFileSync(path.join(WEB, 'core/split-sizing.js'), 'utf8'))(scope);
  return scope.SplitSizing;
}

// %% dragging
test('a pointer in the middle splits the container evenly', function () {
  assert.strictEqual(load().secondPaneFraction(1000, 500), 0.5);
});

test('the second pane gets the space between the pointer and the far edge', function () {
  assert.strictEqual(load().secondPaneFraction(1000, 300), 0.7);
});

test('dragging past the near edge keeps the first pane at the minimum', function () {
  const sizing = load();
  const fraction = sizing.secondPaneFraction(1000, -200);
  assert.strictEqual(fraction, (1000 - sizing.MIN_PANE_PIXELS) / 1000);
});

test('dragging past the far edge keeps the second pane at the minimum', function () {
  const sizing = load();
  const fraction = sizing.secondPaneFraction(1000, 1200);
  assert.strictEqual(fraction, sizing.MIN_PANE_PIXELS / 1000);
});

test('a container too small for two minimum panes splits evenly instead', function () {
  const sizing = load();
  assert.strictEqual(sizing.secondPaneFraction(sizing.MIN_PANE_PIXELS, 0), 0.5);
});

test('a container of no measurable size splits evenly', function () {
  assert.strictEqual(load().secondPaneFraction(0, 0), 0.5);
});

// %% restoring a stored fraction
test('a stored fraction that still fits is kept as it is', function () {
  assert.strictEqual(load().clampFraction(1000, 0.4), 0.4);
});

test('a stored fraction that would starve the first pane is pulled back', function () {
  const sizing = load();
  assert.strictEqual(sizing.clampFraction(1000, 0.99), (1000 - sizing.MIN_PANE_PIXELS) / 1000);
});

test('a stored fraction that would starve the second pane is pulled back', function () {
  const sizing = load();
  assert.strictEqual(sizing.clampFraction(1000, 0.01), sizing.MIN_PANE_PIXELS / 1000);
});

// %% the grid template
test('the template gives the second pane its fraction of the track space', function () {
  assert.strictEqual(load().template(0.35), 'minmax(0,65fr) auto minmax(0,35fr)');
});

test('template sizes are rounded to a tenth of a percent', function () {
  assert.strictEqual(load().template(1 / 3), 'minmax(0,66.7fr) auto minmax(0,33.3fr)');
});
