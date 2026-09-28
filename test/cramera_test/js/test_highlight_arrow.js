const BrowserSource = require('./browser_source');
// Unit tests for web/core/highlight_arrow.js (node:test).
// The arrow bouncing over a highlighted object is driven by two pure pieces: where the
// arrow rests above the object, and how far it has bobbed at a given moment. Both are
// checkable here, without a browser or a 3D scene.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  global.window = {};
  new Function(BrowserSource.read(path.join(WEB, 'core/highlight_arrow.js')))();
  return window.HighlightArrow;
}

// %% resting place above the object
test('the arrow rests clear of the object top by its clearance plus half its height', function () {
  const arrow = load();
  assert.strictEqual(arrow.restAltitude(0.2), 0.2 + arrow.CLEARANCE + arrow.HEIGHT / 2);
});

test('a taller object lifts the arrow by the same amount', function () {
  const arrow = load();
  assert.strictEqual(arrow.restAltitude(1.5) - arrow.restAltitude(0.5), 1.0);
});

// %% the bob over time
test('the bob starts at rest', function () {
  assert.strictEqual(load().bobOffset(0), 0);
});

test('the bob peaks at its amplitude halfway through a period', function () {
  const arrow = load();
  assert.ok(Math.abs(arrow.bobOffset(arrow.BOB_PERIOD_SECONDS / 2) - arrow.BOB_AMPLITUDE) < 1e-9);
});

test('the bob returns to rest after a full period', function () {
  const arrow = load();
  assert.ok(Math.abs(arrow.bobOffset(arrow.BOB_PERIOD_SECONDS)) < 1e-9);
});

test('the bob never leaves the band between rest and its amplitude', function () {
  const arrow = load();
  for (let step = 0; step <= 100; step++) {
    const offset = arrow.bobOffset(step * 0.037);
    assert.ok(offset >= 0 && offset <= arrow.BOB_AMPLITUDE + 1e-9, 'at step ' + step);
  }
});

test('the bob repeats with its period', function () {
  const arrow = load();
  assert.ok(Math.abs(arrow.bobOffset(0.4) - arrow.bobOffset(0.4 + arrow.BOB_PERIOD_SECONDS)) < 1e-9);
});
