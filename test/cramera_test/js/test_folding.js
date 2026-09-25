const BrowserSource = require('./browser_source');
// Unit tests for web/core/folding.js (node:test): which sections the reader folded away.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  global.window = {};
  new Function(BrowserSource.read(path.join(WEB, 'core/folding.js')))();
  return window.Folding;
}

function store(initial) {
  const data = Object.assign({}, initial);
  return {
    getItem: function (key) { return key in data ? data[key] : null; },
    setItem: function (key, value) { data[key] = value; },
    data: data,
  };
}

function refusingStore() {
  return {
    getItem: function () { throw new Error('storage is off'); },
    setItem: function () { throw new Error('storage is off'); },
  };
}

// %% the remembered choice
test('a section starts unfolded', function () {
  assert.strictEqual(load().folded(store({}), 'layers'), false);
});

test('a remembered fold is honoured on the next page', function () {
  const Folding = load();
  const remembered = store({});

  Folding.remember(remembered, 'layers', true);

  assert.strictEqual(Folding.folded(remembered, 'layers'), true);
});

test('unfolding is remembered as well', function () {
  const Folding = load();
  const remembered = store({});
  Folding.remember(remembered, 'layers', true);

  Folding.remember(remembered, 'layers', false);

  assert.strictEqual(Folding.folded(remembered, 'layers'), false);
});

test('each section is remembered on its own', function () {
  const Folding = load();
  const remembered = store({});

  Folding.remember(remembered, 'presets:detected_events', true);

  assert.strictEqual(Folding.folded(remembered, 'presets:detected_events'), true);
  assert.strictEqual(Folding.folded(remembered, 'presets:current_state'), false);
  assert.strictEqual(Folding.folded(remembered, 'layers'), false);
});

test('a store that refuses to answer leaves the section open', function () {
  assert.strictEqual(load().folded(refusingStore(), 'layers'), false);
});

test('a store that refuses to remember does not break the fold', function () {
  load().remember(refusingStore(), 'layers', true);
});

// %% what the button says
test('the button offers to fold the named section away', function () {
  assert.deepStrictEqual(load().button(false, 'the layers'), {
    glyph: '▾',
    title: 'Fold the layers away',
  });
});

test('the button offers to show it again while it is folded', function () {
  assert.deepStrictEqual(load().button(true, 'the layers'), {
    glyph: '▸',
    title: 'Show the layers',
  });
});
