// Unit tests for web/core/scene.js (node:test).
// SceneContext is how every panel learns which onboarded scene the URL names, so its
// API requests target the scene actually shown in the viewer instead of the server's
// fallback default.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function loadWithSearch(search) {
  global.window = { location: { search: search } };
  new Function(fs.readFileSync(path.join(WEB, 'core/scene.js'), 'utf8'))();
}

test('name() is null without a ?scene= param', function () {
  loadWithSearch('');
  assert.strictEqual(window.SceneContext.name(), null);
});

test('name() reads the ?scene= param', function () {
  loadWithSearch('?scene=g1-warehouse');
  assert.strictEqual(window.SceneContext.name(), 'g1-warehouse');
});

test('name() reads ?scene= alongside other params', function () {
  loadWithSearch('?foo=bar&scene=g1-warehouse&baz=qux');
  assert.strictEqual(window.SceneContext.name(), 'g1-warehouse');
});

test('withScene() leaves a url unchanged without an active scene', function () {
  loadWithSearch('');
  assert.strictEqual(window.SceneContext.withScene('/api/kb'), '/api/kb');
});

test('withScene() appends ?scene= to a bare url', function () {
  loadWithSearch('?scene=g1-warehouse');
  assert.strictEqual(window.SceneContext.withScene('/api/kb'), '/api/kb?scene=g1-warehouse');
});

test('withScene() appends &scene= to a url that already has a query string', function () {
  loadWithSearch('?scene=g1-warehouse');
  assert.strictEqual(
    window.SceneContext.withScene('/api/kb/view?name=plan'),
    '/api/kb/view?name=plan&scene=g1-warehouse'
  );
});
