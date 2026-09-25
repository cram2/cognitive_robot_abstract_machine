const BrowserSource = require('./browser_source');
// Unit tests for web/core/query_source.js (node:test).
// The EQL panel answers from the recorded scene or from a running demo; this is the one
// place that decides which, so the panel itself never has to know both URL shapes.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load(search) {
  global.window = { location: { search: search } };
  new Function(BrowserSource.read(path.join(WEB, 'core/scene.js')))();
  new Function(BrowserSource.read(path.join(WEB, 'core/query_source.js')))();
  return window.QuerySource;
}

test('without a live demo the recorded scene answers', function () {
  const source = load('').of({ on: false, url: 'http://localhost:8765' });
  assert.strictEqual(source.live, false);
  assert.strictEqual(source.presetsUrl, '/api/knowledge');
  assert.strictEqual(source.runUrl, '/api/eql');
});

test('the recorded urls carry the active scene', function () {
  const source = load('?scene=Franka_Montessori').of({ on: false });
  assert.strictEqual(source.presetsUrl, '/api/knowledge?scene=Franka_Montessori');
  assert.strictEqual(source.runUrl, '/api/eql?scene=Franka_Montessori');
});

test('an attached demo answers instead of the recorded scene', function () {
  const source = load('?scene=Franka_Montessori').of({
    on: true,
    url: 'http://localhost:8765',
  });
  assert.strictEqual(source.live, true);
  assert.strictEqual(source.presetsUrl, 'http://localhost:8765/presets');
  assert.strictEqual(source.runUrl, 'http://localhost:8765/eql');
});

test('a live demo url is not given the scene parameter', function () {
  // the demo has no scene bundle; ?scene= would be meaningless to its bridge
  const source = load('?scene=Franka_Montessori').of({ on: true, url: 'http://host:1/' });
  assert.strictEqual(source.presetsUrl, 'http://host:1/presets');
  assert.strictEqual(source.runUrl, 'http://host:1/eql');
});

test('being attached without a url falls back to the recorded scene', function () {
  const source = load('').of({ on: true });
  assert.strictEqual(source.live, false);
  assert.strictEqual(source.runUrl, '/api/eql');
});

test('of() tolerates being called with nothing at all', function () {
  const source = load('').of();
  assert.strictEqual(source.live, false);
  assert.strictEqual(source.runUrl, '/api/eql');
});

// %% where the query box asks what it may name

test('the recorded vocabulary and members urls carry the active scene', function () {
  const source = load('?scene=Franka_Montessori').of({ on: false });
  assert.strictEqual(source.vocabularyUrl(), '/api/eql/vocabulary?scene=Franka_Montessori');
  assert.strictEqual(
    source.membersUrl('scene_object'),
    '/api/eql/members?scene=Franka_Montessori&name=scene_object'
  );
});

test('an attached demo is asked what a query of one scope may name', function () {
  const source = load('').of({ on: true, url: 'http://localhost:8765' });
  assert.strictEqual(
    source.vocabularyUrl('episodic_memory'),
    'http://localhost:8765/vocabulary?scope=episodic_memory'
  );
  assert.strictEqual(
    source.membersUrl('shape', 'episodic_memory'),
    'http://localhost:8765/members?name=shape&scope=episodic_memory'
  );
});

test('a scope is left out of the urls when the panel names none', function () {
  const source = load('').of({ on: true, url: 'http://localhost:8765' });
  assert.strictEqual(source.vocabularyUrl(null), 'http://localhost:8765/vocabulary');
  assert.strictEqual(source.membersUrl('shape', null), 'http://localhost:8765/members?name=shape');
});
