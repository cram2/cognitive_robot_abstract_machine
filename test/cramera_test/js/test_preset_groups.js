const BrowserSource = require('./browser_source');
// Unit tests for web/core/preset_groups.js (node:test).
// Ready-made questions are about different bodies of knowledge — what is true of the run
// right now, and what its finished runs recorded — and are offered under a heading each.
// A scene whose questions are all about one thing keeps a single unlabelled row.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  global.window = {};
  new Function(BrowserSource.read(path.join(WEB, 'core/preset_groups.js')))();
  return window.PresetGroups;
}

const LIVE = { text: 'which shapes are inserted?', code: 'a', scope: 'current_state' };
const STORED = { text: 'success rate per shape', code: 'b', scope: 'episodic_memory' };
const SCOPES = [
  { name: 'current_state', label: 'Current State Queries', variables: ['shape'] },
  { name: 'episodic_memory', label: 'Episodic Memory Queries', variables: ['result'] },
];

test('questions about different knowledge get a heading each', function () {
  const groups = load().of([LIVE, STORED], SCOPES);
  assert.deepStrictEqual(
    groups.map(function (group) { return group.label; }),
    ['Current State Queries', 'Episodic Memory Queries']
  );
  assert.deepStrictEqual(groups[1].presets, [STORED]);
});

test('the demo decides the order the headings come in', function () {
  const groups = load().of([STORED, LIVE], SCOPES);
  assert.deepStrictEqual(
    groups.map(function (group) { return group.name; }),
    ['current_state', 'episodic_memory']
  );
});

test('a heading with no questions under it is left out', function () {
  const groups = load().of([LIVE], SCOPES);
  assert.deepStrictEqual(groups.map(function (group) { return group.name; }), ['current_state']);
});

test('questions all about one thing stay one unlabelled row', function () {
  // every other scene's presets: headings would be noise above a single group
  const groups = load().of([LIVE, { text: 'which arms?', code: 'c', scope: 'current_state' }], SCOPES);
  assert.strictEqual(groups.length, 1);
  assert.strictEqual(groups[0].label, null);
  assert.strictEqual(groups[0].presets.length, 2);
});

test('presets that name no scope at all are one unlabelled row', function () {
  const groups = load().of([{ text: 'which robot is this?', code: 'd' }], null);
  assert.strictEqual(groups.length, 1);
  assert.strictEqual(groups[0].label, null);
});

test('a scope the payload never described is still given a readable heading', function () {
  const groups = load().of([LIVE, STORED], null);
  assert.deepStrictEqual(
    groups.map(function (group) { return group.label; }),
    ['Current State Queries', 'Episodic Memory Queries']
  );
});

test('no presets is no groups', function () {
  assert.deepStrictEqual(load().of([], SCOPES), []);
});
