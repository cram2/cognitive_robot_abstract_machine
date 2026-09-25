const BrowserSource = require('./browser_source');
// Unit tests for web/core/completion.js (node:test).
// The query box completes what a query may name: the token under the caret decides what
// is being completed, and the vocabulary decides what is offered for it.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  global.window = {};
  new Function(BrowserSource.read(path.join(WEB, 'core/completion.js')))();
  return global.window.Completion;
}

const VOCABULARY = [
  { name: 'scene_object', kind: 'variable', detail: 'ranges over BenchObject', type: 'BenchObject' },
  { name: 'entity', kind: 'factory', detail: 'One entity of a domain.' },
  { name: 'BenchObject', kind: 'entity_type', detail: 'An object on the bench.', module: 'cramera.knowledge.entities' },
  { name: 'Body', kind: 'class', detail: 'A body.', module: 'semantic_digital_twin.world_description.world_entity' },
  { name: 'BodyCollisionCheck', kind: 'class', detail: 'A check.', module: 'semantic_digital_twin.collision_checking' },
  { name: 'objects', kind: 'value', detail: 'list' },
];

const MEMBERS = [
  { name: 'name', kind: 'field', detail: 'str' },
  { name: 'position', kind: 'field', detail: 'Point3' },
  { name: 'global_pose', kind: 'property', detail: 'Pose of this body.' },
  { name: 'has_collision', kind: 'method', detail: 'Whether it collides.' },
];

function names(suggested) {
  return suggested.map(function (entry) { return entry.name; });
}

// %% what is being completed

test('tokenAt() reads the word being typed at the caret', function () {
  const completion = load();
  const token = completion.tokenAt('the(entity(Bo', 13);
  assert.strictEqual(token.prefix, 'Bo');
  assert.strictEqual(token.start, 11);
  assert.strictEqual(token.end, 13);
  assert.strictEqual(token.owner, '');
});

test('tokenAt() reads an empty prefix where a word may start', function () {
  const completion = load();
  assert.strictEqual(load().tokenAt('the(', 4).prefix, '');
  assert.strictEqual(completion.tokenAt('the(', 4).owner, '');
});

test('tokenAt() names the owner of a member being typed after a dot', function () {
  const completion = load();
  const token = completion.tokenAt('scene_object.nam', 16);
  assert.strictEqual(token.prefix, 'nam');
  assert.strictEqual(token.owner, 'scene_object');
  assert.strictEqual(token.start, 13);
});

test('tokenAt() names the owner right after the dot, before anything is typed', function () {
  const completion = load();
  const token = completion.tokenAt('where(scene_object.', 19);
  assert.strictEqual(token.prefix, '');
  assert.strictEqual(token.owner, 'scene_object');
});

test('tokenAt() takes the nearest name of a dotted chain as the owner', function () {
  const completion = load();
  assert.strictEqual(load().tokenAt('a.body.na', 9).owner, 'body');
});

test('tokenAt() ignores text after the caret', function () {
  const completion = load();
  const token = completion.tokenAt('the(entity(Body))', 13);
  assert.strictEqual(token.prefix, 'Bo');
  assert.strictEqual(token.end, 13);
});

// %% what is offered for it

test('suggest() offers the names starting with the typed prefix', function () {
  const completion = load();
  const suggested = completion.suggest(VOCABULARY, completion.tokenAt('Bo', 2), 10);
  assert.deepStrictEqual(names(suggested).slice(0, 2), ['Body', 'BodyCollisionCheck']);
});

test('suggest() ignores case while typing a prefix', function () {
  const completion = load();
  const suggested = completion.suggest(VOCABULARY, completion.tokenAt('bod', 3), 10);
  assert.deepStrictEqual(names(suggested), ['Body', 'BodyCollisionCheck']);
});

test('suggest() matches the capitals of a name typed as initials', function () {
  const completion = load();
  const suggested = completion.suggest(VOCABULARY, completion.tokenAt('BCC', 3), 10);
  assert.deepStrictEqual(names(suggested), ['BodyCollisionCheck']);
});

test('suggest() offers a prefix match before a name that only contains the prefix', function () {
  const completion = load();
  const suggested = completion.suggest(VOCABULARY, completion.tokenAt('object', 6), 10);
  assert.deepStrictEqual(names(suggested), ['objects', 'scene_object', 'BenchObject']);
});

test('suggest() offers the ready-made variables before the keywords and classes', function () {
  const completion = load();
  const suggested = completion.suggest(VOCABULARY, completion.tokenAt('', 0), 10);
  assert.deepStrictEqual(names(suggested).slice(0, 3), ['scene_object', 'entity', 'BenchObject']);
});

test('suggest() offers a field of the owner before its properties and methods', function () {
  const completion = load();
  const suggested = completion.suggest(MEMBERS, completion.tokenAt('scene_object.', 13), 10);
  assert.deepStrictEqual(names(suggested), ['name', 'position', 'global_pose', 'has_collision']);
});

test('suggest() offers nothing for a prefix nothing matches', function () {
  const completion = load();
  assert.deepStrictEqual(completion.suggest(VOCABULARY, completion.tokenAt('zzz', 3), 10), []);
});

test('suggest() offers no more than it was asked for', function () {
  const completion = load();
  assert.strictEqual(completion.suggest(VOCABULARY, completion.tokenAt('', 0), 2).length, 2);
});

// %% accepting one

test('applied() replaces the typed prefix with the accepted name', function () {
  const completion = load();
  const applied = completion.applied('the(entity(Bo', completion.tokenAt('the(entity(Bo', 13), { name: 'Body' });
  assert.strictEqual(applied.text, 'the(entity(Body');
  assert.strictEqual(applied.caret, 15);
});

test('applied() keeps what follows the caret', function () {
  const completion = load();
  const applied = completion.applied('the(entity(Bo))', completion.tokenAt('the(entity(Bo))', 13), { name: 'Body' });
  assert.strictEqual(applied.text, 'the(entity(Body))');
  assert.strictEqual(applied.caret, 15);
});

test('applied() inserts a member where nothing was typed after the dot', function () {
  const completion = load();
  const applied = completion.applied('scene_object.', completion.tokenAt('scene_object.', 13), { name: 'name' });
  assert.strictEqual(applied.text, 'scene_object.name');
  assert.strictEqual(applied.caret, 17);
});
