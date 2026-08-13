// Unit tests for web/core/panel-arrangement.js (node:test): where each panel sits.
// A stored arrangement is cleaned against the configuration - vanished ids drop,
// new ids appear at their configured spot - and moving a panel lands it at the
// requested slot position exactly once.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  const scope = {};
  new Function('window', fs.readFileSync(path.join(WEB, 'core/panel-arrangement.js'), 'utf8'))(scope);
  return scope.PanelArrangement;
}

function memoryStorage(initial) {
  const values = Object.assign({}, initial);
  return {
    getItem: function (key) { return key in values ? values[key] : null; },
    setItem: function (key, value) { values[key] = String(value); },
  };
}

const CONFIGURED = { left: ['robot-scene'], right: ['eql', 'graph'] };

test('without a stored arrangement the configured layout stands', function () {
  const arrangement = load();
  assert.deepStrictEqual(arrangement.read(memoryStorage(), CONFIGURED), CONFIGURED);
});

test('a stored arrangement survives a reload', function () {
  const arrangement = load();
  const storage = memoryStorage();
  const rearranged = { left: ['eql', 'robot-scene'], right: ['graph'] };
  arrangement.write(storage, rearranged);
  assert.deepStrictEqual(arrangement.read(storage, CONFIGURED), rearranged);
});

test('an id that left the configuration is dropped', function () {
  const arrangement = load();
  assert.deepStrictEqual(
    arrangement.normalize({ left: ['gone', 'robot-scene'], right: ['eql', 'graph'] }, CONFIGURED),
    CONFIGURED);
});

test('an id new to the configuration appears at its configured spot', function () {
  const arrangement = load();
  assert.deepStrictEqual(
    arrangement.normalize({ left: ['robot-scene'], right: ['eql'] }, CONFIGURED),
    { left: ['robot-scene'], right: ['eql', 'graph'] });
});

test('a duplicated stored id is placed once', function () {
  const arrangement = load();
  assert.deepStrictEqual(
    arrangement.normalize({ left: ['eql', 'eql'], right: ['robot-scene', 'graph'] }, CONFIGURED),
    { left: ['eql'], right: ['robot-scene', 'graph'] });
});

test('moving a panel lands it at the requested slot position', function () {
  const arrangement = load();
  assert.deepStrictEqual(
    arrangement.moved(CONFIGURED, 'robot-scene', 'right', 1),
    { left: [], right: ['eql', 'robot-scene', 'graph'] });
});

test('a move index past the end clamps to the end', function () {
  const arrangement = load();
  assert.deepStrictEqual(
    arrangement.moved(CONFIGURED, 'eql', 'left', 99),
    { left: ['robot-scene', 'eql'], right: ['graph'] });
});

test('an unreadable stored value means the configured layout', function () {
  const arrangement = load();
  const storage = memoryStorage({ 'cramera.panel-arrangement': '{broken' });
  assert.deepStrictEqual(arrangement.read(storage, CONFIGURED), CONFIGURED);
});
