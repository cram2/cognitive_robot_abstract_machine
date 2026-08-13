// Unit tests for web/core/panel-visibility.js (node:test): the View menu's state
// rules — configured panels default to visible, stored choices persist, ids that
// left the configuration are dropped, and slots collapse when everything in them
// is hidden.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  const scope = {};
  new Function('window', fs.readFileSync(path.join(WEB, 'core/panel-visibility.js'), 'utf8'))(scope);
  return scope.PanelVisibility;
}

function memoryStorage(initial) {
  const values = Object.assign({}, initial);
  return {
    getItem: function (key) { return key in values ? values[key] : null; },
    setItem: function (key, value) { values[key] = String(value); },
  };
}

test('everything configured defaults to visible', function () {
  const visibility = load();
  assert.deepStrictEqual(
    visibility.read(memoryStorage(), ['robot-scene', 'eql']),
    { 'robot-scene': true, 'eql': true });
});

test('a stored choice survives a reload', function () {
  const visibility = load();
  const storage = memoryStorage();
  visibility.write(storage, { 'robot-scene': true, 'eql': false });
  assert.deepStrictEqual(
    visibility.read(storage, ['robot-scene', 'eql']),
    { 'robot-scene': true, 'eql': false });
});

test('a stored id that is no longer configured is dropped', function () {
  const visibility = load();
  const storage = memoryStorage();
  visibility.write(storage, { 'gone-panel': false });
  assert.deepStrictEqual(visibility.read(storage, ['eql']), { 'eql': true });
});

test('an unreadable stored value means defaults, not a broken page', function () {
  const visibility = load();
  const storage = memoryStorage({ 'cramera.visible-panels': 'not json {' });
  assert.deepStrictEqual(visibility.read(storage, ['eql']), { 'eql': true });
});

test('a slot collapses when everything in it is hidden', function () {
  const visibility = load();
  const layout = { left: ['robot-scene'], right: ['eql', 'graph'] };
  assert.deepStrictEqual(
    visibility.visibleSlots(layout, { 'robot-scene': false, 'eql': true, 'graph': false }),
    { right: ['eql'] });
});

test('panels have readable menu labels', function () {
  const visibility = load();
  assert.strictEqual(visibility.labelOf('robot-scene'), 'Semantic Digital Twin Scene');
  assert.strictEqual(visibility.labelOf('something-custom'), 'something-custom');
});

test('slot names get readable menu labels', function () {
  const visibility = load();
  assert.strictEqual(visibility.slotLabel('left'), 'Left');
  assert.strictEqual(visibility.slotLabel('right'), 'Right');
});
