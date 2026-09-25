const BrowserSource = require('./browser_source');
// Unit tests for core/camera-follow.js (node:test): whether the scene camera keeps the
// moving robot in view, and that the choice survives a reload.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function loadCameraFollow() {
  const scope = {};
  new Function('window', BrowserSource.read(path.join(WEB, 'core/camera-follow.js')))(scope);
  return scope.CameraFollow;
}

function makeStorage(initial) {
  const items = Object.assign({}, initial);
  return {
    items: items,
    getItem(key) { return key in items ? items[key] : null; },
    setItem(key, value) { items[key] = String(value); },
  };
}

// %% the default
test('a viewer that has never touched the switch follows the robot', function () {
  const CameraFollow = loadCameraFollow();

  assert.strictEqual(CameraFollow.on(makeStorage()), true);
});

// %% remembering the choice
test('switching the follow off survives a reload', function () {
  const CameraFollow = loadCameraFollow();
  const storage = makeStorage();

  const stored = CameraFollow.set(storage, false);

  assert.strictEqual(stored, false);
  assert.strictEqual(CameraFollow.on(storage), false);
});

test('switching the follow back on survives a reload', function () {
  const CameraFollow = loadCameraFollow();
  const storage = makeStorage();
  CameraFollow.set(storage, false);

  const stored = CameraFollow.set(storage, true);

  assert.strictEqual(stored, true);
  assert.strictEqual(CameraFollow.on(storage), true);
});

test('the choice is kept under the module\'s own key', function () {
  const CameraFollow = loadCameraFollow();
  const storage = makeStorage();

  CameraFollow.set(storage, false);

  assert.deepStrictEqual(Object.keys(storage.items), [CameraFollow.KEY]);
});
