// Unit tests for web/core/marker-settings.js (node:test): the persisted marker
// settings — hidden namespaces filter the overlay client-side, and the user's topic
// choices survive to be re-applied on the next demo attach.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  const scope = {};
  new Function('window', fs.readFileSync(path.join(WEB, 'core/marker-settings.js'), 'utf8'))(scope);
  return scope.MarkerSettings;
}

function memoryStorage(initial) {
  const values = Object.assign({}, initial);
  return {
    getItem: function (key) { return key in values ? values[key] : null; },
    setItem: function (key, value) { values[key] = String(value); },
  };
}

test('hiding a namespace filters its markers and persists', function () {
  const settings = load();
  const storage = memoryStorage();
  const hidden = settings.setNamespaceHidden(storage, 'debug', true);
  const markers = [{ ns: 'debug' }, { ns: 'costmap' }];
  assert.deepStrictEqual(settings.visibleMarkers(markers, hidden), [{ ns: 'costmap' }]);
  assert.deepStrictEqual(settings.hiddenNamespaces(storage), { debug: true });
});

test('re-showing a namespace removes it from the hidden set', function () {
  const settings = load();
  const storage = memoryStorage();
  settings.setNamespaceHidden(storage, 'debug', true);
  const hidden = settings.setNamespaceHidden(storage, 'debug', false);
  assert.deepStrictEqual(hidden, {});
});

test('namespaces are listed sorted and unique', function () {
  const settings = load();
  assert.deepStrictEqual(
    settings.namespacesOf([{ ns: 'b' }, { ns: 'a' }, { ns: 'b' }]),
    ['a', 'b']);
});

test('topic choices persist as overrides', function () {
  const settings = load();
  const storage = memoryStorage();
  settings.setTopicOverride(storage, '/giskard/markers', true);
  settings.setTopicOverride(storage, '/semworld/viz_marker', false);
  assert.deepStrictEqual(settings.topicOverrides(storage), {
    '/giskard/markers': true,
    '/semworld/viz_marker': false,
  });
});

test('unreadable stored values mean defaults', function () {
  const settings = load();
  const storage = memoryStorage({
    'cramera.hidden-marker-namespaces': '{oops',
    'cramera.marker-topic-overrides': '[not a map]',
  });
  assert.deepStrictEqual(settings.hiddenNamespaces(storage), {});
  assert.deepStrictEqual(settings.topicOverrides(storage), {});
});
