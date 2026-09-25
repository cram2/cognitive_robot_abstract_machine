const BrowserSource = require('./browser_source');
// Unit tests for web/core/scene_picker.js (node:test).
// There is no independent robot/environment mixing: each onboarded scene is a fixed
// (robot, environment) recording. ScenePicker only looks up, among the scenes actually
// onboarded, which one matches a chosen (robot, environment) pair, so the header's two
// dropdowns can look independent while only ever landing on a recorded combination.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  global.window = {};
  new Function(BrowserSource.read(path.join(WEB, 'core/scene_picker.js')))();
}

const SCENES = [
  { name: 'pr2_kitchen', robot: 'pr2', environment: 'apartment', task: 'make breakfast' },
  { name: 'pr2_lab', robot: 'pr2', environment: 'lab' },
  { name: 'garmi_apartment', robot: 'garmi', environment: 'apartment' },
  { name: 'tracy_lab', robot: 'tracy', environment: null },
];

test('robots() lists every robot with an onboarded scene, once each', function () {
  load();
  assert.deepStrictEqual(window.ScenePicker.robots(SCENES), ['pr2', 'garmi', 'tracy']);
});

test('environments() lists the environments onboarded with one robot', function () {
  load();
  assert.deepStrictEqual(window.ScenePicker.environments(SCENES, 'pr2'), ['apartment', 'lab']);
});

test('environments() includes null for a bench-only scene', function () {
  load();
  assert.deepStrictEqual(window.ScenePicker.environments(SCENES, 'tracy'), [null]);
});

test('environments() is empty for a robot with no onboarded scene', function () {
  load();
  assert.deepStrictEqual(window.ScenePicker.environments(SCENES, 'unitreeg1'), []);
});

test('sceneFor() resolves an onboarded (robot, environment) pair to its scene', function () {
  load();
  assert.strictEqual(window.ScenePicker.sceneFor(SCENES, 'pr2', 'lab'), 'pr2_lab');
});

test('sceneFor() resolves a bench-only robot with a null environment', function () {
  load();
  assert.strictEqual(window.ScenePicker.sceneFor(SCENES, 'tracy', null), 'tracy_lab');
});

test('sceneFor() treats a missing environment argument as null', function () {
  load();
  assert.strictEqual(window.ScenePicker.sceneFor(SCENES, 'tracy'), 'tracy_lab');
});

test('sceneFor() is null for a combination that was never onboarded', function () {
  load();
  assert.strictEqual(window.ScenePicker.sceneFor(SCENES, 'garmi', 'lab'), null);
});

test('describe() reports the (robot, environment) a named scene was recorded with', function () {
  load();
  assert.deepStrictEqual(window.ScenePicker.describe(SCENES, 'garmi_apartment'), {
    robot: 'garmi',
    environment: 'apartment',
  });
});

test('describe() reports a null environment for a bench-only scene', function () {
  load();
  assert.deepStrictEqual(window.ScenePicker.describe(SCENES, 'tracy_lab'), {
    robot: 'tracy',
    environment: null,
  });
});

test('describe() is null for a name not in the index', function () {
  load();
  assert.strictEqual(window.ScenePicker.describe(SCENES, 'unknown_scene'), null);
});

test('options() lists every scene alphabetically by name, labelled by its task', function () {
  load();
  assert.deepStrictEqual(
    window.ScenePicker.options(SCENES),
    [
      { name: 'garmi_apartment', task: 'garmi_apartment' },
      { name: 'pr2_kitchen', task: 'make breakfast' },
      { name: 'pr2_lab', task: 'pr2_lab' },
      { name: 'tracy_lab', task: 'tracy_lab' },
    ]);
});

test('options() falls back to the scene name when no task was recorded', function () {
  load();
  const untasked = [{ name: 'run_1', robot: 'pr2', environment: 'environment', task: null }];
  assert.deepStrictEqual(window.ScenePicker.options(untasked), [{ name: 'run_1', task: 'run_1' }]);
});

test('options() disambiguates scenes that share one (robot, environment) pair', function () {
  // several saved live recordings of the same demo: sceneFor() alone could only ever
  // resolve to the first one, so options() is what the picker falls back to
  load();
  const recordings = [
    { name: 'run_1', robot: 'pr2', environment: 'environment', task: 'transport' },
    { name: 'run_2', robot: 'pr2', environment: 'environment', task: 'transport' },
  ];
  assert.deepStrictEqual(window.ScenePicker.options(recordings), [
    { name: 'run_1', task: 'transport' },
    { name: 'run_2', task: 'transport' },
  ]);
});

test('options() is empty without any scenes', function () {
  load();
  assert.deepStrictEqual(window.ScenePicker.options([]), []);
  assert.deepStrictEqual(window.ScenePicker.options(undefined), []);
});
