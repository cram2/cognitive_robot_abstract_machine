// Unit tests for web/core/live-mode.js (node:test): where the live pose stream may attach.
//
// A recorded scene and a running demo are separate worlds, and an object drag means a
// different thing in each -- a client-side offset on a playback pose versus a real move of
// the simulated world posted to the bridge. So the stream attaches only on the reserved
// live scene, and the control offers a navigation rather than a toggle anywhere else.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  const scope = {};
  new Function('window', fs.readFileSync(path.join(WEB, 'core/live-mode.js'), 'utf8'))(scope);
  return scope.LiveMode;
}

// %% where the stream may attach
test('the pose stream attaches only on the live scene', function () {
  const live = load();
  assert.strictEqual(live.attachable(live.SCENE_NAME), true);
  assert.strictEqual(live.attachable('PR2_Apartment'), false);
});

test('a landing page that names no scene is not attachable either', function () {
  /*
   * With no ?scene= the viewer is showing the scene index's default, which is a recording
   * like any other.
   */
  const live = load();
  assert.strictEqual(live.attachable(''), false);
  assert.strictEqual(live.attachable(null), false);
  assert.strictEqual(live.attachable(undefined), false);
});

// %% what the control does
test('pressing live on a recorded scene navigates rather than attaching', function () {
  const live = load();
  assert.strictEqual(live.actionFor('PR2_Apartment'), live.NAVIGATE);
  assert.strictEqual(live.actionFor(''), live.NAVIGATE);
});

test('pressing live on the live scene toggles the stream in place', function () {
  const live = load();
  assert.strictEqual(live.actionFor(live.SCENE_NAME), live.TOGGLE);
});

test('navigating and toggling are distinguishable', function () {
  const live = load();
  assert.notStrictEqual(live.NAVIGATE, live.TOGGLE);
});

// %% what the control says
test('a recorded scene is offered a view switch, not an attach', function () {
  const live = load();
  assert.strictEqual(live.labelFor('PR2_Apartment', false), '◉ Live view');
  assert.match(live.titleFor('PR2_Apartment'), /leaves the recorded scene/);
});

test('the live scene reports whether it is attached', function () {
  const live = load();
  assert.notStrictEqual(
    live.labelFor(live.SCENE_NAME, true), live.labelFor(live.SCENE_NAME, false));
  assert.match(live.labelFor(live.SCENE_NAME, true), /attached/);
});

// %% when the live scene must be rebuilt and reloaded
function boundScene(modelCount) {
  const models = [];
  for (let i = 0; i < modelCount; i++) models.push({ name: 'model' + i });
  return { models: models, worldBound: true };
}

test('the model count alone never triggers a reload — the signature decides', function () {
  const live = load();
  assert.strictEqual(
    live.needsLiveSceneReload(live.SCENE_NAME, true, boundScene(1), { running: true, modelVersion: 2 }),
    false);
  assert.strictEqual(
    live.needsLiveSceneReload(live.SCENE_NAME, true, boundScene(2), { running: true, modelVersion: 1 }),
    false);
});

test('a bundle built before the world attached reloads once the demo is running', function () {
  /*
   * Without the composed world the models' instance prefixes and the robot cannot be
   * identified at bundle time -- joints route nowhere and the base never moves. The
   * bundle must be rebuilt against the attached world. A bundle that does not carry
   * the flag at all was written by code from before it existed and is just as
   * suspect, so only an explicit true counts as bound.
   */
  const live = load();
  const early = { models: [{ name: 'pr2' }, { name: 'apartment' }], worldBound: false };
  assert.strictEqual(
    live.needsLiveSceneReload(live.SCENE_NAME, true, early, { running: true, modelVersion: 2 }),
    true);
  assert.strictEqual(
    live.needsLiveSceneReload(live.SCENE_NAME, true, early, { running: false, modelVersion: 2 }),
    false);
  const unflagged = { models: [{ name: 'pr2' }, { name: 'apartment' }] };
  assert.strictEqual(
    live.needsLiveSceneReload(live.SCENE_NAME, true, unflagged, { running: true, modelVersion: 2 }),
    true);
});

test('a recorded scene is never reloaded away by a live event', function () {
  const live = load();
  const early = { models: [], worldBound: false };
  assert.strictEqual(
    live.needsLiveSceneReload('PR2_Apartment', true, early, { running: true, modelVersion: 5 }),
    false);
  assert.strictEqual(
    live.needsLiveSceneReload(live.SCENE_NAME, false, early, { running: true, modelVersion: 5 }),
    false);
});

test('missing scene or info never triggers a reload', function () {
  const live = load();
  assert.strictEqual(live.needsLiveSceneReload(live.SCENE_NAME, true, null, { running: true }), false);
  assert.strictEqual(live.needsLiveSceneReload(live.SCENE_NAME, true, boundScene(1), null), false);
});

test('a bundle whose signature no longer matches the bridge reloads', function () {
  /*
   * A demo can switch to a different world mid-run with the same number of sources;
   * only the signature (which encodes prefixes, robot and world binding) catches
   * that the loaded bundle describes a world that no longer runs.
   */
  const live = load();
  const scene = { models: [{}, {}], worldBound: true, bundleSignature: 'world-a' };
  assert.strictEqual(
    live.needsLiveSceneReload(live.SCENE_NAME, true, scene,
      { running: true, modelVersion: 2, bundleSignature: 'world-b' }),
    true);
  assert.strictEqual(
    live.needsLiveSceneReload(live.SCENE_NAME, true, scene,
      { running: true, modelVersion: 2, bundleSignature: 'world-a' }),
    false);
});

// %% whether a fresh bundle matches the loaded one
test('a rebundle with the same models and robot changes nothing', function () {
  const live = load();
  const scene = { models: [{ name: 'pr2', prefix: 'pr2' }], robot: { name: 'pr2' }, worldBound: true };
  const fresh = { models: [{ name: 'pr2', prefix: 'pr2' }], robot: { name: 'pr2' }, worldBound: true };
  assert.strictEqual(live.sameBundle(scene, fresh), true);
});

test('a rebundle whose models or robot differ demands a reload', function () {
  /*
   * A page can sit on the live scene across demo runs: attaching to a new demo must
   * not keep showing the previous run's robot or prefixes.
   */
  const live = load();
  const loaded = { models: [{ name: 'pr2', prefix: '' }], robot: null };
  const fresh = { models: [{ name: 'pr2', prefix: 'pr2' }], robot: { name: 'pr2' } };
  assert.strictEqual(live.sameBundle(loaded, fresh), false);
});

test('an unloaded page or a failed rebundle compares as equal', function () {
  /*
   * With nothing to compare there is no evidence of staleness, and reloading on none
   * would loop.
   */
  const live = load();
  assert.strictEqual(live.sameBundle(null, { models: [] }), true);
  assert.strictEqual(live.sameBundle({ models: [] }, null), true);
});

// %% when the stream attaches by itself
test('auto-attach fires once per page by default', function () {
  const live = load();
  assert.strictEqual(live.shouldAutoAttach(live.SCENE_NAME, true, false, false, false), true);
  assert.strictEqual(live.shouldAutoAttach(live.SCENE_NAME, true, false, true, false), false);
});

test('always-live keeps re-attaching whenever the stream is down', function () {
  /*
   * A demo ending detaches the stream after enough failed polls; with always-live on,
   * the next demo run must attach again without a click -- the "attached once
   * already" guard only applies to the one-shot default.
   */
  const live = load();
  assert.strictEqual(live.shouldAutoAttach(live.SCENE_NAME, true, false, true, true), true);
  assert.strictEqual(live.shouldAutoAttach(live.SCENE_NAME, true, true, true, true), false);
});

test('a deliberately chosen recorded scene is never auto-attached away from', function () {
  /*
   * A URL naming a recorded scene is a deliberate choice; even always-live must not
   * navigate away from it. The landing page (no explicit scene) is fair game.
   */
  const live = load();
  assert.strictEqual(live.shouldAutoAttach('PR2_Apartment', true, false, false, true), false);
  assert.strictEqual(live.shouldAutoAttach('', false, false, false, true), true);
});
