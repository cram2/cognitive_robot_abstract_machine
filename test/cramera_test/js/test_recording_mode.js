const BrowserSource = require('./browser_source');
// Unit tests for web/core/recording-mode.js (node:test): the recording controls'
// state machine, independent of the DOM they end up wired into.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  const scope = {};
  new Function('window', BrowserSource.read(path.join(WEB, 'core/recording-mode.js')))(scope);
  return scope.RecordingMode;
}

// %% recognizing the reserved scene
test('the recording scene is named __recording__', function () {
  const recording = load();
  assert.strictEqual(recording.isRecordingScene(recording.SCENE_NAME), true);
  assert.strictEqual(recording.isRecordingScene('PR2_Apartment'), false);
});

// %% whether the controls show at all
test('idle or missing status hides the controls', function () {
  const recording = load();
  assert.strictEqual(recording.controlsVisible(null), false);
  assert.strictEqual(recording.controlsVisible({ state: recording.STATE.IDLE }), false);
});

test('a recording or a finalized one shows the controls', function () {
  const recording = load();
  assert.strictEqual(recording.controlsVisible({ state: recording.STATE.RECORDING }), true);
  assert.strictEqual(recording.controlsVisible({ state: recording.STATE.FINALIZED }), true);
});

// %% stop button
test('the stop button reads "stop recording" only while capture runs', function () {
  const recording = load();
  assert.strictEqual(recording.stopButtonLabel(recording.STATE.RECORDING), '⏹ Stop recording');
  assert.strictEqual(recording.stopButtonLabel(recording.STATE.FINALIZED), '⏹ Stopped');
  assert.strictEqual(recording.stopButtonLabel(recording.STATE.IDLE), '⏹ Stopped');
});

test('stop only does anything while recording', function () {
  const recording = load();
  assert.strictEqual(recording.canStop(recording.STATE.RECORDING), true);
  assert.strictEqual(recording.canStop(recording.STATE.FINALIZED), false);
  assert.strictEqual(recording.canStop(recording.STATE.IDLE), false);
});

// %% save / discard availability
test('save is only offered once capture is finalized', function () {
  const recording = load();
  assert.strictEqual(recording.canSave(recording.STATE.FINALIZED), true);
  assert.strictEqual(recording.canSave(recording.STATE.RECORDING), false);
  assert.strictEqual(recording.canSave(recording.STATE.IDLE), false);
});

test('discard is offered whenever there is something to throw away', function () {
  const recording = load();
  assert.strictEqual(recording.canDiscard(recording.STATE.RECORDING), true);
  assert.strictEqual(recording.canDiscard(recording.STATE.FINALIZED), true);
  assert.strictEqual(recording.canDiscard(recording.STATE.IDLE), false);
});

// %% save name validation
test('a trailing line break is outside the scene-name alphabet', function () {
  const recording = load();
  for (const suffix of ['\n', '\r', '\r\n']) {
    assert.strictEqual(recording.isValidSaveName('episode' + suffix), false);
  }
});

test('a plain name is accepted', function () {
  const recording = load();
  assert.strictEqual(recording.isValidSaveName('kitchen_run-2'), true);
});

test('an empty, oversized or non-string name is rejected', function () {
  const recording = load();
  assert.strictEqual(recording.isValidSaveName(''), false);
  assert.strictEqual(recording.isValidSaveName(null), false);
  assert.strictEqual(recording.isValidSaveName(undefined), false);
  assert.strictEqual(recording.isValidSaveName('a'.repeat(65)), false);
});

test('a name with unsafe characters is rejected', function () {
  const recording = load();
  assert.strictEqual(recording.isValidSaveName('../escape'), false);
  assert.strictEqual(recording.isValidSaveName('has space'), false);
  assert.strictEqual(recording.isValidSaveName('slash/here'), false);
});

test('the reserved recording name itself is rejected', function () {
  const recording = load();
  assert.strictEqual(recording.isValidSaveName(recording.SCENE_NAME), false);
});

// %% playback speed
test('a listed speed passes through unchanged', function () {
  const recording = load();
  recording.SPEED_OPTIONS.forEach(function (speed) {
    assert.strictEqual(recording.clampSpeed(speed), speed);
  });
});

test('an unlisted speed falls back to 1x', function () {
  const recording = load();
  assert.strictEqual(recording.clampSpeed(3), 1);
  assert.strictEqual(recording.clampSpeed(undefined), 1);
});

// %% the single episode control
test('the control offers to stop while capture runs, and to save once finalized', function () {
  const recording = load();
  assert.strictEqual(recording.controlLabel(recording.STATE.RECORDING), '⏹ Stop recording');
  assert.strictEqual(recording.controlLabel(recording.STATE.FINALIZED), '💾 Save episode…');
});

test('the control does nothing while idle', function () {
  const recording = load();
  assert.strictEqual(recording.controlAction(recording.STATE.IDLE), null);
});

test('pressing the control stops capture, then opens the save panel', function () {
  const recording = load();
  assert.strictEqual(recording.controlAction(recording.STATE.RECORDING), recording.STOP);
  assert.strictEqual(recording.controlAction(recording.STATE.FINALIZED), recording.SAVE);
});

// %% trimming before saving
test('the default trim keeps the whole recording', function () {
  const recording = load();
  assert.deepStrictEqual(recording.wholeTrim(5), { first: 0, last: 4 });
});

test('a range inside the recording is valid', function () {
  const recording = load();
  assert.strictEqual(recording.isValidTrim({ first: 0, last: 4 }, 5), true);
  assert.strictEqual(recording.isValidTrim({ first: 2, last: 2 }, 5), true);
});

test('a reversed or out-of-bounds range is rejected', function () {
  const recording = load();
  assert.strictEqual(recording.isValidTrim({ first: 3, last: 1 }, 5), false);
  assert.strictEqual(recording.isValidTrim({ first: -1, last: 4 }, 5), false);
  assert.strictEqual(recording.isValidTrim({ first: 0, last: 5 }, 5), false);
  assert.strictEqual(recording.isValidTrim({ first: 0, last: 0 }, 0), false);
});

test('a trim covering everything is reported as no trim at all', function () {
  const recording = load();
  assert.strictEqual(recording.isWholeTrim({ first: 0, last: 4 }, 5), true);
  assert.strictEqual(recording.isWholeTrim({ first: 1, last: 4 }, 5), false);
});

test('a trim is described by how much of the run it keeps', function () {
  const recording = load();
  assert.strictEqual(recording.trimSummary({ first: 0, last: 9 }, 10, 20), '10 frames · 0.50 s');
  assert.strictEqual(recording.trimSummary({ first: 5, last: 9 }, 10, 20), '5 frames · 0.25 s');
});

test('only the captured episode itself can be trimmed', function () {
  const recording = load();
  // another scene's trajectory is a different run: its frame numbers would cut the
  // episode at the wrong places
  assert.strictEqual(recording.canTrim('pr2_scuess', 500), false);
  assert.strictEqual(recording.canTrim(recording.SCENE_NAME, 500), true);
});

test('an episode the viewer has no trajectory for can only be saved whole', function () {
  const recording = load();
  assert.strictEqual(recording.canTrim(recording.SCENE_NAME, 0), false);
});

// %% where a saved episode lands
test('the destinations mirror the scene roots the backend writes to', function () {
  const recording = load();
  assert.deepStrictEqual(recording.DESTINATION, { LOCAL: 'local', SHARED: 'shared' });
});
