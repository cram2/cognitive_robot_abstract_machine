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
  new Function('window', fs.readFileSync(path.join(WEB, 'core/recording-mode.js'), 'utf8'))(scope);
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
