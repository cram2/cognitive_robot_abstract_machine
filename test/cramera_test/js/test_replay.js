const BrowserSource = require('./browser_source');
// Unit tests for web/core/replay.js (node:test).
// The replay popup's behaviour hangs on three pure pieces: reading the window out of
// the URL, building the popup URL, and mapping playback time onto recorded frames.
// All three are checkable here, without a browser or a bridge.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  global.window = {};
  new Function(BrowserSource.read(path.join(WEB, 'core/replay.js')))();
  return window.Replay;
}

function frames(...ats) {
  return ats.map(function (at, index) {
    return { at: at, frames: {}, base: null, objects: { marker: index } };
  });
}

// %% reading the window out of the URL
test('a replay window is read out of the search string', function () {
  assert.deepStrictEqual(load().fromSearch('?replay=100.5,110.5'), { start: 100.5, end: 110.5 });
});

test('a page without a replay parameter is not a replay page', function () {
  assert.strictEqual(load().fromSearch('?scene=montessori'), null);
  assert.strictEqual(load().fromSearch(''), null);
  assert.strictEqual(load().fromSearch(undefined), null);
});

test('an unusable window reads as no window rather than breaking the page', function () {
  const Replay = load();
  assert.strictEqual(Replay.fromSearch('?replay=abc,def'), null);
  assert.strictEqual(Replay.fromSearch('?replay=100'), null);
  assert.strictEqual(Replay.fromSearch('?replay=110,100'), null);
  assert.strictEqual(Replay.fromSearch('?replay=100,100'), null);
});

// %% building the popup URL
test('the popup URL carries the window on the viewer page itself', function () {
  const url = load().popupUrl('/index.html', '', { start: 100, end: 110 });
  assert.strictEqual(url, '/index.html?replay=100,110');
});

test('an explicit bridge address travels into the popup', function () {
  const url = load().popupUrl('/index.html', '?live=robot-host:8765', { start: 100, end: 110 });
  assert.strictEqual(url, '/index.html?replay=100,110&live=robot-host:8765');
});

test('the popup URL round-trips through fromSearch', function () {
  const Replay = load();
  const url = Replay.popupUrl('/index.html', '', { start: 100.25, end: 110.75 });
  assert.deepStrictEqual(Replay.fromSearch(url.slice(url.indexOf('?'))), { start: 100.25, end: 110.75 });
});

// %% mapping playback time onto frames
test('playback starts on the first recorded frame', function () {
  assert.strictEqual(load().frameAt(frames(100, 101, 102), 0).at, 100);
});

test('playback shows the newest frame not later than the playback time', function () {
  assert.strictEqual(load().frameAt(frames(100, 101, 102), 1.5).at, 101);
});

test('playback holds the last frame before looping', function () {
  // the clip runs 2 s; at 2.5 s elapsed the loop hold keeps the last frame on screen
  assert.strictEqual(load().frameAt(frames(100, 101, 102), 2.5).at, 102);
});

test('playback loops back to the first frame after the hold', function () {
  // 2 s of clip + 1 s hold: at 3.1 s the playback is 0.1 s into its second lap
  assert.strictEqual(load().frameAt(frames(100, 101, 102), 3.1).at, 100);
});

test('a single-frame clip stays on that frame', function () {
  assert.strictEqual(load().frameAt(frames(100), 5).at, 100);
});

test('an empty clip yields no frame', function () {
  assert.strictEqual(load().frameAt([], 1), null);
  assert.strictEqual(load().frameAt(null, 1), null);
});

test('the clip duration spans first to last frame', function () {
  const Replay = load();
  assert.strictEqual(Replay.duration(frames(100, 101, 102.5)), 2.5);
  assert.strictEqual(Replay.duration([]), 0);
});

// %% naming the clip
test('the label reads as a wall-clock time span', function () {
  // hours and minutes depend on the zone the viewer runs in; the shape and the
  // seconds digits do not
  const label = load().label({ start: 1755086425, end: 1755086435 });
  assert.match(label, /^\d{2}:\d{2}:25 – \d{2}:\d{2}:35$/);
});

// %% the scene the answer was given about
test('the scene the opener is showing travels into the popup', function () {
  const url = load().popupUrl('/index.html', '?scene=pr2_breakfast', { start: 10, end: 12 });
  assert.strictEqual(url, '/index.html?replay=10,12&scene=pr2_breakfast');
});

test('a popup of a recorded scene carries the scene and no bridge', function () {
  const url = load().popupUrl('/index.html', '?scene=pr2_breakfast&other=1', { start: 1, end: 2 });
  assert.ok(!url.includes('live='), url);
  assert.ok(url.includes('scene=pr2_breakfast'), url);
});
