const BrowserSource = require('./browser_source');
// Unit tests for web/core/ambient-occlusion.js (node:test).
// The SSAO pass renders the scene twice more with an override material to fill its
// depth and normal buffers. three.js puts a textured scene background into the render
// list as a 2x2 plane at the origin, so under an override material that plane becomes
// real geometry in those buffers, and its occlusion edges show up as a rectangle on the
// floor of every scene. The pass here hides the background for exactly those renders.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

// %% a stand-in for three's SSAOPass that records what its override renders see
class RecordingOverridePass {
  constructor(scene, camera, width, height) {
    this.scene = scene;
    this.camera = camera;
    this.width = width;
    this.height = height;
    this.backgroundsSeen = [];
    this.argumentsSeen = [];
  }
  renderOverride(renderer, overrideMaterial, renderTarget, clearColor, clearAlpha) {
    this.backgroundsSeen.push(this.scene.background);
    this.argumentsSeen.push([renderer, overrideMaterial, renderTarget, clearColor, clearAlpha]);
  }
}

function load(ssaoPass) {
  global.window = {};
  global.THREE = ssaoPass ? { SSAOPass: ssaoPass } : {};
  new Function(BrowserSource.read(path.join(WEB, 'core/ambient-occlusion.js')))();
  return window.BackgroundIgnoringSSAOPass;
}

function passOver(scene) {
  const Pass = load(RecordingOverridePass);
  return new Pass(scene, 'camera', 640, 480);
}

// %% the background is hidden from the depth and normal renders
test('the override renders see no background', function () {
  const scene = { background: 'gradient' };
  const pass = passOver(scene);
  pass.renderOverride('renderer', 'normal-material', 'target', 0x7777ff, 1.0);
  assert.deepStrictEqual(pass.backgroundsSeen, [null]);
});

test('the background is back on the scene once the override render is done', function () {
  const background = { isTexture: true };
  const scene = { background: background };
  const pass = passOver(scene);
  pass.renderOverride('renderer', 'depth-material', 'target', 0x000000, 1.0);
  assert.strictEqual(scene.background, background);
});

test('every render argument reaches the underlying pass unchanged', function () {
  const pass = passOver({ background: 'gradient' });
  pass.renderOverride('renderer', 'normal-material', 'target', 0x7777ff, 1.0);
  assert.deepStrictEqual(pass.argumentsSeen, [['renderer', 'normal-material', 'target', 0x7777ff, 1.0]]);
});

test('a scene without a background is left as it is', function () {
  const scene = { background: null };
  const pass = passOver(scene);
  pass.renderOverride('renderer', 'normal-material', 'target', 0x7777ff, 1.0);
  assert.deepStrictEqual(pass.backgroundsSeen, [null]);
  assert.strictEqual(scene.background, null);
});

// %% construction
test('the pass is built on three\'s SSAOPass with the same constructor arguments', function () {
  const pass = passOver({ background: null });
  assert.ok(pass instanceof RecordingOverridePass);
  assert.deepStrictEqual([pass.camera, pass.width, pass.height], ['camera', 640, 480]);
});

test('without three\'s SSAOPass on the page the pass is not defined', function () {
  assert.strictEqual(load(null), undefined);
});
