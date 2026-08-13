// Unit tests for web/core/collada-mesh.js (node:test).
// THREE.ColladaLoader auto-rotates a <up_axis>Z_UP</up_axis> asset's scene -90 deg
// about X so it stands alone correctly in three.js's Y-up world. A mesh loaded as
// part of a URDF must NOT carry that rotation: the URDF tree already gets a single
// Z-up -> Y-up correction at its world root, so an uncorrected Z_UP mesh (e.g. the
// AWS RoboMaker warehouse assets) ends up rotated twice.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  global.window = {};
  new Function(fs.readFileSync(path.join(WEB, 'core/collada-mesh.js'), 'utf8'))();
}

function fakeColladaScene() {
  return {
    quaternion: {
      x: 0.70710678, y: 0, z: 0, w: 0.70710678,
      identity: function () { this.x = 0; this.y = 0; this.z = 0; this.w = 1; },
    },
    scale: { x: 0.01, y: 0.01, z: 0.01 },
  };
}

test('neutralizeUpAxisRotation() resets the up-axis rotation to identity', function () {
  load();
  const scene = fakeColladaScene();
  window.ColladaMeshUtil.neutralizeUpAxisRotation(scene);
  assert.deepStrictEqual(
    { x: scene.quaternion.x, y: scene.quaternion.y, z: scene.quaternion.z, w: scene.quaternion.w },
    { x: 0, y: 0, z: 0, w: 1 }
  );
});

test('neutralizeUpAxisRotation() leaves the unit-conversion scale untouched', function () {
  load();
  const scene = fakeColladaScene();
  window.ColladaMeshUtil.neutralizeUpAxisRotation(scene);
  assert.deepStrictEqual(scene.scale, { x: 0.01, y: 0.01, z: 0.01 });
});

test('neutralizeUpAxisRotation() returns the same scene it was given', function () {
  load();
  const scene = fakeColladaScene();
  assert.strictEqual(window.ColladaMeshUtil.neutralizeUpAxisRotation(scene), scene);
});
