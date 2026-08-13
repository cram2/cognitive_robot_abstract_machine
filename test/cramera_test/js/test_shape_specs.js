// Unit tests for web/core/shape-specs.js (node:test): mapping a live overlay body's
// published shapes to the build instructions the 3D code executes.
//
// The bridge publishes any world body shape by shape, so the geometry decisions --
// which primitive, which axis convention, which URL -- live in one pure module and are
// pinned here without THREE.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  const scope = {};
  new Function('window', fs.readFileSync(path.join(WEB, 'core/shape-specs.js'), 'utf8'))(scope);
  return scope.ShapeSpecs;
}

// %% primitives
test('a box shape keeps its size, colour and local pose', function () {
  const specs = load();
  assert.deepStrictEqual(
    specs.buildSpec({
      kind: 'box', size: [0.2, 0.3, 0.4], color: '#cc3333', opacity: 1,
      position: [0.1, 0, 0.05], quaternion: [0, 0, 0, 1],
    }, '#123456', ''),
    {
      type: 'box', size: [0.2, 0.3, 0.4], color: '#cc3333', opacity: 1,
      position: [0.1, 0, 0.05], quaternion: [0, 0, 0, 1],
    });
});

test('a cylinder is rotated from the Y axis onto the Z axis the world uses', function () {
  const specs = load();
  const spec = specs.buildSpec({ kind: 'cylinder', radius: 0.05, height: 0.3 }, null, '');
  assert.strictEqual(spec.type, 'cylinder');
  assert.strictEqual(spec.radius, 0.05);
  assert.strictEqual(spec.height, 0.3);
  assert.strictEqual(spec.rotateXDegrees, 90);
});

test('a sphere keeps its radius', function () {
  const specs = load();
  const spec = specs.buildSpec({ kind: 'sphere', radius: 0.05 }, null, '');
  assert.strictEqual(spec.type, 'sphere');
  assert.strictEqual(spec.radius, 0.05);
});

// %% meshes
test('a mesh shape resolves its URL against the live bridge base', function () {
  const specs = load();
  const spec = specs.buildSpec(
    { kind: 'mesh', mesh: '/mesh?key=montessori%2Fboard%230', format: 'OBJ', scale: [1, 2, 3] },
    null, 'http://localhost:8123');
  assert.strictEqual(spec.type, 'mesh');
  assert.strictEqual(spec.url, 'http://localhost:8123/mesh?key=montessori%2Fboard%230');
  assert.strictEqual(spec.format, 'obj');
  assert.deepStrictEqual(spec.scale, [1, 2, 3]);
});

// %% degradation
test('an unusable shape degrades to a fallback box, never to nothing', function () {
  /*
   * The body exists in the running world, so it must occupy its place on screen even
   * when its entry is unreadable -- same reasoning as the bridge's own mesh fallback.
   */
  const specs = load();
  assert.strictEqual(specs.buildSpec({ kind: 'mesh' }, null, '').type, 'box');
  assert.strictEqual(specs.buildSpec({ kind: 'cylinder', radius: 0 }, null, '').type, 'box');
  assert.deepStrictEqual(specs.buildSpec({ kind: 'box' }, null, '').size, specs.FALLBACK_SIZE);
});

test('a shape without a colour of its own falls back to the object colour', function () {
  const specs = load();
  assert.strictEqual(specs.buildSpec({ kind: 'sphere', radius: 0.1 }, '#123456', '').color, '#123456');
  assert.strictEqual(specs.buildSpec({ kind: 'sphere', radius: 0.1 }, null, '').color, specs.FALLBACK_COLOR);
});

// %% whole objects
test('a shape list maps element-wise', function () {
  const specs = load();
  const result = specs.buildSpecs(
    [{ kind: 'box', size: [1, 1, 1] }, { kind: 'sphere', radius: 0.2 }], '#aabbcc', '');
  assert.deepStrictEqual(result.map(function (s) { return s.type; }), ['box', 'sphere']);
});
