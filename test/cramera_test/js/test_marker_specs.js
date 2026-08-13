// Unit tests for web/core/marker-specs.js (node:test): mapping a published CRAM
// debug marker to the build instruction the 3D code executes — which primitive,
// and what the visualization_msgs scale means for it.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  const scope = {};
  new Function('window', fs.readFileSync(path.join(WEB, 'core/marker-specs.js'), 'utf8'))(scope);
  return scope.MarkerSpecs;
}

test('a cube keeps its extents', function () {
  const spec = load().buildSpec({ kind: 'cube', scale: [0.2, 0.3, 0.4], color: '#ff0000', opacity: 0.5, pose: [1, 2, 3, 0, 0, 0, 1] });
  assert.strictEqual(spec.type, 'box');
  assert.deepStrictEqual(spec.size, [0.2, 0.3, 0.4]);
  assert.strictEqual(spec.opacity, 0.5);
  assert.deepStrictEqual(spec.pose, [1, 2, 3, 0, 0, 0, 1]);
});

test('an arrow reads length and diameters off the scale', function () {
  const spec = load().buildSpec({ kind: 'arrow', scale: [0.5, 0.02, 0.04] });
  assert.strictEqual(spec.type, 'arrow');
  assert.strictEqual(spec.length, 0.5);
  assert.strictEqual(spec.shaftDiameter, 0.02);
  assert.strictEqual(spec.headDiameter, 0.08);
});

test('line strips and lists carry their points', function () {
  const specs = load();
  const points = [[0, 0, 0], [1, 0, 0]];
  assert.strictEqual(specs.buildSpec({ kind: 'line_strip', points: points }).type, 'line');
  assert.strictEqual(specs.buildSpec({ kind: 'line_list', points: points }).type, 'segments');
  assert.deepStrictEqual(specs.buildSpec({ kind: 'line_strip', points: points }).points, points);
});

test('point clouds keep the point size and list shape', function () {
  const spec = load().buildSpec({ kind: 'sphere_list', points: [[0, 0, 0]], scale: [0.05, 0, 0] });
  assert.strictEqual(spec.type, 'points');
  assert.strictEqual(spec.size, 0.05);
  assert.strictEqual(spec.shape, 'sphere');
});

test('unrenderable entries are filtered out of the batch', function () {
  const specs = load().buildSpecs([
    { kind: 'cube', scale: [1, 1, 1] },
    { kind: 'never-heard-of-it' },
    null,
  ]);
  assert.strictEqual(specs.length, 1);
});
