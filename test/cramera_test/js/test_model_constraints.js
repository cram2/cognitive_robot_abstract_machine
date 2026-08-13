// Unit tests for web/core/model-constraints.js (node:test): mapping the Models tab's
// constraint rows to the workbench API payload. Incomplete rows mean "unconstrained"
// and are skipped; numeric bounds are ordered; symbolic selections travel as values.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  const scope = {};
  new Function('window', fs.readFileSync(path.join(WEB, 'core/model-constraints.js'), 'utf8'))(scope);
  return scope.ModelConstraints;
}

test('a numeric row becomes an ordered closed interval', function () {
  const constraints = load();
  assert.deepStrictEqual(
    constraints.constraintOf({ variable: 'x', kind: 'continuous', low: '0.8', high: '0.2' }),
    { variable: 'x', intervals: [[0.2, 0.8]] });
});

test('a symbolic row carries its selected values', function () {
  const constraints = load();
  assert.deepStrictEqual(
    constraints.constraintOf({ variable: 'color', kind: 'symbolic', values: ['red', 'blue'] }),
    { variable: 'color', values: ['red', 'blue'] });
});

test('incomplete rows are skipped rather than sent', function () {
  const constraints = load();
  assert.strictEqual(constraints.constraintOf(null), null);
  assert.strictEqual(constraints.constraintOf({ variable: '' }), null);
  assert.strictEqual(constraints.constraintOf({ variable: 'x', kind: 'continuous', low: '', high: '1' }), null);
  assert.strictEqual(constraints.constraintOf({ variable: 'color', kind: 'symbolic', values: [] }), null);
});

test('the payload keeps only the complete rows', function () {
  const constraints = load();
  const rows = [
    { variable: 'x', kind: 'continuous', low: '0', high: '1' },
    { variable: '' },
    { variable: 'color', kind: 'symbolic', values: ['red'] },
  ];
  assert.deepStrictEqual(constraints.payload(rows), [
    { variable: 'x', intervals: [[0, 1]] },
    { variable: 'color', values: ['red'] },
  ]);
});

test('constraints describe themselves readably', function () {
  const constraints = load();
  assert.strictEqual(constraints.describe({ variable: 'x', intervals: [[0, 1]] }), 'x ∈ [0, 1]');
  assert.strictEqual(constraints.describe({ variable: 'color', values: ['red'] }), 'color ∈ {red}');
});
