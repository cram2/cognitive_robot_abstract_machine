const BrowserSource = require('./browser_source');
// Unit tests for web/core/answer_table.js (node:test).
// An EQL answer arrives as a list of row objects whose keys depend on what was asked.
// Turning that into one table with stable columns is what makes an answer readable, and
// it is pure enough to check here rather than by eye in the browser.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  global.window = {};
  new Function(BrowserSource.read(path.join(WEB, 'core/answer_table.js')))();
  return window.AnswerTable;
}

function cellTexts(row) {
  return row.cells.map(function (cell) { return cell.text; });
}

test('a set_of answer becomes one column per asked-for value', function () {
  const table = load().of([
    { name: 'cube', target_hole: 'square_hole' },
    { name: 'cylinder_1', target_hole: 'circular_hole_1' },
  ]);
  assert.deepStrictEqual(table.columns, ['name', 'target_hole']);
  assert.deepStrictEqual(cellTexts(table.rows[0]), ['cube', 'square_hole']);
  assert.deepStrictEqual(cellTexts(table.rows[1]), ['cylinder_1', 'circular_hole_1']);
});

test('a column only later rows carry is still given a place', function () {
  // grouped answers can leave a value out of the first row entirely
  const table = load().of([{ outcome: 'fell_through' }, { outcome: 'wedged', total: 2 }]);
  assert.deepStrictEqual(table.columns, ['outcome', 'total']);
  assert.deepStrictEqual(cellTexts(table.rows[0]), ['fell_through', '—']);
});

test('an entity answer leads with its name and lists its fields', function () {
  const table = load().of([
    { __entity__: 'cube', __type__: 'ShapeUnderTest', shape_key: 'square_hole', attempts: 2 },
  ]);
  assert.deepStrictEqual(table.columns, ['name', 'shape_key', 'attempts']);
  assert.strictEqual(table.rows[0].type, 'ShapeUnderTest');
  assert.deepStrictEqual(cellTexts(table.rows[0]), ['cube', 'square_hole', '2']);
});

test('rows that name no entity carry no type tag', function () {
  const table = load().of([{ name: 'cube' }]);
  assert.strictEqual(table.rows[0].type, null);
});

test('each value is classified so the table can colour it by what it is', function () {
  const table = load().of([
    { count: 3, sorted: true, missed: false, reason: 'wedged_in_hole', detail: null },
  ]);
  const kinds = {};
  table.columns.forEach(function (column, index) {
    kinds[column] = table.rows[0].cells[index].kind;
  });
  assert.deepStrictEqual(kinds, {
    count: 'number',
    sorted: 'true',
    missed: 'false',
    reason: 'text',
    detail: 'empty',
  });
});

test('the entity name is classified as a name, not as plain text', function () {
  const table = load().of([{ __entity__: 'cube', __type__: 'ShapeUnderTest' }]);
  assert.strictEqual(table.rows[0].cells[0].kind, 'name');
});

test('an answer of bare values is given a single value column', function () {
  const table = load().of([{ value: 'cube' }, { value: 'cylinder_1' }]);
  assert.deepStrictEqual(table.columns, ['value']);
  assert.deepStrictEqual(cellTexts(table.rows[1]), ['cylinder_1']);
});

test('a replay window travels beside its row and lands on it', function () {
  const table = load().of([
    { __entity__: 'cube PickUpEvent', __type__: 'SegmindEventRecord',
      timestamp: '2026-08-13 12:00:30' },
    { __entity__: 'cube InsertionEvent', __type__: 'SegmindEventRecord',
      timestamp: '2026-08-13 12:00:40' },
  ], [{ start: 100, end: 110 }, null]);
  assert.deepStrictEqual(table.columns, ['name', 'timestamp']);
  assert.deepStrictEqual(table.rows[0].replay, { start: 100, end: 110 });
  assert.strictEqual(table.rows[1].replay, null);
});

test('an answer sent without windows is a table of rows that replay nothing', function () {
  const table = load().of([{ __entity__: 'cube', __type__: 'ShapeUnderTest' }]);
  assert.strictEqual(table.rows[0].replay, null);
});

test('a key the table does not know is left out rather than shown as a column',
  function () {
    // an older viewer meeting a newer answer: whatever __key__ it carries is the
    // answer's own bookkeeping, and printing it would put [object Object] in a cell
    const table = load().of([
      { name: 'cube', __whatever__: { start: 100, end: 110 } },
    ]);
    assert.deepStrictEqual(table.columns, ['name']);
    assert.deepStrictEqual(cellTexts(table.rows[0]), ['cube']);
  });

test('a value that is not a scalar reads as its JSON rather than as [object Object]',
  function () {
    const table = load().of([{ span: { start: 100, end: 110 } }]);
    assert.deepStrictEqual(cellTexts(table.rows[0]), ['{"start":100,"end":110}']);
  });

test('no rows is an empty table rather than a broken one', function () {
  const table = load().of([]);
  assert.deepStrictEqual(table.columns, []);
  assert.deepStrictEqual(table.rows, []);
});
