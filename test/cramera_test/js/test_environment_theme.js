// Unit tests for web/core/environment-theme.js (node:test).
// Bundled environment URDFs carry whatever grey the authoring tool exported, so the
// panel re-skins them by link name. The lookup is a first-match-wins rule table, which
// makes rule order load-bearing: a link matching two rules must get the specific one.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  global.window = {};
  new Function(fs.readFileSync(path.join(WEB, 'core/environment-theme.js'), 'utf8'))();
  return global.window.EnvironmentTheme;
}

test('lookOf() returns null for a link the vocabulary does not know', function () {
  assert.strictEqual(load().lookOf('some_unmapped_link'), null);
});

test('lookOf() is case-insensitive', function () {
  const theme = load();
  assert.deepStrictEqual(theme.lookOf('Kitchen_WALL_3'), theme.lookOf('kitchen_wall_3'));
});

test('lookOf() tolerates a missing link name', function () {
  const theme = load();
  assert.strictEqual(theme.lookOf(''), null);
  assert.strictEqual(theme.lookOf(undefined), null);
});

test('a countertop asks for the counter texture, a table for the table one', function () {
  const theme = load();
  assert.strictEqual(theme.lookOf('island_countertop').texture, 'counter');
  assert.strictEqual(theme.lookOf('coffee_table').texture, 'table');
});

test('a cooktop wins over the countertop it sits in', function () {
  // "cooktop" also contains no countertop vocabulary, but the two rules sit next to
  // each other and the dark cooktop must not be re-skinned as a white worktop
  const theme = load();
  assert.notStrictEqual(theme.lookOf('kitchen_cooktop').color, theme.lookOf('kitchen_countertop').color);
  assert.strictEqual(theme.lookOf('kitchen_cooktop').texture, null);
});

test('a run of shelved books cycles through the varied palette by index', function () {
  const theme = load();
  const first = theme.lookOf('book_0').color;
  const second = theme.lookOf('book_1').color;
  assert.notStrictEqual(first, second);
  assert.strictEqual(theme.lookOf('book_' + theme.VARIED_PALETTE.length).color, first);
});

test('a book is only varied when its name is exactly book_<n>', function () {
  const theme = load();
  assert.strictEqual(theme.lookOf('bookshelf_2').texture, null);
  assert.notStrictEqual(theme.lookOf('bookshelf_2').color, theme.lookOf('book_2').color);
});

test('every rule yields a complete look descriptor', function () {
  const theme = load();
  ['kitchen_wall', 'sofa_1', 'bed_frame', 'trash_can', 'radiator_2', 'floor_lamp'].forEach(
    function (link) {
      const look = theme.lookOf(link);
      assert.ok(look, link);
      assert.strictEqual(typeof look.color, 'number', link);
      assert.strictEqual(typeof look.roughness, 'number', link);
      assert.strictEqual(typeof look.metalness, 'number', link);
      assert.ok('texture' in look, link);
    }
  );
});
