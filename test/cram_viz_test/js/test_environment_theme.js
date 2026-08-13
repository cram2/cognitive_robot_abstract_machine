// Unit tests for web/core/environment-theme.js (node:test).
// Bundled environment URDFs carry whatever grey the authoring tool exported;
// EnvironmentTheme.lookOf() maps a link's name to a furniture color/finish so the
// viewer renders sofas, shelves, and similar fixtures distinctly instead of uniform
// grey. Exercised here as pure string -> descriptor logic, without THREE or a DOM.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cram_viz', 'src', 'cram_viz', 'web');

function load() {
  global.window = {};
  new Function(fs.readFileSync(path.join(WEB, 'core/environment-theme.js'), 'utf8'))();
}

test('an unmatched link name has no look', function () {
  load();
  assert.strictEqual(window.EnvironmentTheme.lookOf('world'), null);
});

test('a sofa link is themed with the sofa fabric color', function () {
  load();
  assert.deepStrictEqual(window.EnvironmentTheme.lookOf('sofa_0'), {
    color: 0xa85c48, roughness: 0.88, metalness: 0.0, texture: null,
  });
});

test('a bookshelf link is themed brown, with no texture', function () {
  load();
  assert.deepStrictEqual(window.EnvironmentTheme.lookOf('bookshelf_0'), {
    color: 0x6b4226, roughness: 0.65, metalness: 0.0, texture: null,
  });
});

test('a dining table link is themed with the table wood texture', function () {
  load();
  assert.deepStrictEqual(window.EnvironmentTheme.lookOf('dining_table_0'), {
    color: 0xffffff, roughness: 0.5, metalness: 0.02, texture: 'table',
  });
});

test('lookup is case-insensitive', function () {
  load();
  assert.deepStrictEqual(
    window.EnvironmentTheme.lookOf('SOFA_0'),
    window.EnvironmentTheme.lookOf('sofa_0')
  );
});

test('book links cycle through the varied palette by trailing index', function () {
  load();
  const palette = window.EnvironmentTheme.VARIED_PALETTE;
  for (let i = 0; i < palette.length; i++) {
    assert.strictEqual(window.EnvironmentTheme.lookOf('book_' + i).color, palette[i]);
  }
});

test('a book link past the palette length wraps around', function () {
  load();
  const palette = window.EnvironmentTheme.VARIED_PALETTE;
  assert.strictEqual(
    window.EnvironmentTheme.lookOf('book_' + palette.length).color,
    palette[0]
  );
});

test('a bookshelf link is not mistaken for a numbered book', function () {
  load();
  assert.strictEqual(window.EnvironmentTheme.lookOf('bookshelf_0').color, 0x6b4226);
});
