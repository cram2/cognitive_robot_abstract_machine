// Unit tests for web/core/split-resize.js (node:test): the dividers it installs.
//
// The shell stacks the EQL and graph panels in one slot, so besides the column
// divider between the scene and the knowledge column there has to be a row divider
// between the two stacked panels. split-resize.js is loaded with its free variables
// (window, document, location, localStorage, SplitSizing) bound as explicit function
// parameters; SplitSizing is the real core/split-sizing.js, so the geometry under test
// is the one that ships.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');
const SOURCE = fs.readFileSync(path.join(WEB, 'core/split-resize.js'), 'utf8');

function loadSplitSizing() {
  const scope = {};
  new Function('window', fs.readFileSync(path.join(WEB, 'core/split-sizing.js'), 'utf8'))(scope);
  return scope.SplitSizing;
}

// %% a DOM stub with just the interfaces split-resize.js reaches for
function matches(element, selector) {
  if (selector.charAt(0) === '.') return element.classList.contains(selector.slice(1));
  const attribute = /^\[data-([\w-]+)(?:="([^"]*)")?\]$/.exec(selector);
  if (!attribute) return false;
  const value = element.dataset[attribute[1]];
  return attribute[2] === undefined ? value !== undefined : value === attribute[2];
}

function descendants(element) {
  return element.children.reduce(function (all, child) {
    return all.concat([child], descendants(child));
  }, []);
}

function makeElement(className) {
  const listeners = {};
  const element = {
    className: className || '',
    title: '',
    textContent: '',
    style: {},
    dataset: {},
    children: [],
    rect: { left: 0, top: 0, width: 1000, height: 800, right: 1000, bottom: 800 },
    classList: {
      contains: function (name) { return element.className.split(' ').indexOf(name) >= 0; },
      add: function (name) { if (!element.classList.contains(name)) element.className += ' ' + name; },
      remove: function (name) {
        element.className = element.className.split(' ').filter(function (n) { return n !== name; }).join(' ');
      },
      toggle: function (name) {
        const on = !element.classList.contains(name);
        if (on) element.classList.add(name); else element.classList.remove(name);
        return on;
      },
    },
    addEventListener: function (type, callback) { (listeners[type] = listeners[type] || []).push(callback); },
    removeEventListener: function (type, callback) {
      listeners[type] = (listeners[type] || []).filter(function (c) { return c !== callback; });
    },
    dispatch: function (type, event) {
      (listeners[type] || []).slice().forEach(function (callback) { callback(event || {}); });
    },
    setPointerCapture: function () {},
    appendChild: function (child) { element.children.push(child); return child; },
    insertBefore: function (child, reference) {
      const at = element.children.indexOf(reference);
      element.children.splice(at < 0 ? element.children.length : at, 0, child);
      return child;
    },
    getBoundingClientRect: function () { return element.rect; },
    querySelector: function (selector) {
      return descendants(element).filter(function (e) { return matches(e, selector); })[0] || null;
    },
    querySelectorAll: function (selector) {
      return descendants(element).filter(function (e) { return matches(e, selector); });
    },
  };
  return element;
}

function makePanel(id) {
  const panel = makeElement('panel panel-' + id);
  panel.dataset.panel = id;
  return panel;
}

// the shell as config.js lays it out: the scene on the left, EQL above the graph
// on the right
function install(stored) {
  const split = makeElement('split');
  const left = makeElement('slot');
  const right = makeElement('slot');
  left.dataset.slot = 'left';
  right.dataset.slot = 'right';
  const eql = makePanel('eql');
  const graph = makePanel('graph');
  eql.appendChild(makeElement('panel-head'));
  graph.appendChild(makeElement('graph-wrap'));
  right.appendChild(eql);
  right.appendChild(graph);
  split.appendChild(left);
  split.appendChild(right);

  const store = Object.assign({}, stored);
  const document = {
    querySelector: function (selector) {
      if (selector === 'main.split') return split;
      return [left, right].filter(function (e) { return matches(e, selector); })[0] || null;
    },
    createElement: function () { return makeElement(); },
    addEventListener: function () {},
  };
  const window = { dispatchEvent: function () {} };
  const localStorage = {
    getItem: function (key) { return key in store ? store[key] : null; },
    setItem: function (key, value) { store[key] = value; },
  };
  new Function('window', 'document', 'location', 'localStorage', 'SplitSizing', SOURCE)(
    window, document, { pathname: '/index.html' }, localStorage, loadSplitSizing());

  return {
    split: split, left: left, right: right, eql: eql, graph: graph, store: store,
    columnDivider: split.children.filter(function (e) { return matches(e, '.split-divider'); })[0],
    rowDivider: right.children.filter(function (e) { return matches(e, '.slot-divider'); })[0],
  };
}

function drag(divider, event) {
  divider.dispatch('pointerdown', Object.assign({ preventDefault: function () {}, pointerId: 1 }, event));
  divider.dispatch('pointermove', event);
  divider.dispatch('pointerup', event);
}

// %% the row divider between the stacked EQL and graph panels
test('a row divider is installed between the two stacked panels', function () {
  const shell = install();
  assert.ok(shell.rowDivider, 'no .slot-divider in the right slot');
  assert.deepStrictEqual(shell.right.children.map(function (e) { return e.className; }),
    ['panel panel-eql', 'pane-divider slot-divider', 'panel panel-graph']);
});

test('the slot holding the divider is laid out as rows, the graph taking the larger share', function () {
  const shell = install();
  assert.strictEqual(shell.right.style.display, 'grid');
  assert.strictEqual(shell.right.style.gridTemplateRows, 'minmax(0,40fr) auto minmax(0,60fr)');
});

test('dragging the row divider resizes the graph against the EQL panel', function () {
  const shell = install();
  drag(shell.rowDivider, { clientY: 200 });
  assert.strictEqual(shell.right.style.gridTemplateRows, 'minmax(0,25fr) auto minmax(0,75fr)');
});

test('the dragged row size is remembered per page', function () {
  const shell = install();
  drag(shell.rowDivider, { clientY: 600 });
  assert.strictEqual(shell.store['splitBottom:index.html'], '0.250');
});

test('a remembered row size is restored on load', function () {
  const shell = install({ 'splitBottom:index.html': '0.3' });
  assert.strictEqual(shell.right.style.gridTemplateRows, 'minmax(0,70fr) auto minmax(0,30fr)');
});

test('double-clicking the row divider restores the default split', function () {
  const shell = install({ 'splitBottom:index.html': '0.3' });
  shell.rowDivider.dispatch('dblclick');
  assert.strictEqual(shell.right.style.gridTemplateRows, 'minmax(0,40fr) auto minmax(0,60fr)');
  assert.strictEqual(shell.store['splitBottom:index.html'], '0.600');
});

// %% the column divider between the scene and the knowledge column
test('dragging the column divider resizes the knowledge column against the scene', function () {
  const shell = install();
  drag(shell.columnDivider, { clientX: 250 });
  assert.strictEqual(shell.split.style.gridTemplateColumns, 'minmax(0,25fr) auto minmax(0,75fr)');
  assert.strictEqual(shell.store['splitRight:index.html'], '0.750');
});

test('a remembered column size is restored on load', function () {
  const shell = install({ 'splitRight:index.html': '0.6' });
  assert.strictEqual(shell.split.style.gridTemplateColumns, 'minmax(0,40fr) auto minmax(0,60fr)');
});
