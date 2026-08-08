// Unit tests for web/core/bus.js and web/core/registry.js (node:test).
// The panel architecture's contract: panels only talk via the bus, config.js
// decides what mounts, and a broken/missing panel never takes the page down.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cram_viz', 'src', 'cram_viz', 'web');

// %% a minimal DOM the registry can mount panels into
function freshDom() {
  const slots = {};
  function makeEl(tag) {
    return {
      tagName: tag,
      children: [],
      dataset: {},
      className: '',
      innerHTML: '',
      appendChild(child) { this.children.push(child); },
    };
  }
  global.document = {
    createElement: makeEl,
    querySelector(selector) {
      const match = /\[data-slot="(.+)"\]/.exec(selector);
      if (match) {
        if (!slots[match[1]]) slots[match[1]] = makeEl('div');
        return slots[match[1]];
      }
      return null;
    },
  };
  global.window = {};
  return slots;
}

function load(file) {
  new Function(fs.readFileSync(path.join(WEB, file), 'utf8'))();
}

test('bus delivers to subscribers and off() unsubscribes', function () {
  freshDom();
  load('core/bus.js');
  const got = [];
  const cb = window.Bus.on('x', function (p) { got.push(p); });
  window.Bus.emit('x', 1);
  window.Bus.off('x', cb);
  window.Bus.emit('x', 2);
  assert.deepStrictEqual(got, [1]);
});

test('a throwing listener does not break the others', function () {
  freshDom();
  load('core/bus.js');
  const got = [];
  window.Bus.on('x', function () { throw new Error('boom'); });
  window.Bus.on('x', function (p) { got.push(p); });
  const err = console.error; console.error = function () {};
  window.Bus.emit('x', 'ok');
  console.error = err;
  assert.deepStrictEqual(got, ['ok']);
});

test('events without listeners are a no-op (panels are removable)', function () {
  freshDom();
  load('core/bus.js');
  assert.doesNotThrow(function () { window.Bus.emit('nobody:listens', {}); });
});

test('boot mounts configured panels into their slots', function () {
  const slots = freshDom();
  load('core/bus.js');
  load('core/registry.js');
  const mountedRoots = [];
  window.Panels.define('a', function (root, bus) {
    mountedRoots.push(root);
    assert.strictEqual(bus, window.Bus);
  });
  window.Panels.define('b', function () {});
  window.CRAM_VIZ_CONFIG = { layout: { left: ['a'], right: ['b', 'a'] } };
  window.Panels.boot();
  assert.deepStrictEqual(window.Panels.mounted(), ['a', 'b', 'a']);
  assert.strictEqual(slots.left.children.length, 1);
  assert.strictEqual(slots.right.children.length, 2);
  assert.strictEqual(mountedRoots[0].dataset.panel, 'a');
  assert.ok(mountedRoots[0].className.indexOf('panel-a') >= 0);
});

test('an unknown configured panel is reported, not fatal', function () {
  freshDom();
  load('core/bus.js');
  load('core/registry.js');
  window.Panels.define('real', function () {});
  window.CRAM_VIZ_CONFIG = { layout: { left: ['ghost', 'real'] } };
  const errors = [];
  const err = console.error; console.error = function (m) { errors.push(String(m)); };
  window.Panels.boot();
  console.error = err;
  assert.deepStrictEqual(window.Panels.mounted(), ['real']);
  assert.ok(errors.some(function (m) { return m.indexOf('ghost') >= 0; }));
});

test('unmounting tears every panel down so its timers stop', function () {
  freshDom();
  load('core/bus.js');
  load('core/registry.js');
  const destroyed = [];
  window.Panels.define('a', function () {
    return { destroy: function () { destroyed.push('a'); } };
  });
  window.Panels.define('b', function () { /* no cleanup needed */ });
  window.CRAM_VIZ_CONFIG = { layout: { left: ['a', 'b'] } };
  window.Panels.boot();
  window.Panels.unmountAll();
  assert.deepStrictEqual(destroyed, ['a']);
  assert.deepStrictEqual(window.Panels.mounted(), []);
});

test('a panel whose destroy throws does not block the others', function () {
  freshDom();
  load('core/bus.js');
  load('core/registry.js');
  const destroyed = [];
  window.Panels.define('broken', function () {
    return { destroy: function () { throw new Error('nope'); } };
  });
  window.Panels.define('fine', function () {
    return { destroy: function () { destroyed.push('fine'); } };
  });
  window.CRAM_VIZ_CONFIG = { layout: { left: ['broken', 'fine'] } };
  window.Panels.boot();
  const err = console.error; console.error = function () {};
  window.Panels.unmountAll();
  console.error = err;
  assert.deepStrictEqual(destroyed, ['fine']);
});

test('a panel that throws while mounting shows an error, others still mount', function () {
  freshDom();
  load('core/bus.js');
  load('core/registry.js');
  window.Panels.define('broken', function () { throw new Error('nope'); });
  window.Panels.define('fine', function () {});
  window.CRAM_VIZ_CONFIG = { layout: { left: ['broken', 'fine'] } };
  const err = console.error; console.error = function () {};
  window.Panels.boot();
  console.error = err;
  assert.ok(window.Panels.mounted().indexOf('fine') >= 0);
});
