const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const web = path.resolve(__dirname, '../../../cramera/src/cramera/web');
const read = name => fs.readFileSync(path.join(web, name), 'utf8');

test('the packaged viewer exposes only its scene page', () => {
  assert.deepEqual(fs.readdirSync(web).filter(name => name.endsWith('.html')), ['index.html']);
  for (const href of read('index.html').matchAll(/href="([^"]+\.html)"/g)) {
    assert.equal(href[1], 'index.html');
  }
});

test('scene and graph inspection never issue robot editing requests', () => {
  for (const name of ['panels/robot_scene/panel.js', 'panels/graph/panel.js']) {
    const source = read(name);
    for (const route of ['/move', '/joint', '/constraint', '/teleop', '/api/plan/']) {
      assert.equal(source.includes("'" + route), false, `${name} calls ${route}`);
    }
  }
});

test('the scene uses procedural lighting and backgrounds without bundled imagery', () => {
  assert.equal(fs.existsSync(path.join(web, 'env')), false);
  assert.equal(fs.existsSync(path.join(web, 'img')), false);
  assert.match(read('panels/robot_scene/panel.js'), /RoomEnvironment/);
  assert.doesNotMatch(read('index.html') + read('app.css'), /(?:img|env)\//);
});

test('the viewer does not distribute deferred workbench or tracking libraries', () => {
  assert.equal(fs.existsSync(path.join(web, 'vendor/mediapipe')), false);
  assert.equal(fs.existsSync(path.join(web, 'vendor/plotly.min.js')), false);
});
