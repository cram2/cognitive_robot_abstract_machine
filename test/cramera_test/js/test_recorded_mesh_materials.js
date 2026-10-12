'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const BrowserSource = require('./browser_source');
const ScenePanelFunctions = require('./scene_panel_functions');

// %% recorded geometry through the shipped OBJ loader
const FALLBACK_COLOR = '#778899';

/** Render a real parsed OBJ through the panel, optionally removing its vertex colours. */
function renderRecordedMesh(withVertexColors) {
  const webDirectory = path.join(__dirname, '../../../cramera/src/cramera/web');
  const THREE = {...require(path.join(webDirectory, 'vendor/three.min.js'))};
  vm.runInNewContext(BrowserSource.read(path.join(webDirectory, 'vendor/OBJLoader.js')), {THREE});
  const content = new THREE.OBJLoader().parse(
    BrowserSource.read(path.join(__dirname, '../dataset/vertex_colors.obj')));
  const mesh = content.children[0];
  if (!withVertexColors) mesh.geometry.deleteAttribute('color');
  THREE.OBJLoader.prototype.load = function (url, loaded) { loaded(content); };

  const panel = new ScenePanelFunctions({
    THREE, objectMeshes: {}, objectPending: {}, objectIdByKey: {}, objectKeyById: {},
    objectLabels: {}, worldRoot: new THREE.Group(), labelsOn: false, needsRender: false,
  });
  panel.scope.makeLabel = () => new THREE.Object3D();
  panel.scope.refreshFrameAxes = () => {};
  panel.scope.addObject({id: 'painted', key: 'painted', meshUrl: 'recorded.obj', color: FALLBACK_COLOR});
  return mesh;
}

// %% materials without a companion library
test('a recorded OBJ keeps its vertex colour material without an MTL', () => {
  const mesh = renderRecordedMesh(true);
  assert.equal(mesh.material.vertexColors, true);
  assert.equal(mesh.material.color.getHexString(), 'ffffff');
  assert.deepEqual(Array.from(mesh.geometry.getAttribute('color').array), [
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
  ]);
});

test('an uncoloured recorded OBJ retains the catalog fallback tint', () => {
  const mesh = renderRecordedMesh(false);
  assert.equal(mesh.material.vertexColors, false);
  assert.equal(mesh.material.color.getHexString(), FALLBACK_COLOR.slice(1));
});
