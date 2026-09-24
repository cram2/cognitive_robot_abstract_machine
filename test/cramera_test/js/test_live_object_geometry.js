'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const ScenePanelFunctions = require('./scene_panel_functions');

// %% geometry catalog compatibility
async function spawnCatalogObject(object) {
  const spawned = [];
  const panel = new ScenePanelFunctions({
    liveOn: true, liveSyncing: false, objectMeshes: {}, liveSpawned: {},
    liveStateKeys: {}, needsRender: false,
    fetch: async () => ({json: async () => ({objects: [object]})}),
  });
  panel.scope.liveUrl = () => 'http://localhost:8123';
  panel.scope.addObject = spec => spawned.push(spec);
  panel.scope.syncLiveObjects();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(spawned.length, 1);
  return spawned[0];
}

test('a shape catalog object renders without a redundant object kind', async () => {
  const shapes = [{kind: 'sphere', radius: 0.2}];
  const spec = await spawnCatalogObject({key: 'ball', id: 'ball', shapes});
  assert.equal(spec.shapes, shapes);
  assert.equal(spec.liveBase, 'http://localhost:8123');
});

test('legacy shape catalog objects remain renderable', async () => {
  const shapes = [{kind: 'sphere', radius: 0.2}];
  const spec = await spawnCatalogObject({key: 'ball', id: 'ball', kind: 'shapes', shapes});
  assert.equal(spec.shapes, shapes);
});

test('legacy mesh catalog objects retain their served URL', async () => {
  const spec = await spawnCatalogObject({
    key: 'cup', id: 'cup', kind: 'mesh', mesh: '/mesh?key=cup', format: 'obj',
  });
  assert.equal(spec.meshUrl, 'http://localhost:8123/mesh?key=cup');
  assert.equal(spec.format, 'obj');
});

test('legacy box catalog objects retain their dimensions', async () => {
  const size = [0.2, 0.3, 0.4];
  const spec = await spawnCatalogObject({key: 'box', id: 'box', kind: 'box', size});
  assert.equal(spec.box, size);
});
