'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const BrowserSource = require('./browser_source');
const ScenePanelFunctions = require('./scene_panel_functions');

// %% real material loading with captured file and image transport
class MaterialRequests extends ScenePanelFunctions {
  constructor() {
    const webDirectory = path.join(__dirname, '../../../cramera/src/cramera/web');
    const THREE = {...require(path.join(webDirectory, 'vendor/three.min.js'))};
    const textures = [];
    const materialFile = path.join(__dirname, 'fixtures/mesh-materials.mtl');
    THREE.FileLoader = class extends THREE.FileLoader {
      load(url, onLoad) { onLoad(BrowserSource.read(materialFile)); }
    };
    THREE.TextureLoader = class extends THREE.TextureLoader {
      load(url) {
        textures.push(this.manager.resolveURL(url));
        return new THREE.Texture();
      }
    };
    vm.runInNewContext(BrowserSource.read(path.join(webDirectory, 'vendor/MTLLoader.js')), {THREE});
    THREE.OBJLoader = class extends THREE.Loader {
      setMaterials() {}
      load(url, onLoad) { onLoad(new THREE.Group()); }
    };
    super({
      THREE, URLSearchParams, objectMeshes: {}, objectPending: {}, objectIdByKey: {},
      objectKeyById: {}, objectLabels: {}, worldRoot: new THREE.Group(), labelsOn: false,
      needsRender: false,
    }, ['shape-specs.js']);
    this.scope.makeLabel = () => new THREE.Group();
    this.scope.refreshFrameAxes = () => {};
    this.textures = textures;
  }
}

// %% live material-relative texture requests
for (const materialDirectory of ['', 'materials/']) {
  test('native live textures use the declared material directory: ' + materialDirectory, () => {
    const scene = new MaterialRequests();
    const meshUrl = 'http://localhost:8123/mesh?key=' + encodeURIComponent('/native/mesh.obj');
    scene.scope.addObject({
      key: 'textured', id: 'textured', shapes: [{
        kind: 'mesh', format: 'obj', mesh: meshUrl,
        mtl: meshUrl + '&side=' + encodeURIComponent(materialDirectory + 'paint.mtl'),
      }],
    });

    assert.deepEqual(scene.textures, [
      meshUrl + '&side=' + encodeURIComponent(materialDirectory + 'textures/paint.png'),
      meshUrl + '&side=' + encodeURIComponent(materialDirectory + '../textures/shared.png'),
      '/images/paint.png',
      'https://assets.example.test/paint.png',
      'data:image/png;base64,aW1hZ2U=',
      'blob:https://viewer.example.test/texture',
    ]);
  });
}

test('recorded material textures retain their bundled directory', () => {
  const scene = new MaterialRequests();
  const directory = 'http://localhost:8000/scenes/recorded/meshes/';
  scene.scope.addObject({
    key: 'recorded', id: 'recorded', format: 'obj',
    meshUrl: directory + 'object.obj', mtlUrl: directory + 'paint.mtl',
  });

  assert.deepEqual(scene.textures.slice(0, 2), [
    directory + 'textures/paint.png', directory + '../textures/shared.png',
  ]);
});
