'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const test = require('node:test');
const ScenePanelFunctions = require('./scene_panel_functions');

// %% the panel's marker builders in their shared function scope
class MarkerScene extends ScenePanelFunctions {
  constructor() {
    const webDirectory = path.join(__dirname, '../../../cramera/src/cramera/web');
    const THREE = require(path.join(webDirectory, 'vendor/three.min.js'));
    super({
      THREE, window: {}, SCENE: null, playbackSpeedMultiplier: 1, statusEl: null,
      linkToPart: {}, models: [], manager: {}, needsRender: false, PLACE0: null,
      marker: new THREE.Group(), markerRoot: new THREE.Group(), worldRoot: new THREE.Group(),
      lastMarkerPayload: null, hiddenMarkerNs: {}, setTimeout() {},
    }, ['marker-specs.js', 'marker-settings.js']);
  }

  load(placeTarget) {
    this.scope.loadScene({models: [], objects: [], placeTarget});
  }

  showDebugMarker(marker) {
    this.scope.lastMarkerPayload = {markers: [marker]};
    this.scope.rebuildMarkers();
  }
}

// %% place targets and ROS overlays coexist
test('loading a place target draws its brackets while ROS markers use their own geometry', () => {
  const scene = new MarkerScene();
  const target = {position: [1, 2], z: 0.75};
  scene.load(target);

  assert.equal(scene.scope.marker.parent, scene.scope.worldRoot);
  assert.deepEqual(scene.scope.marker.position.toArray(), [...target.position, target.z]);
  assert.equal(scene.scope.marker.children.length, 9);

  const marker = {kind: 'sphere', ns: 'debug', scale: [0.2, 0.3, 0.4], pose: [3, 4, 5, 0, 0, 0, 1]};
  scene.showDebugMarker(marker);

  assert.equal(scene.scope.markerRoot.children.length, 1);
  const debugMarker = scene.scope.markerRoot.children[0];
  assert.deepEqual(debugMarker.position.toArray(), marker.pose.slice(0, 3));
  assert.equal(debugMarker.children[0].geometry.type, 'SphereGeometry');
  assert.deepEqual(debugMarker.children[0].scale.toArray(), marker.scale);
  assert.equal(scene.scope.marker.children.length, 9);
});

test('a scene without a place target hides only the target marker', () => {
  const scene = new MarkerScene();
  scene.load(null);
  scene.showDebugMarker({kind: 'cube', ns: 'debug'});

  assert.equal(scene.scope.marker.visible, false);
  assert.equal(scene.scope.markerRoot.visible, true);
  assert.equal(scene.scope.markerRoot.children.length, 1);
});
