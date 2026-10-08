'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const test = require('node:test');
const BrowserSource = require('./browser_source');
const ScenePanelFunctions = require('./scene_panel_functions');

const WEB = path.join(__dirname, '../../../cramera/src/cramera/web');
const THREE = require(path.join(WEB, 'vendor/three.min.js'));

// %% native camera descriptions and loaded links
function loadRobotCamera() {
  const scope = {};
  new Function('window', BrowserSource.read(path.join(WEB, 'core/robot-camera.js')))(scope);
  return scope.RobotCamera;
}

function cameraDefinition(link, values = {}) {
  return {name: 'Inspection camera', link, forward: [1, 0, 0],
    horizontalAngle: Math.PI / 2, verticalAngle: Math.PI / 3, default: false, ...values};
}

function makeModel(name, cameras, robot = true) {
  const object = new THREE.Group();
  object.links = {};
  for (const definition of cameras) {
    const link = new THREE.Group();
    link.name = definition.link;
    object.links[link.name] = link;
    object.add(link);
  }
  return {name, prefix: '', robot, cameras, obj: object};
}

test('only loaded robot models contribute explicitly annotated cameras', () => {
  const RobotCamera = loadRobotCamera();
  const robot = makeModel('robot', [cameraDefinition('unexpected/body')]);
  const environment = makeModel('room', [cameraDefinition('room/sensor')], false);
  const pending = {...robot, obj: null};
  const sources = RobotCamera.sources([environment, pending, robot]);
  assert.deepEqual(sources.map(source => source.link), [robot.obj.links['unexpected/body']]);
  assert.equal(sources[0].definition, robot.cameras[0]);
});

test('camera-looking names do not invent sensors for older bundles', () => {
  const RobotCamera = loadRobotCamera();
  const robot = makeModel('robot', [cameraDefinition('wide_stereo_optical_frame'), cameraDefinition('head_link')]);
  delete robot.cameras;
  assert.deepEqual(RobotCamera.sources([robot]), []);
});

test('native default camera is offered first independently of link names', () => {
  const RobotCamera = loadRobotCamera();
  const robot = makeModel('robot', [cameraDefinition('a'), cameraDefinition('z', {default: true})]);
  assert.deepEqual(RobotCamera.sources([robot]).map(source => source.link), [robot.obj.links.z, robot.obj.links.a]);
});

test('namespaced camera roots match exact exported link names', () => {
  const RobotCamera = loadRobotCamera();
  const robot = makeModel('robot', [cameraDefinition('second/eye')]);
  robot.obj.links.eye = new THREE.Group();
  const source = RobotCamera.sources([robot])[0];
  assert.equal(source.link, robot.obj.links['second/eye']);
  delete robot.obj.links['second/eye'];
  assert.deepEqual(RobotCamera.sources([robot]), []);
});

test('distinct robot models retain stable camera identities across load order', () => {
  const RobotCamera = loadRobotCamera();
  const first = makeModel('first', [cameraDefinition('eye')]);
  const second = makeModel('second', [cameraDefinition('eye')]);
  const sources = RobotCamera.sources([second, first]);
  assert.equal(new Set(sources.map(source => source.id)).size, 2);
  assert.equal(new Set(sources.map(source => source.label)).size, 2);
  assert.deepEqual(sources.map(source => source.id), RobotCamera.sources([first, second]).map(source => source.id));
});

test('unusable projection or direction metadata does not expose an unusable camera', () => {
  const RobotCamera = loadRobotCamera();
  for (const values of [{forward: [0, 0, 0]}, {forward: [0, NaN, 1]}, {forward: [1, 0]},
    {horizontalAngle: 0}, {horizontalAngle: Math.PI}, {verticalAngle: -1}, {verticalAngle: Infinity}]) {
    assert.deepEqual(RobotCamera.sources([makeModel('robot', [cameraDefinition('eye', values)])]), []);
  }
});

// %% articulation in the viewer's rotated world
class ArticulatedViewpoint {
  constructor(definition = {}) {
    this.world = new THREE.Group();
    this.world.rotation.x = -Math.PI / 2;
    this.robot = makeModel('robot', [cameraDefinition('eye', definition)]);
    this.robot.obj.position.set(3, -2, 0.5);
    this.head = this.robot.obj.links.eye;
    this.head.position.set(0.2, 0, 1.4);
    this.world.add(this.robot.obj);
    this.source = loadRobotCamera().sources([this.robot])[0];
  }
}

function assertVector(actual, expected) {
  const tolerance = 1e-10;
  assert.ok(actual.distanceTo(new THREE.Vector3(...expected)) < tolerance,
    'expected ' + expected + ', received ' + actual.toArray());
}

function cameraUp(camera) {
  return new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion);
}

test('native forward direction and field of view determine the rendered camera', () => {
  const scene = new ArticulatedViewpoint();
  const pose = new (loadRobotCamera().Pose)(THREE);
  assert.equal(pose.update(scene.source, 640, 360), true);
  assertVector(pose.camera.position, [3.2, 1.9, 2]);
  assertVector(pose.camera.getWorldDirection(new THREE.Vector3()), [1, 0, 0]);
  assertVector(cameraUp(pose.camera), [0, 1, 0]);
  assert.ok(Math.abs(THREE.MathUtils.degToRad(pose.camera.fov) - scene.source.definition.verticalAngle) < 1e-10);
  const horizontalAngle = 2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(pose.camera.fov) / 2) * pose.camera.aspect);
  assert.ok(Math.abs(horizontalAngle - scene.source.definition.horizontalAngle) < 1e-10);
});

test('a forward direction along local Z uses negative local Y as display up', () => {
  const scene = new ArticulatedViewpoint({forward: [0, 0, 1]});
  const pose = new (loadRobotCamera().Pose)(THREE);
  pose.update(scene.source, 640, 360);
  assertVector(pose.camera.getWorldDirection(new THREE.Vector3()), [0, 1, 0]);
  assertVector(cameraUp(pose.camera), [0, 0, 1]);
});

test('arbitrary native axes are normalized without name or optical-frame assumptions', () => {
  const scene = new ArticulatedViewpoint({forward: [2, 2, 0]});
  const pose = new (loadRobotCamera().Pose)(THREE);
  pose.update(scene.source, 640, 360);
  assertVector(pose.camera.getWorldDirection(new THREE.Vector3()), [Math.SQRT1_2, 0, -Math.SQRT1_2]);
});

test('camera follows robot placement and articulation without a prior scene render', () => {
  const scene = new ArticulatedViewpoint();
  const pose = new (loadRobotCamera().Pose)(THREE);
  pose.update(scene.source, 640, 360);
  scene.robot.obj.position.set(5, 1, 0.5);
  scene.head.position.z = 2;
  scene.head.rotation.z = Math.PI / 2;
  pose.update(scene.source, 640, 360);
  assertVector(pose.camera.position, [5.2, 2.5, -1]);
  assertVector(pose.camera.getWorldDirection(new THREE.Vector3()), [0, 0, -1]);
  assertVector(cameraUp(pose.camera), [0, 1, 0]);
});

test('changing camera updates projection while resizing preserves its complete image', () => {
  const scene = new ArticulatedViewpoint();
  const pose = new (loadRobotCamera().Pose)(THREE);
  pose.update(scene.source, 320, 180);
  const projection = pose.camera.projectionMatrix.clone();
  for (const [width, height] of [[500, 500], [900, 180], [180, 900]]) {
    pose.update(scene.source, width, height);
    assert.equal(pose.camera.projectionMatrix.equals(projection), true);
  }
  const other = new ArticulatedViewpoint({verticalAngle: Math.PI / 4});
  pose.update(other.source, 320, 180);
  assert.equal(pose.camera.projectionMatrix.equals(projection), false);
  assert.equal(pose.camera.fov, THREE.MathUtils.radToDeg(other.source.definition.verticalAngle));
});

test('missing source or zero-sized viewport does not change the camera', () => {
  const scene = new ArticulatedViewpoint();
  const pose = new (loadRobotCamera().Pose)(THREE);
  pose.update(scene.source, 320, 180);
  const position = pose.camera.position.clone();
  const projection = pose.camera.projectionMatrix.clone();
  assert.equal(pose.update(null, 320, 180), false);
  for (const size of [[0, 180], [320, 0], [-10, 180], [320, -10], [NaN, 180], [320, Infinity]]) {
    assert.equal(pose.update(scene.source, ...size), false);
  }
  assert.equal(pose.camera.position.equals(position), true);
  assert.equal(pose.camera.projectionMatrix.equals(projection), true);
});

// %% scene loading and render integration
test('loaded robot scene carries native camera definitions into the viewpoint model', () => {
  const object = new THREE.Group();
  const panel = new ScenePanelFunctions({SCENE: null, playbackSpeedMultiplier: 1,
    statusEl: null, linkToPart: {}, sceneBase: 'scenes/native/', models: [], robotModel: null,
    worldRoot: new THREE.Group(), manager: {}, needsRender: false, setTimeout() {}});
  panel.scope.makeUrdfLoader = () => ({load(address, onLoad) { onLoad(object); }});
  panel.scope.refreshFrameAxes = () => {};
  panel.scope.buildPlaceTargetMarker = () => {};
  const cameras = [cameraDefinition('robot/eye')];
  panel.scope.loadScene({name: 'Native robot', robot: {cameras},
    models: [{name: 'robot', robot: true, urdf: 'robot.urdf'}], objects: []});
  assert.equal(panel.scope.robotModel.cameras, cameras);
});

test('a scene without camera metadata still loads with no invented viewpoints', () => {
  const object = new THREE.Group();
  const panel = new ScenePanelFunctions({SCENE: null, playbackSpeedMultiplier: 1,
    statusEl: null, linkToPart: {}, sceneBase: 'scenes/old/', models: [], robotModel: null,
    worldRoot: new THREE.Group(), manager: {}, needsRender: false, setTimeout() {}});
  panel.scope.makeUrdfLoader = () => ({load(address, onLoad) { onLoad(object); }});
  panel.scope.refreshFrameAxes = () => {};
  panel.scope.buildPlaceTargetMarker = () => {};
  panel.scope.loadScene({name: 'Older robot', robot: {},
    models: [{name: 'robot', robot: true, urdf: 'robot.urdf'}], objects: []});
  assert.equal(panel.scope.robotModel.cameras.length, 0);
});

test('a changed scene draws the articulated robot view after the overview', () => {
  const calls = [];
  const models = [];
  const robot = {};
  const panel = new ScenePanelFunctions({running: true, requestAnimationFrame() {},
    clock: {getDelta() { return 0; }}, models, robotModel: robot, replayClip: null,
    highlightArrows: {}, controls: {update() { return false; }, autoRotate: false},
    playing: false, needsRender: true,
    robotCameraPanel: {refreshModels(actualModels, actualRobot) {
      assert.equal(actualModels, models);
      assert.equal(actualRobot, robot);
      calls.push('refresh');
    }, render() { calls.push('robot'); }}});
  panel.scope.renderFrame = () => calls.push('overview');
  panel.scope.tick();
  assert.deepEqual(calls, ['overview', 'refresh', 'robot']);
  assert.equal(panel.scope.needsRender, false);
});
