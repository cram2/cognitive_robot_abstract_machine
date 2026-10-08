'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const test = require('node:test');
const BrowserSource = require('./browser_source');
const ScenePanelFunctions = require('./scene_panel_functions');

const WEB = path.join(__dirname, '../../../cramera/src/cramera/web');
const THREE = require(path.join(WEB, 'vendor/three.min.js'));

// %% native camera descriptions and loaded links
/** Load a fresh camera module so each case owns its camera constructors.
 * @returns {typeof window.RobotCamera} The browser module's public camera API.
 */
function loadRobotCamera() {
  const scope = {};
  new Function('window', BrowserSource.read(path.join(WEB, 'core/robot-camera.js')))(scope);
  return scope.RobotCamera;
}

/** Describe one sensor without deriving its semantics from the link name.
 * @param {string} link Exact model link to bind.
 * @param {Partial<CameraDefinition>} [values] Native fields overridden by the case.
 * @returns {CameraDefinition} Camera metadata consumed by the browser module.
 */
function cameraDefinition(link, values = {}) {
  return {name: 'Inspection camera', link, forward: [1, 0, 0],
    horizontalAngle: Math.PI / 2, verticalAngle: Math.PI / 3, default: false, ...values};
}

/** Build independently articulated links for a scene model.
 * @param {string} name Model identity used in source selection.
 * @param {CameraDefinition[]} cameras Sensors attached to the model.
 * @param {boolean} [robot] Whether the model represents a robot.
 * @returns {CameraModel} Loaded model with camera roots.
 */
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

test('multiple native defaults retain the robot sensor order instead of label order', () => {
  const RobotCamera = loadRobotCamera();
  const robot = makeModel('robot', [
    cameraDefinition('first', {name: 'Zulu', default: true}),
    cameraDefinition('second', {name: 'Alpha', default: true}),
  ]);
  assert.deepEqual(RobotCamera.sources([robot]).map(source => source.definition), robot.cameras);
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

test('malformed camera descriptors do not prevent valid sources from loading', () => {
  const RobotCamera = loadRobotCamera();
  const robot = makeModel('robot', [cameraDefinition('eye')]);
  const valid = robot.cameras[0];
  robot.cameras.unshift(null, {}, {...valid, name: null}, {...valid, link: 42});
  assert.deepEqual(RobotCamera.sources([robot]).map(source => source.definition), [valid]);
});

test('malformed camera collections leave the scene without camera sources', () => {
  const RobotCamera = loadRobotCamera();
  const robot = makeModel('robot', [cameraDefinition('eye')]);
  for (const cameras of [{}, 'eye']) {
    robot.cameras = cameras;
    assert.deepEqual(RobotCamera.sources([robot]), []);
  }
});

// %% articulation in the viewer's rotated world
/** A placed robot under the viewer's Z-up to Y-up world transform. */
class ArticulatedViewpoint {
  /** @param {Partial<CameraDefinition>} [definition] Sensor fields overridden by the case. */
  constructor(definition = {}) {
    /** Root supplying the same axis conversion as the rendered world. @type {THREE.Group} */
    this.world = new THREE.Group();
    this.world.rotation.x = -Math.PI / 2;
    /** Robot model whose independent placement can change. @type {CameraModel} */
    this.robot = makeModel('robot', [cameraDefinition('eye', definition)]);
    this.robot.obj.position.set(3, -2, 0.5);
    /** Articulated camera root whose position and rotation can change. @type {THREE.Object3D} */
    this.head = this.robot.obj.links.eye;
    this.head.position.set(0.2, 0, 1.4);
    this.world.add(this.robot.obj);
    /** Published sensor bound to the fixture's camera root. @type {ReturnType<typeof window.RobotCamera.sources>[number]} */
    this.source = loadRobotCamera().sources([this.robot])[0];
  }
}

/** Compare spatial coordinates within floating-point transform precision.
 * @param {THREE.Vector3} actual Position or direction produced by the camera.
 * @param {number[]} expected Analytically determined XYZ coordinates.
 * @returns {void}
 */
function assertVector(actual, expected) {
  const tolerance = 1e-10;
  assert.ok(actual.distanceTo(new THREE.Vector3(...expected)) < tolerance,
    'expected ' + expected + ', received ' + actual.toArray());
}

/** Recover the camera image's upward direction in world coordinates.
 * @param {THREE.PerspectiveCamera} camera Rendered camera orientation.
 * @returns {THREE.Vector3} Image-up direction independent of the requested up vector.
 */
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
  const notifications = [];
  const panel = new ScenePanelFunctions({SCENE: null, playbackSpeedMultiplier: 1,
    statusEl: null, linkToPart: {}, sceneBase: 'scenes/native/', models: [], robotModel: null,
    robotCameraPanel: {setModels(models, primary) { notifications.push({models: [...models], primary}); }},
    worldRoot: new THREE.Group(), manager: {}, needsRender: false, setTimeout() {}});
  panel.scope.makeUrdfLoader = () => ({load(address, onLoad) { onLoad(object); }});
  panel.scope.refreshFrameAxes = () => {};
  panel.scope.buildPlaceTargetMarker = () => {};
  const cameras = [cameraDefinition('robot/eye')];
  panel.scope.loadScene({name: 'Native robot', robot: {cameras},
    models: [{name: 'robot', robot: true, urdf: 'robot.urdf'}], objects: []});
  assert.equal(panel.scope.robotModel.cameras, cameras);
  assert.equal(notifications.length, 1);
  assert.equal(notifications[0].models[0], panel.scope.robotModel);
  assert.equal(notifications[0].primary, panel.scope.robotModel);
});

test('a scene without camera metadata still loads with no invented viewpoints', () => {
  const object = new THREE.Group();
  const panel = new ScenePanelFunctions({SCENE: null, playbackSpeedMultiplier: 1,
    statusEl: null, linkToPart: {}, sceneBase: 'scenes/old/', models: [], robotModel: null,
    robotCameraPanel: {setModels() {}},
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
  const panel = new ScenePanelFunctions({running: true, requestAnimationFrame() {},
    clock: {getDelta() { return 0; }}, models: [], replayClip: null,
    highlightArrows: {}, controls: {update() { return false; }, autoRotate: false},
    playing: false, needsRender: true,
    robotCameraPanel: {render() { calls.push('robot'); }}});
  panel.scope.renderFrame = () => calls.push('overview');
  panel.scope.tick();
  assert.deepEqual(calls, ['overview', 'robot']);
  assert.equal(panel.scope.needsRender, false);
});


test('closing Robot view focuses a visible control when Layers are folded or open', () => {
  for (const visible of [false, true]) {
    const focused = [];
    const checkbox = {getClientRects: () => visible ? [{}] : [], focus: () => focused.push(checkbox)};
    const foldButton = {focus: () => focused.push(foldButton)};
    const controls = new Map([['lyr-robot-view', checkbox], ['layers-fold', foldButton]]);
    const panel = new ScenePanelFunctions({$: identifier => controls.get(identifier)});
    panel.scope.restoreRobotViewFocus();
    assert.deepEqual(focused, [visible ? checkbox : foldButton]);
  }
});
