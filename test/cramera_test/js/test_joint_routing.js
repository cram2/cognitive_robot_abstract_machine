// Unit tests for web/core/joint-routing.js (node:test): finding the joint a recorded
// connection name drives.
//
// A recording keys each frame by the world's own connection name, while a bundled model's
// joints carry the names its URDF declares and the model carries whatever prefix the
// bundler derived for its *bodies*. Those three namings do not have to agree: a
// world-merged MJCF robot yields bodies named "montessori/link0" -- so the model's prefix
// is "montessori" -- while its connections stay bare ("joint1") or start with a slash
// ("/finger_joint1"). Splitting the key on its first slash and trusting the left half to
// be a model prefix therefore finds nothing, and the arm stands still.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  const scope = {};
  new Function('window', fs.readFileSync(path.join(WEB, 'core/joint-routing.js'), 'utf8'))(scope);
  return scope.JointRouting;
}

// a loaded model as the shell holds it: a prefix plus the joints its URDF declared
function model(prefix, jointNames) {
  const joints = {};
  jointNames.forEach(function (name) { joints[name] = { name: name }; });
  return { prefix: prefix, obj: { joints: joints } };
}

// what onboarding the Franka Montessori demo produces: the robot model prefixed after its
// bodies, plus the synthesized environment model that carries no prefix and no joints
const FRANKA = [
  model('montessori', ['joint1', 'joint7', '/finger_joint1']),
  model('', []),
];

// %% the recorded keys of a world-merged robot
test('a bare connection name finds its joint whatever prefix the model carries', function () {
  const routing = load();
  assert.strictEqual(routing.jointFor(FRANKA, 'joint1').name, 'joint1');
  assert.strictEqual(routing.jointFor(FRANKA, 'joint7').name, 'joint7');
});

test('a prefixless model with no joints does not swallow the robot keys', function () {
  /*
   * The environment model's prefix is "", which is also what splitting a bare key yields,
   * so a prefix-first lookup lands on the model that has no joints at all.
   */
  const routing = load();
  assert.notStrictEqual(routing.jointFor(FRANKA, 'joint1'), null);
});

test('a connection name that starts with a slash is a name, not a prefix', function () {
  const routing = load();
  assert.strictEqual(routing.jointFor(FRANKA, '/finger_joint1').name, '/finger_joint1');
});

// %% keys that do carry their model's prefix
test('a prefixed key resolves through the model that owns the prefix', function () {
  const routing = load();
  const models = [model('pr2', ['torso_lift_joint'])];
  assert.strictEqual(
    routing.jointFor(models, 'pr2/torso_lift_joint').name, 'torso_lift_joint');
});

test('a prefix keeps two models that declare the same joint name apart', function () {
  const routing = load();
  const left = model('left_arm', ['joint1']);
  const right = model('right_arm', ['joint1']);
  assert.strictEqual(routing.jointFor([left, right], 'right_arm/joint1'),
    right.obj.joints.joint1);
  assert.strictEqual(routing.jointFor([left, right], 'left_arm/joint1'),
    left.obj.joints.joint1);
});

test('a key whose prefix no model claims still finds the joint by its bare name', function () {
  const routing = load();
  const models = [model('montessori', ['joint1'])];
  assert.strictEqual(routing.jointFor(models, 'somewhere_else/joint1').name, 'joint1');
});

// %% nothing to drive
test('an unknown key resolves to nothing rather than throwing', function () {
  const routing = load();
  assert.strictEqual(routing.jointFor(FRANKA, 'no_such_joint'), null);
  assert.strictEqual(routing.jointFor([], 'joint1'), null);
});
