// Unit tests for core/frame-axes.js (node:test): which frames the in-scene TF display
// draws a triad on, and the settings it remembers between visits.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function loadFrameAxes() {
  const scope = {};
  new Function('window', fs.readFileSync(path.join(WEB, 'core/frame-axes.js'), 'utf8'))(scope);
  return scope.FrameAxes;
}

// %% stubs of the three.js objects the display reads
function makeLink(name) {
  return { isURDFLink: true, name: name, children: [] };
}

// a loaded model as panels/robot_scene/panel.js keeps it, with three.js' traverse
function makeModel(prefix, linkNames, name) {
  const links = linkNames.map(makeLink);
  return {
    name: name || prefix || 'model',
    prefix: prefix,
    obj: {
      isURDFLink: false,
      traverse(visit) { visit(this); links.forEach(visit); },
    },
    links: links,
  };
}

function makeStorage(initial) {
  const items = Object.assign({}, initial);
  return {
    items: items,
    getItem(key) { return key in items ? items[key] : null; },
    setItem(key, value) { items[key] = String(value); },
  };
}

// %% which frames get a triad
test('every link of every loaded model is a frame', function () {
  const FrameAxes = loadFrameAxes();
  const model = makeModel('', ['base_link', 'gripper_link']);

  const frames = FrameAxes.framesOf([model], {});

  assert.deepStrictEqual(frames.map(function (f) { return f.name; }),
    ['base_link', 'gripper_link']);
});

test('a frame carries the object its triad is parented to', function () {
  const FrameAxes = loadFrameAxes();
  const model = makeModel('', ['base_link']);

  const frames = FrameAxes.framesOf([model], {});

  assert.strictEqual(frames[0].object, model.links[0]);
});

test('a model prefix keeps same-named links of two models apart', function () {
  const FrameAxes = loadFrameAxes();
  const frames = FrameAxes.framesOf(
    [makeModel('pr2', ['base_link']), makeModel('tracy', ['base_link'])], {});

  assert.deepStrictEqual(frames.map(function (f) { return f.name; }),
    ['pr2/base_link', 'tracy/base_link']);
});

test('a loose object is a frame of its own', function () {
  const FrameAxes = loadFrameAxes();
  const milk = { isGroup: true };

  const frames = FrameAxes.framesOf([], { 'milk.stl': milk });

  assert.deepStrictEqual(frames, [{
    id: FrameAxes.frameId(FrameAxes.OBJECT_SOURCE, 'milk.stl'),
    name: 'milk.stl',
    source: FrameAxes.OBJECT_SOURCE,
    object: milk,
  }]);
});

// two URDFs routinely each carry a root link of the same name, and one model's frame
// must not stand in for the other's
test('same-named links of two models are two frames', function () {
  const FrameAxes = loadFrameAxes();
  const robot = makeModel('', ['world_root'], 'pr2');
  const environment = makeModel('', ['world_root'], 'apartment');

  const frames = FrameAxes.framesOf([robot, environment], {});

  assert.notStrictEqual(frames[0].id, frames[1].id);
  assert.deepStrictEqual(frames.map(function (f) { return f.object; }),
    [environment.links[0], robot.links[0]]);
});

test('hiding one model\'s frame leaves the same-named one of the other', function () {
  const FrameAxes = loadFrameAxes();
  const robot = makeModel('', ['world_root'], 'pr2');
  const environment = makeModel('', ['world_root'], 'apartment');
  const frames = FrameAxes.framesOf([robot, environment], {});

  const shown = FrameAxes.visibleFrames(frames,
    { frames: { [FrameAxes.frameId('pr2', 'world_root')]: true } });

  assert.deepStrictEqual(shown.map(function (f) { return f.source; }), ['apartment']);
});

test('anything in the model tree that is not a link is not a frame', function () {
  const FrameAxes = loadFrameAxes();
  const model = makeModel('', ['base_link']);
  model.obj.traverse = function (visit) {
    visit(this);
    visit({ isURDFLink: false, name: 'base_link_visual' });   // the link's mesh
    visit(model.links[0]);
  };

  const frames = FrameAxes.framesOf([model], {});

  assert.deepStrictEqual(frames.map(function (f) { return f.name; }), ['base_link']);
});

test('frames come out in a stable order whatever the scene loaded first', function () {
  const FrameAxes = loadFrameAxes();
  const model = makeModel('', ['torso', 'base_link']);

  const frames = FrameAxes.framesOf([model], { 'milk.stl': {} });

  assert.deepStrictEqual(frames.map(function (f) { return f.name; }),
    ['base_link', 'milk.stl', 'torso']);
});

// %% filtering by source, and by single frame below it
function scene(FrameAxes) {
  const robot = makeModel('pr2', ['base_link', 'gripper_link']);
  const environment = makeModel('', ['sink'], 'environment');
  const frames = FrameAxes.framesOf([robot, environment], { 'milk.stl': {} });
  return { robot: robot, environment: environment, frames: frames };
}

const names = function (frames) {
  return frames.map(function (frame) { return frame.name; });
};

test('a source is every loaded model, plus the loose objects when there are any', function () {
  const FrameAxes = loadFrameAxes();
  const models = [makeModel('pr2', ['base_link']), makeModel('', ['sink'], 'environment')];

  assert.deepStrictEqual(FrameAxes.sourcesOf(models, { 'milk.stl': {} }),
    ['environment', FrameAxes.OBJECT_SOURCE, 'pr2']);
});

test('a scene with no loose objects offers no object source', function () {
  const FrameAxes = loadFrameAxes();
  const model = makeModel('', ['base_link'], 'pr2');

  assert.deepStrictEqual(FrameAxes.sourcesOf([model], {}), ['pr2']);
});

test('a frame knows which source it came from', function () {
  const FrameAxes = loadFrameAxes();
  const model = makeModel('', ['base_link'], 'pr2');

  const frames = FrameAxes.framesOf([model], { 'milk.stl': {} });

  assert.deepStrictEqual(frames.map(function (f) { return f.source; }),
    ['pr2', FrameAxes.OBJECT_SOURCE]);
});

test('a source lists the frames under it', function () {
  const FrameAxes = loadFrameAxes();
  const built = scene(FrameAxes);

  assert.deepStrictEqual(names(FrameAxes.framesOfSource(built.frames, 'pr2')),
    ['pr2/base_link', 'pr2/gripper_link']);
});

test('hiding a source drops its frames and leaves the others', function () {
  const FrameAxes = loadFrameAxes();
  const built = scene(FrameAxes);

  const shown = FrameAxes.visibleFrames(built.frames, { sources: { pr2: true } });

  assert.deepStrictEqual(names(shown), ['milk.stl', 'sink']);
});

test('hiding one frame leaves the rest of its source on screen', function () {
  const FrameAxes = loadFrameAxes();
  const built = scene(FrameAxes);

  const shown = FrameAxes.visibleFrames(built.frames,
    { frames: { [FrameAxes.frameId('pr2', 'pr2/base_link')]: true } });

  assert.deepStrictEqual(names(shown), ['milk.stl', 'pr2/gripper_link', 'sink']);
});

test('a fully shown source reads as all of it', function () {
  const FrameAxes = loadFrameAxes();
  const built = scene(FrameAxes);

  assert.strictEqual(FrameAxes.sourceState(built.frames, 'pr2', {}),
    FrameAxes.SourceState.ALL);
});

test('a source with one frame hidden reads as partly shown', function () {
  const FrameAxes = loadFrameAxes();
  const built = scene(FrameAxes);

  const state = FrameAxes.sourceState(built.frames, 'pr2',
    { frames: { [FrameAxes.frameId('pr2', 'pr2/base_link')]: true } });

  assert.strictEqual(state, FrameAxes.SourceState.SOME);
});

test('a source with every frame hidden one by one reads as none of it', function () {
  const FrameAxes = loadFrameAxes();
  const built = scene(FrameAxes);

  const state = FrameAxes.sourceState(built.frames, 'pr2', {
    frames: {
      [FrameAxes.frameId('pr2', 'pr2/base_link')]: true,
      [FrameAxes.frameId('pr2', 'pr2/gripper_link')]: true,
    },
  });

  assert.strictEqual(state, FrameAxes.SourceState.NONE);
});

test('a hidden source and hidden frames survive a reload', function () {
  const FrameAxes = loadFrameAxes();
  const storage = makeStorage();

  const base = FrameAxes.frameId('pr2', 'pr2/base_link');
  FrameAxes.setSourceHidden(storage, 'environment', true, [FrameAxes.frameId('environment', 'sink')]);
  const stored = FrameAxes.setFrameHidden(storage, base, true, 'pr2');

  assert.deepStrictEqual(stored,
    { sources: { environment: true }, frames: { [base]: true } });
  assert.deepStrictEqual(FrameAxes.hidden(storage), stored);
});

test('ticking a source back on clears the frames picked off inside it', function () {
  const FrameAxes = loadFrameAxes();
  const storage = makeStorage();
  const ids = ['pr2/base_link', 'pr2/gripper_link']
    .map(function (name) { return FrameAxes.frameId('pr2', name); });
  FrameAxes.setFrameHidden(storage, ids[0], true, 'pr2');
  FrameAxes.setSourceHidden(storage, 'pr2', true, ids);

  const stored = FrameAxes.setSourceHidden(storage, 'pr2', false, ids);

  assert.deepStrictEqual(stored, { sources: {}, frames: {} });
});

test('ticking a frame back on shows its source again, so the tick is not ignored', function () {
  const FrameAxes = loadFrameAxes();
  const storage = makeStorage();
  FrameAxes.setSourceHidden(storage, 'pr2', true, []);

  const stored = FrameAxes.setFrameHidden(
    storage, FrameAxes.frameId('pr2', 'pr2/base_link'), false, 'pr2');

  assert.deepStrictEqual(stored, { sources: {}, frames: {} });
});

// %% the remembered display settings
test('a viewer that has never touched the display starts with it off', function () {
  const FrameAxes = loadFrameAxes();

  assert.deepStrictEqual(FrameAxes.settings(makeStorage()), {
    visible: false, names: false, size: FrameAxes.DEFAULT_SIZE,
  });
});

test('the display state survives a reload', function () {
  const FrameAxes = loadFrameAxes();
  const storage = makeStorage();

  FrameAxes.setVisible(storage, true);
  FrameAxes.setNames(storage, true);
  const stored = FrameAxes.setSize(storage, 0.4);

  assert.deepStrictEqual(stored, { visible: true, names: true, size: 0.4 });
  assert.deepStrictEqual(FrameAxes.settings(storage), stored);
});

test('a size outside the range the control offers is clamped to it', function () {
  const FrameAxes = loadFrameAxes();

  assert.strictEqual(FrameAxes.clampSize(99), FrameAxes.MAX_SIZE);
  assert.strictEqual(FrameAxes.clampSize(0), FrameAxes.MIN_SIZE);
});

test('an unreadable stored size falls back to the default', function () {
  const FrameAxes = loadFrameAxes();
  const storage = makeStorage({ 'cramera.frame-axes-size': 'huge' });

  assert.strictEqual(FrameAxes.settings(storage).size, FrameAxes.DEFAULT_SIZE);
});

test('the triad arms follow the axis-to-colour convention', function () {
  const FrameAxes = loadFrameAxes();

  assert.deepStrictEqual(FrameAxes.AXES.map(function (arm) { return arm.axis; }),
    ['x', 'y', 'z']);
});
