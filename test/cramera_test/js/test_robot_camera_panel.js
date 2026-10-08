'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const BrowserSource = require('./browser_source');
const path = require('node:path');
const WEB = path.join(__dirname, '../../../cramera/src/cramera/web');
const THREE = require(path.join(WEB, 'vendor/three.min.js'));

// %% panel controls without a browser GPU
/** @typedef {{left: number, top?: number, right?: number, bottom?: number, width: number, height: number}} ViewBounds */
/** @typedef {{stopPropagation: function(): void, preventDefault: function(): void, target: ViewElement, pointerId: number, button: number, isPrimary: boolean, clientX?: number, clientY?: number, key?: string, pointerType?: string, buttonName?: string}} ViewInput */
/** @typedef {[source: {width: number, height: number}, sourceX: number, sourceY: number, sourceWidth: number, sourceHeight: number, destinationX: number, destinationY: number, destinationWidth: number, destinationHeight: number]} ImageCopy */
/** @typedef {{drawImage: (...coordinates: ImageCopy) => void, fillRect: function(...number): void, fillStyle?: string}} ImageContext */
/** @typedef {{panel: InstanceType<typeof window.RobotCameraPanel>, root: ViewElement, layer: ViewElement, renderer: InsetRenderer, elements: Map<string, ViewElement>, header: ViewElement, container: ViewElement, changes: function(): number, resize: function(ViewElement): void}} MountedView */

/** Element behavior needed to observe window gestures, bounds and image copies. */
class ViewElement {
  /** Create independent event, layout and drawing state. */
  constructor() {
    /** Registered handler for each exercised event name. @type {Map<string, function(ViewInput): void>} */
    this.handlers = new Map();
    /** String attributes exposed by controls. @type {Map<string, string>} */
    this.attributes = new Map();
    /** Child elements owned by this node. @type {ViewElement[]} */
    this.children = [];
    /** Current checkbox selection. @type {boolean} */
    this.checked = false;
    /** Whether the element is hidden. @type {boolean} */
    this.hidden = false;
    /** Selected camera identifier. @type {string} */
    this.value = '';
    /** Inline positions and chosen dimensions. @type {{left: string, top: string, right: string, bottom: string, width?: string, height?: string}} */
    this.style = {left: '', top: '', right: '', bottom: ''};
    /** Pointer identifiers retained by a gesture. @type {Set<number>} */
    this.capturedPointers = new Set();
    /** Active CSS state classes. @type {Set<string>} */
    this.classes = new Set();
    /** Source and destination rectangles copied into the image. @type {ImageCopy[]} */
    this.draws = [];
    /** Opaque background rectangles requested by the panel. @type {number[][]} */
    this.fills = [];
    /** Recorded image operations without a browser canvas. @type {ImageContext} */
    this.context = {
      drawImage: (...arguments_) => this.draws.push(arguments_),
      fillRect: (...arguments_) => this.fills.push(arguments_),
    };
    /** CSS state operations backed by the class set. @type {{toggle: function(string, boolean): (Set<string>|boolean)}} */
    this.classList = {toggle: (name, enabled) => enabled ? this.classes.add(name) : this.classes.delete(name)};
    /** Factory for owned controls and canvas elements. @type {{createElement: function(): ViewElement}} */
    this.ownerDocument = {createElement: () => new ViewElement()};
  }
  /** Retain a handler for explicit dispatch. @param {string} name Event name. @param {function(ViewInput): void} handler Event callback. */
  addEventListener(name, handler) { this.handlers.set(name, handler); }
  /** Remove the retained event callback. @param {string} name Event name. */
  removeEventListener(name) { this.handlers.delete(name); }
  /** Preserve DOM string coercion. @param {string} name Attribute name. @param {string|number|boolean} value Attribute value. */
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  /** Give the child this node as its owner. @param {ViewElement} child Added element. */
  appendChild(child) {
    this.children.push(child);
    /** Owning node cleared when the element is detached. @type {ViewElement|null} */
    child.parent = this;
  }
  /** Detach this element without removing unrelated siblings. */
  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter(child => child !== this);
    this.parent = null;
  }
  /** Clear the camera options before replacement. */
  replaceChildren() { this.children = []; }
  /** Record an explicit focus request without simulating browser event bubbling. */
  focus() {
    if (this.hidden) return;
    /** Whether this control has received a focus request. @type {boolean} */
    this.focused = true;
    /** Number of explicit focus requests made to this control. @type {number} */
    this.focusCount = (this.focusCount || 0) + 1;
  }
  /** Identify a button target for drag exclusion. @param {string} selector Ancestor selector. @returns {ViewElement|null} Matching button. */
  closest(selector) { return selector === 'button' && this.button ? this : null; }
  /** Start retaining a gesture pointer. @param {number} pointerId Gesture identifier. */
  setPointerCapture(pointerId) { this.capturedPointers.add(pointerId); }
  /** Release a retained gesture pointer. @param {number} pointerId Gesture identifier. */
  releasePointerCapture(pointerId) { this.capturedPointers.delete(pointerId); }
  /** Test ownership of one pointer. @param {number} pointerId Gesture identifier. @returns {boolean} Whether capture remains active. */
  hasPointerCapture(pointerId) { return this.capturedPointers.has(pointerId); }
  /** Apply inline window coordinates to the fixture's original bounds. @returns {ViewBounds} Rectangle in page coordinates. */
  getBoundingClientRect() {
    const bounds = {...this.bounds};
    if (this.positionContainer) {
      if (this.style.width) bounds.width = parseFloat(this.style.width);
      if (this.style.height) bounds.height = parseFloat(this.style.height);
      if (this.style.left) bounds.left = this.positionContainer.bounds.left + parseFloat(this.style.left);
      if (this.style.top) bounds.top = this.positionContainer.bounds.top + parseFloat(this.style.top);
      bounds.right = bounds.left + bounds.width;
      bounds.bottom = bounds.top + bounds.height;
    }
    return bounds;
  }
  /** Return the operation recorder used as the canvas context. @returns {ImageContext} Recorded drawing operations. */
  getContext() { return this.context; }
  /** Invoke one retained handler without browser propagation.
   * @param {string} name Event name.
   * @param {Partial<ViewInput>} [extra] Case-specific pointer, keyboard or callback values.
   */
  dispatch(name, extra = {}) {
    const handler = this.handlers.get(name);
    if (handler) handler({stopPropagation() {}, preventDefault() {}, target: this,
      pointerId: 1, button: 0, isPrimary: true, ...extra});
  }
}

/** Renderer state and draw observations needed to verify borrowed viewport ownership. */
class InsetRenderer {
  /** Start with distinct viewport and scissor bounds at double device resolution. */
  constructor() {
    /** Current renderer viewport in CSS pixels. @type {THREE.Vector4} */
    this.viewport = new THREE.Vector4(0, 0, 800, 600);
    /** Current renderer clipping rectangle. @type {THREE.Vector4} */
    this.scissor = new THREE.Vector4(2, 3, 400, 300);
    /** Whether clipping is enabled. @type {boolean} */
    this.scissorTest = false;
    /** Scene, camera and viewport observed at each render. @type {{scene: THREE.Scene, camera: THREE.Camera, viewport: THREE.Vector4, scissorTest: boolean}[]} */
    this.renders = [];
    /** Physical canvas dimensions supplied to image-copy operations. @type {{width: number, height: number}} */
    this.domElement = {width: 1600, height: 1200};
  }
  /** Copy the borrowed viewport. @param {THREE.Vector4} target Destination. @returns {THREE.Vector4} Copied bounds. */
  getViewport(target) { return target.copy(this.viewport); }
  /** Return the device-to-CSS pixel ratio. @returns {number} Device pixel ratio. */
  getPixelRatio() { return 2; }
  /** Copy the borrowed scissor rectangle. @param {THREE.Vector4} target Destination. @returns {THREE.Vector4} Copied bounds. */
  getScissor(target) { return target.copy(this.scissor); }
  /** Read clipping state. @returns {boolean} Whether clipping is enabled. */
  getScissorTest() { return this.scissorTest; }
  /** Accept either saved or explicit viewport bounds. @param {...(THREE.Vector4|number)} values Saved vector or XYWH coordinates. */
  setViewport(...values) { this.viewport = values[0].isVector4 ? values[0].clone() : new THREE.Vector4(...values); }
  /** Accept either saved or explicit clipping bounds. @param {...(THREE.Vector4|number)} values Saved vector or XYWH coordinates. */
  setScissor(...values) { this.scissor = values[0].isVector4 ? values[0].clone() : new THREE.Vector4(...values); }
  /** Change the clipping state. @param {boolean} value Whether clipping is enabled. */
  setScissorTest(value) { this.scissorTest = value; }
  /** Accept clearing without allocating a GPU buffer. */
  clear() {}
  /** Capture the renderer state at a draw call. @param {THREE.Scene} scene Rendered world. @param {THREE.Camera} camera Selected perspective. */
  render(scene, camera) { this.renders.push({scene, camera, viewport: this.viewport.clone(), scissorTest: this.scissorTest}); }
}

/** A loaded robot with explicitly declared camera frames and a 4:3 lens.
 * @param {string} name Model identity.
 * @param {string[]} [cameraNames] Native sensor and root names.
 * @returns {CameraModel} Loaded robot with annotated camera definitions.
 */
function cameraModel(name, cameraNames = ['wide_stereo_optical_frame']) {
  const object = new THREE.Group();
  object.links = {};
  for (const name of cameraNames) {
    const link = new THREE.Object3D();
    link.name = name;
    object.links[name] = link;
    object.add(link);
  }
  return {name, robot: true, obj: object,
    cameras: cameraNames.map((name, index) => ({name, link: name, forward: [0, 0, 1],
      horizontalAngle: 2 * Math.atan(4 / 3 * Math.tan(Math.PI / 6)),
      verticalAngle: Math.PI / 3, default: index === 0})),
  };
}

/** Mount independent controls with recorded rendering and manually delivered resizes.
 * @param {function(): void} [restoreFocus] Scene-owned action choosing the control to refocus after closing.
 * @returns {MountedView} Panel, elements and observation controls for one case.
 */
function mount(restoreFocus) {
  const stylesheet = BrowserSource.read(path.join(WEB, 'app.css'));
  const cameraRule = stylesheet.match(/\.robot-camera\s*\{([^}]+)\}/)[1];
  const properties = new Map(Array.from(cameraRule.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g),
    match => [match[1], match[2]]));
  const scope = {getComputedStyle: () => ({getPropertyValue: name => properties.get(name) || ''})};
  for (const file of ['robot-camera.js', 'robot-camera-panel.js']) {
    new Function('window', BrowserSource.read(path.join(WEB, 'core', file)))(scope);
  }
  const root = new ViewElement();
  root.bounds = {left: 500, top: 250, width: 320, height: 248};
  const elements = new Map(['source', 'expand', 'close', 'viewport', 'status'].map(name => ['[data-robot-camera="' + name + '"]', new ViewElement()]));
  const header = new ViewElement();
  elements.set('.robot-camera-head', header);
  for (const name of ['expand', 'close']) elements.get('[data-robot-camera="' + name + '"]').button = true;
  root.querySelector = selector => elements.get(selector);
  const viewport = elements.get('[data-robot-camera="viewport"]');
  viewport.bounds = {left: 500, bottom: 500, width: 280, height: 180};
  const layer = new ViewElement();
  const renderer = new InsetRenderer();
  const container = new ViewElement();
  container.bounds = {left: 100, top: 50, bottom: 650, width: 800, height: 600};
  root.positionContainer = container;
  let changes = 0;
  const observers = [];
  /** Resize subscriptions whose delivery is controlled by the case. */
  class SizeObserver {
    /** @param {function(): void} callback Layout update delivered by the harness. */
    constructor(callback) {
      /** Layout callback retained until disconnect. @type {function(): void} */
      this.callback = callback;
      /** Elements that can trigger this observer. @type {Set<ViewElement>} */
      this.observed = new Set();
      observers.push(this);
    }
    /** Subscribe to an element's dimensions. @param {ViewElement} element Observed window or scene. */
    observe(element) { this.observed.add(element); }
    /** Suppress all future manually delivered notifications. */
    disconnect() {
      /** Whether further observer delivery is disabled. @type {boolean} */
      this.disconnected = true;
    }
  }
  const panel = new scope.RobotCameraPanel({THREE, root, layer, renderer,
    scene: new THREE.Scene(), container, invalidate: () => changes++,
    restoreFocus: restoreFocus || (() => layer.focus()), ResizeObserver: SizeObserver});
  return {panel, root, layer, renderer, elements, header, container, changes: () => changes,
    resize: element => observers.filter(observer => observer.observed.has(element) && !observer.disconnected)
      .forEach(observer => observer.callback())};
}

/** Show the inset through its actual layer handler. @param {MountedView} view Mounted controls. */
function show(view) {
  view.layer.checked = true;
  view.layer.dispatch('change');
}

/** Complete a title-bar pointer gesture.
 * @param {MountedView} view Mounted controls.
 * @param {number} left Horizontal pointer displacement.
 * @param {number} top Vertical pointer displacement.
 */
function drag(view, left, top) {
  const bounds = view.root.getBoundingClientRect();
  view.header.dispatch('pointerdown', {clientX: bounds.left + 40, clientY: bounds.top + 12});
  view.header.dispatch('pointermove', {clientX: bounds.left + 40 + left, clientY: bounds.top + 12 + top});
  view.header.dispatch('pointerup');
}

/** Resolve a named corner control from the panel's owned children.
 * @param {MountedView} view Mounted controls.
 * @param {string} corner Corner name published by the control.
 * @returns {ViewElement} The matching resize button.
 */
function resizeGrip(view, corner) {
  const grip = view.root.children.find(child => child.attributes.get('data-robot-camera-resize') === corner);
  assert.ok(grip, 'a resize grip is available at ' + corner);
  return grip;
}

/** Complete a pointer resize from one corner.
 * @param {MountedView} view Mounted controls.
 * @param {string} corner Corner name published by the control.
 * @param {number} horizontal Horizontal corner displacement.
 * @param {number} vertical Vertical corner displacement.
 * @returns {ViewElement} Control that received the gesture.
 */
function resizePanel(view, corner, horizontal, vertical) {
  const bounds = view.root.getBoundingClientRect();
  const grip = resizeGrip(view, corner);
  const clientX = corner.endsWith('left') ? bounds.left : bounds.right;
  const clientY = corner.startsWith('top') ? bounds.top : bounds.bottom;
  grip.dispatch('pointerdown', {clientX, clientY});
  grip.dispatch('pointermove', {clientX: clientX + horizontal, clientY: clientY + vertical});
  grip.dispatch('pointerup');
  return grip;
}

// %% visibility and expansion
test('layer toggle, enlarge, Escape and close stay synchronized', () => {
  const view = mount();
  assert.equal(view.root.hidden, true);
  view.layer.checked = true;
  view.layer.dispatch('change');
  assert.equal(view.root.hidden, false);
  view.elements.get('[data-robot-camera="expand"]').dispatch('click');
  assert.equal(view.root.classes.has('expanded'), true);
  view.root.dispatch('keydown', {key: 'Escape'});
  assert.equal(view.root.classes.has('expanded'), false);
  assert.equal(view.root.hidden, false);
  view.elements.get('[data-robot-camera="close"]').dispatch('click');
  assert.equal(view.layer.checked, false);
  assert.equal(view.root.hidden, true);
});

test('hidden and missing-camera panels skip the second render', () => {
  const view = mount();
  view.panel.render();
  view.layer.checked = true;
  view.layer.dispatch('change');
  view.panel.setModels([]);
  view.panel.render();
  assert.equal(view.renderer.renders.length, 0);
  assert.equal(view.elements.get('[data-robot-camera="status"]').hidden, false);
});

test('inset centers the camera image in CSS coordinates and restores the main renderer state', () => {
  const view = mount();
  const originalViewport = view.renderer.viewport.clone();
  const originalScissor = view.renderer.scissor.clone();
  view.panel.setModels([cameraModel('pr2')]);
  view.layer.checked = true;
  view.layer.dispatch('change');
  view.panel.render();
  assert.equal(view.renderer.renders.length, 1);
  assert.deepEqual(view.renderer.renders[0].viewport.toArray(), [420, 150, 240, 180]);
  assert.equal(view.renderer.renders[0].scissorTest, true);
  assert.deepEqual(view.renderer.viewport, originalViewport);
  assert.deepEqual(view.renderer.scissor, originalScissor);
  assert.equal(view.renderer.scissorTest, false);
});

test('destroy disconnects resize observation and removes control listeners', () => {
  const view = mount();
  const before = view.changes();
  view.panel.destroy();
  view.layer.checked = true;
  view.layer.dispatch('change');
  assert.equal(view.changes(), before);
  assert.equal(view.panel.observer.disconnected, true);
});

// %% dragging within the scene
test('the title bar moves the panel by the pointer delta and releases it on pointerup', () => {
  const view = mount();
  show(view);
  view.header.dispatch('pointerdown', {pointerId: 7, clientX: 540, clientY: 262});
  assert.equal(view.header.hasPointerCapture(7), true);
  view.header.dispatch('pointermove', {pointerId: 7, clientX: 590, clientY: 312});
  assert.deepEqual(view.root.style, {left: '450px', top: '250px', right: 'auto', bottom: 'auto'});
  view.header.dispatch('pointerup', {pointerId: 7});
  assert.equal(view.header.hasPointerCapture(7), false);
  view.header.dispatch('pointermove', {pointerId: 7, clientX: 640, clientY: 362});
  assert.equal(view.root.style.left, '450px');
  assert.equal(view.root.style.top, '250px');
});

test('dragging keeps all four panel edges inside the scene', () => {
  const view = mount();
  show(view);
  drag(view, -2000, -2000);
  assert.equal(view.root.style.left, '8px');
  assert.equal(view.root.style.top, '8px');
  drag(view, 2000, 2000);
  assert.equal(view.root.style.left, '472px');
  assert.equal(view.root.style.top, '344px');
});

test('buttons, nonprimary pointers, and right clicks do not start dragging', () => {
  for (const event of [{button: 2}, {isPrimary: false}, {buttonName: 'expand'}, {buttonName: 'close'}]) {
    const view = mount();
    show(view);
    const target = event.buttonName ? view.elements.get('[data-robot-camera="' + event.buttonName + '"]') : view.header;
    view.header.dispatch('pointerdown', {clientX: 540, clientY: 262, target, ...event});
    view.header.dispatch('pointermove', {clientX: 590, clientY: 312});
    assert.equal(view.header.hasPointerCapture(1), false);
    assert.equal(view.root.style.left, '');
    assert.equal(view.root.style.top, '');
  }
});

test('cancelled or lost pointers cannot continue moving the panel', () => {
  for (const ending of ['pointercancel', 'lostpointercapture']) {
    const view = mount();
    show(view);
    view.header.dispatch('pointerdown', {clientX: 540, clientY: 262});
    view.header.dispatch('pointermove', {clientX: 590, clientY: 312});
    view.header.dispatch(ending);
    view.header.dispatch('pointermove', {clientX: 640, clientY: 362});
    assert.equal(view.root.style.left, '450px');
    assert.equal(view.root.style.top, '250px');
  }
});

test('expansion suspends dragging and restores the saved position within the resized scene', () => {
  const view = mount();
  show(view);
  drag(view, 50, 50);
  view.elements.get('[data-robot-camera="expand"]').dispatch('click');
  assert.deepEqual(view.root.style, {left: '', top: '', right: '', bottom: ''});
  drag(view, -100, -100);
  assert.equal(view.root.style.left, '');
  assert.equal(view.root.style.top, '');
  view.container.bounds.width = 600;
  view.container.bounds.height = 400;
  view.resize(view.container);
  view.root.dispatch('keydown', {key: 'Escape'});
  assert.equal(view.root.style.left, '272px');
  assert.equal(view.root.style.top, '144px');
});

test('container and panel resizing keeps a dragged panel within reach', () => {
  const view = mount();
  show(view);
  drag(view, 50, 50);
  view.container.bounds.width = 600;
  view.container.bounds.height = 400;
  view.resize(view.container);
  assert.equal(view.root.style.left, '272px');
  assert.equal(view.root.style.top, '144px');
  view.root.bounds.width = 400;
  view.root.bounds.height = 300;
  view.resize(view.elements.get('[data-robot-camera="viewport"]'));
  assert.equal(view.root.style.left, '192px');
  assert.equal(view.root.style.top, '92px');
});

test('a hidden panel retains its dragged position and clamps it when shown again', () => {
  const view = mount();
  show(view);
  drag(view, 50, 50);
  view.elements.get('[data-robot-camera="close"]').dispatch('click');
  show(view);
  assert.equal(view.root.style.left, '450px');
  assert.equal(view.root.style.top, '250px');
  view.elements.get('[data-robot-camera="close"]').dispatch('click');
  view.container.bounds.width = 600;
  view.container.bounds.height = 400;
  show(view);
  assert.equal(view.root.style.left, '272px');
  assert.equal(view.root.style.top, '144px');
});

test('destroy releases an active drag and removes its input handlers', () => {
  const view = mount();
  show(view);
  view.header.dispatch('pointerdown', {clientX: 540, clientY: 262});
  view.header.dispatch('pointermove', {clientX: 590, clientY: 312});
  view.panel.destroy();
  assert.equal(view.header.hasPointerCapture(1), false);
  view.header.dispatch('pointermove', {clientX: 640, clientY: 362});
  assert.equal(view.root.style.left, '450px');
  assert.equal(view.root.style.top, '250px');
  assert.equal(view.header.handlers.size, 0);
});

// %% resizing within the scene
test('the bottom-right grip resizes the panel while keeping the opposite corner fixed', () => {
  const view = mount();
  show(view);
  const before = view.root.getBoundingClientRect();
  resizePanel(view, 'bottom-right', 50, 60);
  assert.deepEqual(view.root.style, {
    left: '400px', top: '200px', right: 'auto', bottom: 'auto', width: '370px', height: '308px',
  });
  const after = view.root.getBoundingClientRect();
  assert.equal(after.left, before.left);
  assert.equal(after.top, before.top);
});

test('the top-left grip grows the default panel toward the available scene space', () => {
  const view = mount();
  show(view);
  const before = view.root.getBoundingClientRect();
  resizePanel(view, 'top-left', -150, -100);
  assert.deepEqual(view.root.style, {
    left: '250px', top: '100px', right: 'auto', bottom: 'auto', width: '470px', height: '348px',
  });
  const after = view.root.getBoundingClientRect();
  assert.equal(after.right, before.right);
  assert.equal(after.bottom, before.bottom);
});

test('the other corner grips preserve the diagonally opposite corner', () => {
  for (const corner of ['top-right', 'bottom-left']) {
    const view = mount();
    show(view);
    const before = view.root.getBoundingClientRect();
    const isLeft = corner.endsWith('left');
    const isTop = corner.startsWith('top');
    resizePanel(view, corner, isLeft ? -50 : 50, isTop ? -60 : 60);
    const after = view.root.getBoundingClientRect();
    assert.equal(after.width, before.width + 50);
    assert.equal(after.height, before.height + 60);
    assert.equal(after[isLeft ? 'right' : 'left'], before[isLeft ? 'right' : 'left']);
    assert.equal(after[isTop ? 'bottom' : 'top'], before[isTop ? 'bottom' : 'top']);
  }
});

test('each resize grip stops at the minimum usable size without moving its anchor', () => {
  for (const corner of ['top-left', 'top-right', 'bottom-left', 'bottom-right']) {
    const view = mount();
    show(view);
    const before = view.root.getBoundingClientRect();
    const isLeft = corner.endsWith('left');
    const isTop = corner.startsWith('top');
    resizePanel(view, corner, isLeft ? 2000 : -2000, isTop ? 2000 : -2000);
    const after = view.root.getBoundingClientRect();
    assert.equal(after.width, 220);
    assert.equal(after.height, 140);
    assert.equal(after[isLeft ? 'right' : 'left'], before[isLeft ? 'right' : 'left']);
    assert.equal(after[isTop ? 'bottom' : 'top'], before[isTop ? 'bottom' : 'top']);
  }
});

test('resizing stops at every scene edge', () => {
  const upper = mount();
  show(upper);
  resizePanel(upper, 'top-left', -2000, -2000);
  assert.deepEqual(upper.root.style, {
    left: '8px', top: '8px', right: 'auto', bottom: 'auto', width: '712px', height: '440px',
  });
  const lower = mount();
  show(lower);
  resizePanel(lower, 'bottom-right', 2000, 2000);
  assert.deepEqual(lower.root.style, {
    left: '400px', top: '200px', right: 'auto', bottom: 'auto', width: '392px', height: '392px',
  });
});

test('resizing captures its pointer and ignores movement from other pointers', () => {
  const view = mount();
  show(view);
  const grip = resizeGrip(view, 'top-left');
  grip.dispatch('pointerdown', {pointerId: 7, clientX: 500, clientY: 250, pointerType: 'touch'});
  assert.equal(grip.hasPointerCapture(7), true);
  grip.dispatch('pointermove', {pointerId: 8, clientX: 350, clientY: 150});
  assert.equal(view.root.getBoundingClientRect().width, 320);
  assert.equal(view.root.getBoundingClientRect().height, 248);
  grip.dispatch('pointermove', {pointerId: 7, clientX: 350, clientY: 150});
  assert.equal(view.root.style.width, '470px');
  assert.equal(view.root.style.height, '348px');
});

test('released, cancelled, and lost resize pointers cannot keep changing size', () => {
  for (const ending of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    const view = mount();
    show(view);
    const grip = resizeGrip(view, 'top-left');
    grip.dispatch('pointerdown', {pointerId: 7, clientX: 500, clientY: 250});
    grip.dispatch('pointermove', {pointerId: 7, clientX: 350, clientY: 150});
    grip.dispatch(ending, {pointerId: 7});
    grip.dispatch('pointermove', {pointerId: 7, clientX: 250, clientY: 100});
    assert.equal(grip.hasPointerCapture(7), false);
    assert.equal(view.root.style.width, '470px');
    assert.equal(view.root.style.height, '348px');
  }
});

test('nonprimary pointers and right clicks do not start resizing', () => {
  for (const event of [{button: 2}, {isPrimary: false}]) {
    const view = mount();
    show(view);
    const grip = resizeGrip(view, 'top-left');
    grip.dispatch('pointerdown', {clientX: 500, clientY: 250, ...event});
    grip.dispatch('pointermove', {clientX: 350, clientY: 150});
    assert.equal(grip.hasPointerCapture(1), false);
    assert.equal(view.root.getBoundingClientRect().width, 320);
    assert.equal(view.root.getBoundingClientRect().height, 248);
  }
});

test('hidden, expanded, and currently dragged panels cannot start resizing', () => {
  for (const state of ['hidden', 'expanded', 'dragging']) {
    const view = mount();
    if (state !== 'hidden') show(view);
    if (state === 'expanded') view.elements.get('[data-robot-camera="expand"]').dispatch('click');
    if (state === 'dragging') view.header.dispatch('pointerdown', {clientX: 540, clientY: 262});
    const grip = resizeGrip(view, 'top-left');
    const before = {...view.root.style};
    grip.dispatch('pointerdown', {clientX: 500, clientY: 250});
    grip.dispatch('pointermove', {clientX: 350, clientY: 150});
    assert.equal(grip.hasPointerCapture(1), false);
    assert.deepEqual(view.root.style, before);
  }
});

test('the chosen size survives hiding and expansion', () => {
  const view = mount();
  show(view);
  resizePanel(view, 'top-left', -150, -100);
  const chosen = {...view.root.style};
  view.elements.get('[data-robot-camera="close"]').dispatch('click');
  show(view);
  assert.deepEqual(view.root.style, chosen);
  view.elements.get('[data-robot-camera="expand"]').dispatch('click');
  assert.equal(view.root.style.width, '');
  assert.equal(view.root.style.height, '');
  view.root.dispatch('keydown', {key: 'Escape'});
  assert.deepEqual(view.root.style, chosen);
});

test('a smaller scene clamps both the chosen size and panel position', () => {
  const view = mount();
  show(view);
  resizePanel(view, 'top-left', -150, -100);
  view.container.bounds.width = 400;
  view.container.bounds.height = 300;
  view.resize(view.container);
  assert.deepEqual(view.root.style, {
    left: '8px', top: '8px', right: 'auto', bottom: 'auto', width: '384px', height: '284px',
  });
});

test('hiding or expanding releases an active resize', () => {
  for (const control of ['close', 'expand']) {
    const view = mount();
    show(view);
    const grip = resizeGrip(view, 'top-left');
    grip.dispatch('pointerdown', {clientX: 500, clientY: 250});
    grip.dispatch('pointermove', {clientX: 350, clientY: 150});
    view.elements.get('[data-robot-camera="' + control + '"]').dispatch('click');
    assert.equal(grip.hasPointerCapture(1), false);
    const stopped = {...view.root.style};
    grip.dispatch('pointermove', {clientX: 250, clientY: 100});
    assert.deepEqual(view.root.style, stopped);
  }
});

test('destroy releases an active resize and removes all grip handlers', () => {
  const view = mount();
  show(view);
  const grip = resizeGrip(view, 'top-left');
  grip.dispatch('pointerdown', {clientX: 500, clientY: 250});
  grip.dispatch('pointermove', {clientX: 350, clientY: 150});
  const grips = ['top-left', 'top-right', 'bottom-left', 'bottom-right'].map(corner => resizeGrip(view, corner));
  view.panel.destroy();
  assert.equal(grip.hasPointerCapture(1), false);
  grip.dispatch('pointermove', {clientX: 250, clientY: 100});
  assert.equal(view.root.style.width, '470px');
  assert.equal(view.root.style.height, '348px');
  for (const handle of grips) assert.equal(handle.handlers.size, 0);
});

// %% asynchronously loaded robot instances
test('the scene primary robot supplies the default camera even when its name sorts last', () => {
  const view = mount();
  const first = cameraModel('alpha', ['camera_optical_frame']);
  const primary = cameraModel('zebra', ['camera_optical_frame']);
  view.panel.setModels([first]);
  view.panel.setModels([first, primary], primary);
  assert.equal(view.panel.selected.link, primary.obj.links.camera_optical_frame);
});

test('an explicit camera choice survives another model finishing loading', () => {
  const view = mount();
  const robot = cameraModel('pr2', ['wide_stereo_optical_frame', 'r_forearm_cam_optical_frame']);
  view.panel.setModels([robot]);
  const hand = view.panel.sources.find(source => source.link === robot.obj.links.r_forearm_cam_optical_frame);
  const select = view.elements.get('[data-robot-camera="source"]');
  select.value = hand.id;
  select.dispatch('change');
  view.panel.setModels([robot, {name: 'room', robot: false, obj: {links: {}}}], robot);
  assert.equal(view.panel.selected.link, hand.link);
  assert.equal(select.value, hand.id);
});

test('the window centers an undistorted image on opaque margins at device resolution', () => {
  const view = mount();
  view.panel.setModels([cameraModel('pr2')]);
  view.layer.checked = true;
  view.layer.dispatch('change');
  view.panel.render();
  const viewport = view.elements.get('[data-robot-camera="viewport"]');
  const canvas = viewport.children[0];
  assert.ok(canvas);
  assert.equal(canvas.width, 560);
  assert.equal(canvas.height, 360);
  assert.deepEqual(canvas.fills, [[0, 0, 560, 360]]);
  assert.deepEqual(canvas.draws, [[view.renderer.domElement, 840, 540, 480, 360, 40, 0, 480, 360]]);
});

test('tall and wide windows preserve the camera projection and clear obsolete image areas', () => {
  const view = mount();
  view.panel.setModels([cameraModel('pr2')]);
  show(view);
  const viewport = view.elements.get('[data-robot-camera="viewport"]');
  viewport.bounds = {left: 500, bottom: 500, width: 400, height: 180};
  view.panel.render();
  const projection = view.panel.pose.camera.projectionMatrix.clone();
  const canvas = viewport.children[0];
  assert.deepEqual(view.renderer.renders[0].viewport.toArray(), [480, 150, 240, 180]);
  assert.deepEqual(canvas.draws[0].slice(5), [160, 0, 480, 360]);

  viewport.bounds = {left: 500, bottom: 500, width: 280, height: 360};
  view.panel.render();
  assert.equal(view.panel.pose.camera.projectionMatrix.equals(projection), true);
  assert.deepEqual(view.renderer.renders[1].viewport.toArray(), [400, 225, 280, 210]);
  assert.deepEqual(canvas.draws[1].slice(5), [0, 150, 560, 420]);
  assert.deepEqual(canvas.fills, [[0, 0, 800, 360], [0, 0, 560, 720]]);
  assert.equal(canvas.context.fillStyle, '#000000');
});

// %% renderer ownership and scene replacement
test('a failed inset render restores every shared renderer setting', () => {
  for (const operation of ['render', 'copy']) {
    const view = mount();
    view.panel.setModels([cameraModel('pr2')]);
    show(view);
    view.renderer.scissorTest = true;
    const viewport = view.renderer.viewport.clone();
    const scissor = view.renderer.scissor.clone();
    const failure = new Error('Image unavailable');
    if (operation === 'render') view.renderer.render = () => { throw failure; };
    else view.panel.context.drawImage = () => { throw failure; };
    assert.throws(() => view.panel.render(), error => error === failure);
    assert.deepEqual(view.renderer.viewport, viewport);
    assert.deepEqual(view.renderer.scissor, scissor);
    assert.equal(view.renderer.scissorTest, true);
  }
});

test('notifying an equal-size model collection follows the replacement robot', () => {
  const view = mount();
  const original = cameraModel('pr2');
  const replacement = cameraModel('pr2');
  view.panel.setModels([original]);
  view.panel.setModels([replacement]);
  assert.equal(view.panel.selected.link, replacement.obj.links.wide_stereo_optical_frame);
});

test('a replaced articulated object refreshes an existing model camera', () => {
  const view = mount();
  const robot = cameraModel('pr2');
  view.panel.setModels([robot]);
  robot.obj = cameraModel('replacement').obj;
  view.panel.setModels([robot]);
  assert.equal(view.panel.selected.link, robot.obj.links.wide_stereo_optical_frame);
});

test('a previous scene camera choice does not override the next scene primary robot', () => {
  const view = mount();
  const robot = cameraModel('pr2');
  view.panel.setModels([robot]);
  view.elements.get('[data-robot-camera="source"]').dispatch('change');
  view.panel.setModels([]);
  const first = cameraModel('alpha', ['camera_optical_frame']);
  const primary = cameraModel('zebra', ['camera_optical_frame']);
  view.panel.setModels([first]);
  view.panel.setModels([first, primary], primary);
  assert.equal(view.panel.selected.link, primary.obj.links.camera_optical_frame);
});

test('destroy removes only the owned image and resize controls and hides the view', () => {
  const view = mount();
  const unrelated = new ViewElement();
  view.root.appendChild(unrelated);
  show(view);
  view.panel.destroy();
  assert.equal(view.root.hidden, true);
  assert.equal(view.layer.checked, false);
  assert.deepEqual(view.root.children, [unrelated]);
  assert.deepEqual(view.elements.get('[data-robot-camera="viewport"]').children, []);
  view.panel.destroy();
  assert.deepEqual(view.root.children, [unrelated]);
});

// %% keyboard access
test('double-click expansion focuses its restore control so Escape reaches the panel', () => {
  const view = mount();
  const expand = view.elements.get('[data-robot-camera="expand"]');
  assert.equal(expand.focusCount, undefined);
  show(view);
  view.layer.focus();
  view.elements.get('[data-robot-camera="viewport"]').dispatch('dblclick');
  assert.equal(expand.focused, true);
  assert.equal(expand.focusCount, 1);
  view.root.dispatch('keydown', {key: 'Escape'});
  assert.equal(view.root.classes.has('expanded'), false);
  assert.equal(view.root.hidden, false);
  assert.equal(expand.focusCount, 1);
});

test('arrow keys move each focused corner by ten pixels and preserve its opposite anchor', () => {
  for (const corner of ['top-left', 'top-right', 'bottom-left', 'bottom-right']) {
    const view = mount();
    show(view);
    const before = view.root.getBoundingClientRect();
    const grip = resizeGrip(view, corner);
    const left = corner.endsWith('left');
    const top = corner.startsWith('top');
    let consumed = 0;
    grip.dispatch('keydown', {key: left ? 'ArrowLeft' : 'ArrowRight',
      preventDefault: () => consumed++, stopPropagation: () => consumed++});
    grip.dispatch('keydown', {key: top ? 'ArrowUp' : 'ArrowDown',
      preventDefault: () => consumed++, stopPropagation: () => consumed++});
    const after = view.root.getBoundingClientRect();
    assert.equal(after.width, before.width + 10);
    assert.equal(after.height, before.height + 10);
    assert.equal(after[left ? 'right' : 'left'], before[left ? 'right' : 'left']);
    assert.equal(after[top ? 'bottom' : 'top'], before[top ? 'bottom' : 'top']);
    assert.equal(consumed, 4);
  }
});

test('keyboard resizing respects scene edges and the same minimum size as pointer resizing', () => {
  const view = mount();
  const pointer = mount();
  show(view);
  show(pointer);
  const grip = resizeGrip(view, 'bottom-right');
  resizePanel(pointer, 'bottom-right', 2000, 2000);
  for (let count = 0; count < 100; count++) {
    grip.dispatch('keydown', {key: 'ArrowRight'});
    grip.dispatch('keydown', {key: 'ArrowDown'});
  }
  assert.equal(view.root.style.width, pointer.root.style.width);
  assert.equal(view.root.style.height, pointer.root.style.height);
  resizePanel(pointer, 'bottom-right', -2000, -2000);
  for (let count = 0; count < 100; count++) {
    grip.dispatch('keydown', {key: 'ArrowLeft'});
    grip.dispatch('keydown', {key: 'ArrowUp'});
  }
  assert.equal(view.root.style.width, pointer.root.style.width);
  assert.equal(view.root.style.height, pointer.root.style.height);
});

test('unrelated keys and inactive or captured windows leave keyboard resizing untouched', () => {
  for (const state of ['other-key', 'hidden', 'expanded', 'dragging', 'resizing']) {
    const view = mount();
    if (state !== 'hidden') show(view);
    if (state === 'expanded') view.elements.get('[data-robot-camera="expand"]').dispatch('click');
    if (state === 'dragging') view.header.dispatch('pointerdown', {clientX: 540, clientY: 262});
    const grip = resizeGrip(view, 'top-left');
    if (state === 'resizing') grip.dispatch('pointerdown', {clientX: 500, clientY: 250});
    const before = {...view.root.style};
    let consumed = false;
    grip.dispatch('keydown', {key: state === 'other-key' ? 'Escape' : 'ArrowLeft',
      preventDefault: () => consumed = true});
    assert.deepEqual(view.root.style, before);
    assert.equal(consumed, false);
  }
});


test('closing with folded Layers delegates focus to the visible scene control', () => {
  for (const action of ['close', 'Escape']) {
    const foldButton = new ViewElement();
    const view = mount(() => foldButton.focus());
    show(view);
    view.layer.hidden = true;
    if (action === 'close') view.elements.get('[data-robot-camera="close"]').dispatch('click');
    else view.root.dispatch('keydown', {key: action});
    assert.equal(view.root.hidden, true);
    assert.equal(foldButton.focused, true);
    assert.equal(view.layer.focused, undefined);
  }
});
