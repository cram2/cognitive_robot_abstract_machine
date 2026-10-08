/* %% robot viewpoint controls and shared renderer viewport */
(function (global) {
  'use strict';

  const WINDOW_MARGIN = 8;
  const MINIMUM_SIZE = Object.freeze({width: 220, height: 140});
  const KEYBOARD_RESIZE_STEP = 10;
  /** Distance in CSS pixels moved by one arrow-key press on a corner control. */
  const RESIZE_KEYS = Object.freeze({
    ArrowLeft: {horizontal: -1, vertical: 0},
    ArrowRight: {horizontal: 1, vertical: 0},
    ArrowUp: {horizontal: 0, vertical: -1},
    ArrowDown: {horizontal: 0, vertical: 1},
  });
  const RESIZE_CORNERS = Object.freeze([
    {name: 'top-left', horizontal: -1, vertical: -1},
    {name: 'top-right', horizontal: 1, vertical: -1},
    {name: 'bottom-left', horizontal: -1, vertical: 1},
    {name: 'bottom-right', horizontal: 1, vertical: 1},
  ]);

  /** A movable robot view with camera selection and an image from the shared renderer. */
  class RobotCameraPanel {
    /**
     * Mount the view and its input handlers.
     *
     * @param {object} options Panel elements, scene resources and browser dependencies.
     * @param {typeof THREE} options.THREE Three.js constructors for the camera and viewport state.
     * @param {HTMLElement} options.root Window containing the robot camera controls.
     * @param {HTMLInputElement} options.layer Checkbox controlling visibility.
     * @param {THREE.WebGLRenderer} options.renderer Scene renderer borrowed for camera images.
     * @param {THREE.Scene} options.scene Articulated scene to render.
     * @param {HTMLElement} options.container Scene area bounding window movement.
     * @param {function(): void} options.invalidate Request another scene frame.
     * @param {typeof ResizeObserver} options.ResizeObserver Browser resize observer constructor.
     */
    constructor(options) {
      this.renderer = options.renderer;
      /** Shared scene renderer, retained without taking ownership. */
      this.scene = options.scene;
      /** Articulated world viewed by the camera. */
      this.container = options.container;
      /** Scene area that bounds the movable window. */
      this.invalidate = options.invalidate;
      /** Request another frame after an input or layout change. */
      this.root = options.root;
      /** Window containing the camera controls and image. */
      this.layer = options.layer;
      /** Visibility checkbox in the scene controls. */
      this.pose = new global.RobotCamera.Pose(options.THREE);
      /** Perspective camera following the selected articulated link. */
      this.sources = [];
      /** Available camera frames from the current scene models. */
      this.models = [];
      /** Model, loaded object and camera-list identities at the last source refresh. */
      this.primary = null;
      /** Robot whose default camera is preferred before a user chooses one. */
      this.selected = null;
      /** Source currently used to render the robot view. */
      this.userSelected = false;
      /** Whether an explicit selection takes precedence over the primary robot. */
      this.expanded = false;
      /** Whether the window fills the scene instead of using the saved inset bounds. */
      this.position = null;
      /** Last inset position relative to the scene container. */
      this.drag = null;
      /** Captured pointer and its offset within the window. */
      this.size = null;
      /** Chosen inset size, retained while the scene is temporarily smaller. */
      this.resizing = null;
      /** Captured resize pointer and the window geometry where it started. */
      this.listeners = [];
      /** Event subscriptions owned by the panel. */
      this.header = this.root.querySelector('.robot-camera-head');
      /** Title bar accepting window movement gestures. */
      this.header.title = 'Drag to move the robot view';
      this.viewport = this.root.querySelector('[data-robot-camera="viewport"]');
      /** Image area whose bounds determine the camera image size. */
      this.canvas = this.root.ownerDocument.createElement('canvas');
      /** Owned image copy that leaves the overview canvas unobstructed. */
      this.canvas.setAttribute('aria-hidden', 'true');
      this.context = this.canvas.getContext('2d', {alpha: false});
      /** Opaque drawing context for the robot view and its letterbox margins. */
      this.viewport.appendChild(this.canvas);
      this.status = this.root.querySelector('[data-robot-camera="status"]');
      /** Message shown when the scene has no available cameras. */
      this.select = this.root.querySelector('[data-robot-camera="source"]');
      /** Native select listing the scene's camera frames. */
      this.expand = this.root.querySelector('[data-robot-camera="expand"]');
      /** Button toggling inset and expanded window sizes. */
      this.savedViewport = new options.THREE.Vector4();
      /** Borrowed renderer viewport to restore after a camera image. */
      this.savedScissor = new options.THREE.Vector4();
      /** Borrowed renderer scissor bounds to restore after a camera image. */
      this.resizeHandles = [];
      /** Corner controls created and removed by the panel. */
      this.bindControls();
      this.addResizeHandles();
      this.observer = new options.ResizeObserver(() => {
        this.fitSize();
        this.fitPosition();
        this.invalidate();
      });
      /** Observer keeping the window inside a resized scene. */
      this.observer.observe(this.viewport);
      this.observer.observe(this.container);
      this.setVisible(false);
      this.setExpanded(false);
      this.setModels([]);
    }

    /** Bind camera selection, visibility and movement to the window controls. */
    bindControls() {
      this.listen(this.layer, 'change', () => this.setVisible(this.layer.checked));
      this.listen(this.expand, 'click', () => this.setExpanded(!this.expanded));
      this.listen(this.root.querySelector('[data-robot-camera="close"]'), 'click', () => {
        this.setVisible(false);
        this.layer.focus();
      });
      this.listen(this.select, 'change', () => {
        this.selected = this.sources.find(source => source.id === this.select.value) || null;
        this.userSelected = true;
        this.invalidate();
      });
      this.listen(this.viewport, 'dblclick', () => this.setExpanded(!this.expanded));
      this.listen(this.root, 'keydown', event => {
        if (event.key !== 'Escape') return;
        event.stopPropagation();
        if (this.expanded) this.setExpanded(false);
        else { this.setVisible(false); this.layer.focus(); }
      });
      for (const name of ['pointerdown', 'wheel', 'dblclick']) {
        this.listen(this.root, name, event => event.stopPropagation());
      }
      this.listen(this.header, 'pointerdown', event => this.startDrag(event));
      this.listen(this.header, 'pointermove', event => this.moveDrag(event));
      for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) {
        this.listen(this.header, name, event => this.endDrag(event));
      }
    }

    /**
     * Track subscriptions for removal when the panel is destroyed.
     * @param {EventTarget} element Control receiving the event.
     * @param {string} name Browser event name.
     * @param {EventListener} handler Panel action for that event.
     */
    listen(element, name, handler) {
      element.addEventListener(name, handler);
      this.listeners.push({element, name, handler});
    }

    /**
     * Keep the layer checkbox and window visibility in agreement.
     * @param {boolean} visible Whether the robot view is shown.
     */
    setVisible(visible) {
      if (!visible) { this.endDrag(); this.endResize(); }
      this.layer.checked = visible;
      this.root.hidden = !visible;
      if (visible) { this.fitSize(); this.fitPosition(); }
      this.invalidate();
    }

    /**
     * Switch between the inset and a large view within the scene panel.
     * @param {boolean} expanded Whether the view fills the scene area.
     */
    setExpanded(expanded) {
      this.endDrag();
      this.endResize();
      this.expanded = expanded;
      this.root.classList.toggle('expanded', expanded);
      if (expanded) {
        for (const edge of ['left', 'top', 'right', 'bottom']) this.root.style[edge] = '';
        if (this.size) { this.root.style.width = ''; this.root.style.height = ''; }
      } else { this.fitSize(); this.fitPosition(); }
      const label = expanded ? 'Restore robot view' : 'Enlarge robot view';
      this.expand.textContent = expanded ? '↙' : '⛶';
      this.expand.title = label;
      this.expand.setAttribute('aria-label', label);
      this.expand.setAttribute('aria-expanded', expanded);
      if (expanded) this.expand.focus({preventScroll: true});
      this.invalidate();
    }

    // %% window movement
    /**
     * Capture a primary pointer on the title bar, leaving buttons clickable.
     * @param {PointerEvent} event Pointer pressed on the title bar.
     */
    startDrag(event) {
      if (this.root.hidden || this.expanded || this.drag || this.resizing || event.button !== 0 ||
          event.isPrimary === false || event.target.closest('button')) return;
      const bounds = this.root.getBoundingClientRect();
      this.drag = {pointer: event.pointerId, offsetX: event.clientX - bounds.left,
        offsetY: event.clientY - bounds.top};
      this.header.setPointerCapture(event.pointerId);
      this.root.classList.toggle('dragging', true);
      event.preventDefault();
    }

    /**
     * Move the inset with its captured pointer while keeping it inside the scene.
     * @param {PointerEvent} event Current position of the dragging pointer.
     */
    moveDrag(event) {
      if (!this.drag || event.pointerId !== this.drag.pointer) return;
      const container = this.container.getBoundingClientRect();
      this.position = {left: event.clientX - container.left - this.drag.offsetX,
        top: event.clientY - container.top - this.drag.offsetY};
      this.fitPosition();
      this.invalidate();
    }

    /**
     * Release a completed or interrupted drag without moving the overview camera.
     * @param {PointerEvent} [event] Pointer that ended, or omitted to cancel any active drag.
     */
    endDrag(event) {
      if (!this.drag || (event && event.pointerId !== this.drag.pointer)) return;
      const pointer = this.drag.pointer;
      this.drag = null;
      if (this.header.hasPointerCapture(pointer)) this.header.releasePointerCapture(pointer);
      this.root.classList.toggle('dragging', false);
    }

    /** Keep a manually placed window reachable after resizing or restoring it. */
    fitPosition() {
      if (!this.position || this.root.hidden || this.expanded) return;
      const container = this.container.getBoundingClientRect();
      const bounds = this.root.getBoundingClientRect();
      if (!container.width || !container.height) return;
      const horizontalMargin = Math.min(WINDOW_MARGIN, Math.max(0, (container.width - bounds.width) / 2));
      const verticalMargin = Math.min(WINDOW_MARGIN, Math.max(0, (container.height - bounds.height) / 2));
      this.position.left = Math.max(horizontalMargin, Math.min(this.position.left, container.width - bounds.width - horizontalMargin));
      this.position.top = Math.max(verticalMargin, Math.min(this.position.top, container.height - bounds.height - verticalMargin));
      this.root.style.left = this.position.left + 'px';
      this.root.style.top = this.position.top + 'px';
      this.root.style.right = 'auto';
      this.root.style.bottom = 'auto';
    }

    // %% window resizing
    /** Provide corner grips accepting pointers and arrow keys. */
    addResizeHandles() {
      for (const corner of RESIZE_CORNERS) {
        const handle = this.root.ownerDocument.createElement('button');
        handle.type = 'button';
        handle.className = 'robot-camera-resize';
        handle.setAttribute('data-robot-camera-resize', corner.name);
        handle.setAttribute('aria-label', 'Resize robot view from ' + corner.name.replace('-', ' '));
        handle.title = 'Drag or use arrow keys to resize the robot view';
        handle.setAttribute('aria-description', 'Arrow keys move this corner by ' + KEYBOARD_RESIZE_STEP + ' pixels.');
        this.root.appendChild(handle);
        this.resizeHandles.push(handle);
        this.listen(handle, 'pointerdown', event => this.startResize(event, handle, corner));
        this.listen(handle, 'pointermove', event => this.moveResize(event));
        this.listen(handle, 'keydown', event => this.resizeWithKeyboard(event, corner));
        for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) {
          this.listen(handle, name, event => this.endResize(event));
        }
      }
    }

    /**
     * Fix the opposite corner before capturing a resize gesture.
     * @param {PointerEvent} event Pointer pressed on a corner grip.
     * @param {HTMLButtonElement} handle Corner control retaining pointer capture.
     * @param {{name: string, horizontal: number, vertical: number}} corner Resize directions.
     */
    startResize(event, handle, corner) {
      if (this.root.hidden || this.expanded || this.drag || this.resizing ||
          event.button !== 0 || event.isPrimary === false) return;
      this.resizing = {...this.resizeGeometry(corner), pointer: event.pointerId, handle,
        clientX: event.clientX, clientY: event.clientY};
      handle.setPointerCapture(event.pointerId);
      event.preventDefault();
      event.stopPropagation();
    }

    /**
     * Resize toward the captured pointer without moving the opposite corner.
     * @param {PointerEvent} event Current position of the resizing pointer.
     */
    moveResize(event) {
      const gesture = this.resizing;
      if (!gesture || event.pointerId !== gesture.pointer) return;
      this.resizeWindow(gesture, event.clientX - gesture.clientX, event.clientY - gesture.clientY);
    }

    /**
     * Move the focused corner with arrow keys without capturing a pointer.
     * @param {KeyboardEvent} event Key pressed on a corner control.
     * @param {{horizontal: number, vertical: number}} corner Resize directions.
     */
    resizeWithKeyboard(event, corner) {
      const direction = RESIZE_KEYS[event.key];
      if (!direction || this.root.hidden || this.expanded || this.drag || this.resizing) return;
      event.preventDefault();
      event.stopPropagation();
      this.resizeWindow(this.resizeGeometry(corner),
        direction.horizontal * KEYBOARD_RESIZE_STEP, direction.vertical * KEYBOARD_RESIZE_STEP);
    }

    /**
     * Capture the window rectangle relative to its containing scene.
     * @param {{horizontal: number, vertical: number}} corner Directions of the moving corner.
     * @returns {object} Resize origin and bounds in CSS pixels.
     */
    resizeGeometry(corner) {
      const bounds = this.root.getBoundingClientRect();
      const container = this.container.getBoundingClientRect();
      return {corner, left: bounds.left - container.left, top: bounds.top - container.top,
        width: bounds.width, height: bounds.height};
    }

    /**
     * Apply a corner displacement while preserving its opposite anchor and size limits.
     * @param {object} geometry Window rectangle and moving corner at the resize origin.
     * @param {number} horizontal Horizontal corner displacement in CSS pixels.
     * @param {number} vertical Vertical corner displacement in CSS pixels.
     */
    resizeWindow(geometry, horizontal, vertical) {
      const container = this.container.getBoundingClientRect();
      const leftward = geometry.corner.horizontal < 0;
      const upward = geometry.corner.vertical < 0;
      const right = geometry.left + geometry.width;
      const bottom = geometry.top + geometry.height;
      const maximumWidth = Math.max(1, (leftward ? right : container.width - geometry.left) - WINDOW_MARGIN);
      const maximumHeight = Math.max(1, (upward ? bottom : container.height - geometry.top) - WINDOW_MARGIN);
      const width = geometry.width + geometry.corner.horizontal * horizontal;
      const height = geometry.height + geometry.corner.vertical * vertical;
      this.size = {
        width: Math.max(Math.min(MINIMUM_SIZE.width, maximumWidth), Math.min(width, maximumWidth)),
        height: Math.max(Math.min(MINIMUM_SIZE.height, maximumHeight), Math.min(height, maximumHeight)),
      };
      this.position = {left: leftward ? right - this.size.width : geometry.left,
        top: upward ? bottom - this.size.height : geometry.top};
      this.fitSize();
      this.fitPosition();
      this.invalidate();
    }

    /**
     * Release resize capture when the gesture or window ends.
     * @param {PointerEvent} [event] Pointer that ended, or omitted to cancel any active resize.
     */
    endResize(event) {
      if (!this.resizing || (event && event.pointerId !== this.resizing.pointer)) return;
      const {handle, pointer} = this.resizing;
      this.resizing = null;
      if (handle.hasPointerCapture(pointer)) handle.releasePointerCapture(pointer);
    }

    /** Fit the chosen size to the current scene without losing its saved dimensions. */
    fitSize() {
      if (!this.size || this.root.hidden || this.expanded) return;
      const container = this.container.getBoundingClientRect();
      const maximumWidth = container.width - WINDOW_MARGIN * 2;
      const maximumHeight = container.height - WINDOW_MARGIN * 2;
      if (maximumWidth <= 0 || maximumHeight <= 0) return;
      this.root.style.width = Math.min(this.size.width, maximumWidth) + 'px';
      this.root.style.height = Math.min(this.size.height, maximumHeight) + 'px';
    }

    // %% camera selection and rendering
    /**
     * Refresh camera sources after model loading, retaining a still available user choice.
     * @param {object[]} models Scene models and their published camera definitions.
     * @param {object} [primary] Robot whose default camera is preferred.
     */
    setModels(models, primary) {
      this.models = models.map(model => ({model, object: model.obj, cameras: model.cameras}));
      this.primary = primary;
      this.sources = global.RobotCamera.sources(models);
      const previous = this.userSelected && this.selected
        ? this.sources.find(source => source.id === this.selected.id) : null;
      this.userSelected = !!previous;
      const preferred = primary ? global.RobotCamera.sources([primary])[0] : null;
      this.selected = previous || (preferred && this.sources.find(source => source.id === preferred.id)) || this.sources[0] || null;
      this.select.replaceChildren();
      for (const source of this.sources) {
        const option = this.root.ownerDocument.createElement('option');
        option.value = source.id;
        option.textContent = source.label;
        this.select.appendChild(option);
      }
      this.select.value = this.selected ? this.selected.id : '';
      this.select.disabled = !this.selected;
      this.status.textContent = 'No robot camera available in this scene.';
      this.status.hidden = !!this.selected;
      this.canvas.hidden = !this.selected;
      this.invalidate();
    }

    /**
     * Resolve changed model, articulated object or camera-list identities before rendering.
     * @param {object[]} models Current scene models, including asynchronous loads.
     * @param {object} [primary] Robot whose default camera is preferred.
     */
    refreshModels(models, primary) {
      const unchanged = this.models.length === models.length && this.models.every((entry, index) => {
        const model = models[index];
        return entry.model === model && entry.object === model.obj && entry.cameras === model.cameras;
      });
      if (!unchanged || this.primary !== primary) this.setModels(models, primary);
    }

    /** Draw the current robot pose without changing the overview camera or GPU context. */
    render() {
      if (this.root.hidden || !this.selected) return;
      const bounds = this.viewport.getBoundingClientRect();
      if (!this.pose.update(this.selected, bounds.width, bounds.height)) return;
      const container = this.container.getBoundingClientRect();
      const left = bounds.left - container.left;
      const bottom = container.bottom - bounds.bottom;
      const height = Math.min(bounds.height, bounds.width / this.pose.camera.aspect);
      const width = height * this.pose.camera.aspect;
      const renderer = this.renderer;
      const pixelRatio = renderer.getPixelRatio();
      // Use one device-pixel rectangle for rendering and copying the image.
      const image = {
        left: Math.round((left + (bounds.width - width) / 2) * pixelRatio) / pixelRatio,
        bottom: Math.round((bottom + (bounds.height - height) / 2) * pixelRatio) / pixelRatio,
        width: Math.round(width * pixelRatio) / pixelRatio,
        height: Math.round(height * pixelRatio) / pixelRatio,
      };
      renderer.getViewport(this.savedViewport);
      renderer.getScissor(this.savedScissor);
      const scissorTest = renderer.getScissorTest();
      try {
        renderer.setViewport(image.left, image.bottom, image.width, image.height);
        renderer.setScissor(image.left, image.bottom, image.width, image.height);
        renderer.setScissorTest(true);
        renderer.clear();
        renderer.render(this.scene, this.pose.camera);
        this.copyViewport(renderer.domElement, pixelRatio, image, bounds.width, bounds.height);
      } finally {
        renderer.setViewport(this.savedViewport);
        renderer.setScissor(this.savedScissor);
        renderer.setScissorTest(scissorTest);
      }
    }

    /**
     * Copy the rendered image into an opaque canvas centered within the window.
     * @param {HTMLCanvasElement} source Shared renderer canvas containing the camera image.
     * @param {number} pixelRatio Device pixels per CSS pixel.
     * @param {{left: number, bottom: number, width: number, height: number}} image Rendered bounds.
     * @param {number} width Available canvas width in CSS pixels.
     * @param {number} height Available canvas height in CSS pixels.
     */
    copyViewport(source, pixelRatio, image, width, height) {
      const pixelWidth = Math.round(width * pixelRatio);
      const pixelHeight = Math.round(height * pixelRatio);
      const imageWidth = Math.round(image.width * pixelRatio);
      const imageHeight = Math.round(image.height * pixelRatio);
      if (this.canvas.width !== pixelWidth) this.canvas.width = pixelWidth;
      if (this.canvas.height !== pixelHeight) this.canvas.height = pixelHeight;
      this.context.fillStyle = '#000000';
      this.context.fillRect(0, 0, pixelWidth, pixelHeight);
      this.context.drawImage(source,
        Math.round(image.left * pixelRatio), source.height - Math.round(image.bottom * pixelRatio) - imageHeight,
        imageWidth, imageHeight, Math.floor((pixelWidth - imageWidth) / 2),
        Math.floor((pixelHeight - imageHeight) / 2), imageWidth, imageHeight);
    }

    /** Release observers and input handlers when the scene is unmounted. */
    destroy() {
      this.endDrag();
      this.endResize();
      this.observer.disconnect();
      this.listeners.forEach(({element, name, handler}) => element.removeEventListener(name, handler));
      this.listeners = [];
      this.resizeHandles.forEach(handle => handle.remove());
      this.resizeHandles = [];
      this.canvas.remove();
      this.root.hidden = true;
      this.layer.checked = false;
    }
  }

  global.RobotCameraPanel = RobotCameraPanel;
})(window);
