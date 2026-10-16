/* %% robot camera window controls */
(function (global) {
  'use strict';

  /** Distance in CSS pixels moved by one arrow-key press on a corner control. */
  const KEYBOARD_RESIZE_STEP = 10;
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

  /**
   * One moving corner and the signs of its outward movement.
   * @typedef {{name: string, horizontal: number, vertical: number}} ResizeCorner
   */
  /**
   * Window bounds relative to the scene when a resize begins.
   * @typedef {{corner: ResizeCorner, left: number, top: number, width: number, height: number}} ResizeOrigin
   */

  /** Movable, resizable window controls bounded by the robot scene. */
  class RobotCameraWindow {
    /**
     * Mount window input and observe its available space.
     * @param {object} options Window controls and scene layout dependencies.
     * @param {HTMLElement} options.root Window containing the camera controls.
     * @param {HTMLElement} options.viewport Image area whose layout affects window bounds.
     * @param {HTMLInputElement} options.layer Checkbox controlling visibility.
     * @param {HTMLElement} options.container Scene area bounding window movement.
     * @param {function(): void} options.invalidate Request another scene frame.
     * @param {function(): void} options.restoreFocus Focus a visible scene control after closing.
     * @param {typeof ResizeObserver} options.ResizeObserver Browser resize observer constructor.
     */
    constructor(options) {
      /** Window containing the camera controls and image. @type {HTMLElement} */
      this.root = options.root;
      /** Image area observed when the window layout changes. @type {HTMLElement} */
      this.viewport = options.viewport;
      /** Scene area that bounds window movement. @type {HTMLElement} */
      this.container = options.container;
      /** Request another frame after an input or layout change. @type {function(): void} */
      this.invalidate = options.invalidate;
      /** Scene-owned action returning focus to a visible control. @type {function(): void} */
      this.restoreFocus = options.restoreFocus;
      /** Visibility checkbox in the scene controls. @type {HTMLInputElement} */
      this.layer = options.layer;
      const style = global.getComputedStyle(this.root);
      /** Space retained between the window and scene edges, in CSS pixels. @type {number} */
      this.margin = parseFloat(style.getPropertyValue('--robot-camera-margin'));
      /** Minimum inset dimensions specified by the stylesheet. @type {{width: number, height: number}} */
      this.minimumSize = {
        width: parseFloat(style.getPropertyValue('--robot-camera-minimum-width')),
        height: parseFloat(style.getPropertyValue('--robot-camera-minimum-height')),
      };
      /** Whether the window uses the expanded scene layout. @type {boolean} */
      this.expanded = false;
      /** Last inset position relative to the scene. @type {{left: number, top: number}|null} */
      this.position = null;
      /** Captured movement pointer and window offset. @type {{pointer: number, offsetX: number, offsetY: number}|null} */
      this.drag = null;
      /** Chosen inset size, retained while the scene is smaller. @type {{width: number, height: number}|null} */
      this.size = null;
      /** Captured resize origin. @type {(ResizeOrigin & {pointer: number, handle: HTMLButtonElement, clientX: number, clientY: number})|null} */
      this.resizing = null;
      /** Owned event subscriptions. @type {{element: EventTarget, name: string, handler: EventListener}[]} */
      this.listeners = [];
      /** Title bar accepting window movement gestures. @type {HTMLElement} */
      this.header = this.root.querySelector('.robot-camera-head');
      this.header.title = 'Drag to move the robot view';
      /** Button toggling inset and expanded sizes. @type {HTMLButtonElement} */
      this.expand = this.root.querySelector('[data-robot-camera="expand"]');
      /** Corner controls owned by the window. @type {HTMLButtonElement[]} */
      this.resizeHandles = [];
      this.bindControls();
      this.addResizeHandles();
      /** Observer keeping the window inside a resized scene. @type {ResizeObserver} */
      this.observer = new options.ResizeObserver(() => {
        this.fitSize();
        this.fitPosition();
        this.invalidate();
      });
      this.observer.observe(this.viewport);
      this.observer.observe(this.container);
      this.setVisible(false);
      this.setExpanded(false);
    }

    /** Bind visibility, expansion and movement to the window controls. */
    bindControls() {
      this.listen(this.layer, 'change', () => this.setVisible(this.layer.checked));
      this.listen(this.expand, 'click', () => this.setExpanded(!this.expanded));
      this.listen(this.root.querySelector('[data-robot-camera="close"]'), 'click', () => this.close());
      this.listen(this.viewport, 'dblclick', () => this.setExpanded(!this.expanded));
      this.listen(this.root, 'keydown', event => {
        if (event.key !== 'Escape') return;
        event.stopPropagation();
        if (this.expanded) this.setExpanded(false);
        else this.close();
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
     * Track subscriptions for removal when the window is destroyed.
     * @param {EventTarget} element Control receiving the event.
     * @param {string} name Browser event name.
     * @param {EventListener} handler Window action for that event.
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

    /** Hide the view and return keyboard focus to its scene controls. @returns {void} */
    close() {
      this.setVisible(false);
      this.restoreFocus();
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
      const horizontalMargin = Math.min(this.margin, Math.max(0, (container.width - bounds.width) / 2));
      const verticalMargin = Math.min(this.margin, Math.max(0, (container.height - bounds.height) / 2));
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
     * @param {ResizeCorner} corner Resize directions.
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
     * @param {ResizeCorner} corner Resize directions.
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
     * @param {ResizeCorner} corner Directions of the moving corner.
     * @returns {ResizeOrigin} Resize origin and bounds in CSS pixels.
     */
    resizeGeometry(corner) {
      const bounds = this.root.getBoundingClientRect();
      const container = this.container.getBoundingClientRect();
      return {corner, left: bounds.left - container.left, top: bounds.top - container.top,
        width: bounds.width, height: bounds.height};
    }

    /**
     * Apply a corner displacement while preserving its opposite anchor and size limits.
     * @param {ResizeOrigin} geometry Window rectangle and moving corner at the resize origin.
     * @param {number} horizontal Horizontal corner displacement in CSS pixels.
     * @param {number} vertical Vertical corner displacement in CSS pixels.
     */
    resizeWindow(geometry, horizontal, vertical) {
      const container = this.container.getBoundingClientRect();
      const leftward = geometry.corner.horizontal < 0;
      const upward = geometry.corner.vertical < 0;
      const right = geometry.left + geometry.width;
      const bottom = geometry.top + geometry.height;
      const maximumWidth = Math.max(1, (leftward ? right : container.width - geometry.left) - this.margin);
      const maximumHeight = Math.max(1, (upward ? bottom : container.height - geometry.top) - this.margin);
      const width = geometry.width + geometry.corner.horizontal * horizontal;
      const height = geometry.height + geometry.corner.vertical * vertical;
      this.size = {
        width: Math.max(Math.min(this.minimumSize.width, maximumWidth), Math.min(width, maximumWidth)),
        height: Math.max(Math.min(this.minimumSize.height, maximumHeight), Math.min(height, maximumHeight)),
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
      const maximumWidth = container.width - this.margin * 2;
      const maximumHeight = container.height - this.margin * 2;
      if (maximumWidth <= 0 || maximumHeight <= 0) return;
      this.root.style.width = Math.min(this.size.width, maximumWidth) + 'px';
      this.root.style.height = Math.min(this.size.height, maximumHeight) + 'px';
    }

    /** Release pointer capture, observation and owned window controls. */
    destroy() {
      this.endDrag();
      this.endResize();
      this.observer.disconnect();
      this.listeners.forEach(({element, name, handler}) => element.removeEventListener(name, handler));
      this.listeners = [];
      this.resizeHandles.forEach(handle => handle.remove());
      this.resizeHandles = [];
      this.root.hidden = true;
      this.layer.checked = false;
    }
  }

  global.RobotCameraWindow = RobotCameraWindow;
})(window);
