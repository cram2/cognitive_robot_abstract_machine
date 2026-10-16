/* %% robot camera selection and panel lifecycle */
(function (global) {
  'use strict';

  /** Camera selection coordinating a movable window and its rendered image. */
  class RobotCameraPanel {
    /**
     * Coordinate camera selection, window controls and the rendered image.
     *
     * @param {object} options Panel elements, scene resources and browser dependencies.
     * @param {typeof THREE} options.THREE Three.js constructors for the camera and viewport state.
     * @param {HTMLElement} options.root Window containing the robot camera controls.
     * @param {HTMLInputElement} options.layer Checkbox controlling visibility.
     * @param {THREE.WebGLRenderer} options.renderer Scene renderer borrowed for camera images.
     * @param {THREE.Scene} options.scene Articulated scene to render.
     * @param {HTMLElement} options.container Scene area bounding window movement.
     * @param {function(): void} options.invalidate Request another scene frame.
     * @param {function(): void} options.restoreFocus Focus a visible scene control after closing.
     * @param {typeof ResizeObserver} options.ResizeObserver Browser resize observer constructor.
     */
    constructor(options) {
      /** Window containing the camera controls and image. @type {HTMLElement} */
      this.root = options.root;
      /** Request another frame after changing the selected camera. @type {function(): void} */
      this.invalidate = options.invalidate;
      /** Available frames from the scene. @type {ReturnType<typeof window.RobotCamera.sources>} */
      this.sources = [];
      /** Rendered camera source. @type {ReturnType<typeof window.RobotCamera.sources>[number]|null} */
      this.selected = null;
      /** Whether a user choice takes precedence over the primary robot. @type {boolean} */
      this.userSelected = false;
      /** Message shown when no cameras are available. @type {HTMLElement} */
      this.status = this.root.querySelector('[data-robot-camera="status"]');
      /** Native selector for the scene's cameras. @type {HTMLSelectElement} */
      this.select = this.root.querySelector('[data-robot-camera="source"]');
      const viewport = this.root.querySelector('[data-robot-camera="viewport"]');
      /** Owned image using the scene's shared renderer. @type {InstanceType<typeof window.RobotCameraImage>} */
      this.image = new global.RobotCameraImage({
        THREE: options.THREE, renderer: options.renderer, scene: options.scene,
        container: options.container, viewport,
      });
      /** Owned window controls and layout. @type {InstanceType<typeof window.RobotCameraWindow>} */
      this.window = new global.RobotCameraWindow({
        root: this.root, layer: options.layer, container: options.container, viewport,
        invalidate: this.invalidate, restoreFocus: options.restoreFocus, ResizeObserver: options.ResizeObserver,
      });
      /** Camera-choice subscription removed with the panel. @type {EventListener} */
      this.onSelectionChange = () => {
        this.selected = this.sources.find(source => source.id === this.select.value) || null;
        this.userSelected = true;
        this.invalidate();
      };
      this.select.addEventListener('change', this.onSelectionChange);
      this.setModels([]);
    }

    // %% camera selection
    /**
     * Refresh camera sources after model loading, retaining a still available user choice.
     * @param {CameraModel[]} models Scene models and their published camera definitions.
     * @param {CameraModel} [primary] Robot whose default camera is preferred.
     */
    setModels(models, primary) {
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
      this.image.canvas.hidden = !this.selected;
      this.invalidate();
    }

    /** Draw the selected robot view when its window is visible. */
    render() {
      if (this.root.hidden || !this.selected) return;
      this.image.render(this.selected);
    }

    /** Release camera selection, window controls and the owned image. */
    destroy() {
      this.select.removeEventListener('change', this.onSelectionChange);
      this.window.destroy();
      this.image.destroy();
    }
  }

  global.RobotCameraPanel = RobotCameraPanel;
})(window);
