/* %% robot camera image from the shared scene renderer */
(function (global) {
  'use strict';

  /** Camera image copied from a borrowed renderer into an owned canvas. */
  class RobotCameraImage {
    /**
     * Mount the image and allocate camera and renderer-state vectors.
     * @param {object} options Scene resources and image destination.
     * @param {typeof THREE} options.THREE Three.js constructors for camera and renderer state.
     * @param {THREE.WebGLRenderer} options.renderer Scene renderer borrowed for camera images.
     * @param {THREE.Scene} options.scene Articulated scene to render.
     * @param {HTMLElement} options.container Scene area defining renderer coordinates.
     * @param {HTMLElement} options.viewport Image area containing the copied camera image.
     */
    constructor(options) {
      /** Shared scene renderer, retained without taking ownership. @type {THREE.WebGLRenderer} */
      this.renderer = options.renderer;
      /** Articulated world viewed by the camera. @type {THREE.Scene} */
      this.scene = options.scene;
      /** Scene area defining viewport coordinates. @type {HTMLElement} */
      this.container = options.container;
      /** Image area determining the camera image size. @type {HTMLElement} */
      this.viewport = options.viewport;
      /** Articulated perspective camera. @type {InstanceType<typeof window.RobotCamera.Pose>} */
      this.pose = new global.RobotCamera.Pose(options.THREE);
      /** Owned image copy covering the robot viewport. @type {HTMLCanvasElement} */
      this.canvas = this.viewport.ownerDocument.createElement('canvas');
      this.canvas.setAttribute('aria-hidden', 'true');
      /** Opaque drawing context for the robot view. @type {CanvasRenderingContext2D} */
      this.context = this.canvas.getContext('2d', {alpha: false});
      this.viewport.appendChild(this.canvas);
      /** Viewport restored after a camera image. @type {THREE.Vector4} */
      this.savedViewport = new options.THREE.Vector4();
      /** Scissor bounds restored after a camera image. @type {THREE.Vector4} */
      this.savedScissor = new options.THREE.Vector4();
    }

    /**
     * Draw the selected camera without changing the overview camera or GPU context.
     * @param {ReturnType<typeof window.RobotCamera.sources>[number]} source Camera bound to its articulated scene link.
     */
    render(source) {
      const bounds = this.viewport.getBoundingClientRect();
      if (!this.pose.update(source, bounds.width, bounds.height)) return;
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

    /** Remove the owned image without disposing the scene's renderer. */
    destroy() {
      this.canvas.remove();
    }
  }

  global.RobotCameraImage = RobotCameraImage;
})(window);
