/* %% viewpoints from native camera annotations */
/**
 * Camera properties published by the scene bundle.
 * @typedef {object} CameraDefinition
 * @property {string} name Native sensor name.
 * @property {string} link Exact exported root-link name.
 * @property {number[]} forward Viewing direction in root-local coordinates.
 * @property {number} horizontalAngle Horizontal field of view in radians.
 * @property {number} verticalAngle Vertical field of view in radians.
 * @property {boolean} default Whether the native sensor is a default camera.
 */

/**
 * A scene model and its native camera annotations.
 * @typedef {object} CameraModel
 * @property {string} name Bundle model identity.
 * @property {boolean} robot Whether the model represents a robot.
 * @property {(THREE.Object3D & {links: Object<string, THREE.Object3D>})|null} obj Loaded model.
 * @property {CameraDefinition[]} [cameras] Camera metadata, absent in older bundles.
 */
(function (global) {
  'use strict';

  /** Viewer clipping distances in meters and tolerance for choosing display up. */
  const PROJECTION = Object.freeze({near: 0.02, far: 200, parallelTolerance: 1e-6});

  /** A native camera description bound to its loaded Three.js link. */
  class CameraSource {
    /**
     * Bind a published camera to its loaded model.
     * @param {CameraModel} model Loaded scene model with its published camera descriptions.
     * @param {CameraDefinition} definition Native camera name, root, direction and view angles.
     */
    constructor(model, definition) {
      /** Stable identity across asynchronous reloads of the same model. */
      this.id = encodeURIComponent(model.name) + '/' + encodeURIComponent(definition.name);
      /** Display label preserving the native camera name. */
      this.label = model.name + ' · ' + definition.name;
      /** Loaded scene node whose articulation determines the camera pose. */
      this.link = model.obj.links[definition.link];
      /** Published native camera parameters, retained without inferred replacements. */
      this.definition = definition;
    }

    /**
     * Check whether a bundle entry identifies a usable perspective camera.
     * @param {unknown} definition Camera metadata read from the scene bundle.
     * @returns {definition is CameraDefinition} Whether identity, projection and direction are valid.
     */
    static isValidDefinition(definition) {
      if (!definition || typeof definition !== 'object') return false;
      const camera = /** @type {Record<string, unknown>} */ (definition);
      if (typeof camera.name !== 'string' || typeof camera.link !== 'string' ||
          typeof camera.default !== 'boolean') return false;
      const angles = [camera.horizontalAngle, camera.verticalAngle];
      return angles.every(angle => typeof angle === 'number' && Number.isFinite(angle) && angle > 0 && angle < Math.PI) &&
        Array.isArray(camera.forward) && camera.forward.length === 3 &&
        camera.forward.every(Number.isFinite) && camera.forward.some(value => value !== 0);
    }

    /**
     * Bind annotated cameras to exact model links in a stable order, defaults first.
     * @param {CameraModel[]} models Scene models, including models still loading.
     * @returns {CameraSource[]} Renderable native camera viewpoints.
     */
    static forModels(models) {
      return models.filter(model => model.robot && model.obj && model.obj.links && Array.isArray(model.cameras))
        .sort((first, second) => first.name.localeCompare(second.name))
        .flatMap(model => model.cameras
          .filter(definition => CameraSource.isValidDefinition(definition) && model.obj.links[definition.link])
          // Native default selection uses the first flagged sensor in robot order.
          .sort((first, second) => Number(second.default) - Number(first.default))
          .map(definition => new CameraSource(model, definition)));
    }
  }

  // %% articulated camera pose
  /** A perspective view following a native camera's direction and field of view. */
  class CameraPose {
    /**
     * Create the camera and reusable articulation vectors.
     * @param {typeof THREE} three The viewer's Three.js namespace.
     */
    constructor(three) {
      /** Shared-renderer camera whose projection comes from the selected source. */
      this.camera = new three.PerspectiveCamera();
      this.camera.near = PROJECTION.near;
      this.camera.far = PROJECTION.far;
      this.camera.updateProjectionMatrix();
      /** Selected link's current world orientation. */
      this.orientation = new three.Quaternion();
      /** Normalized viewing direction in the rendered world. */
      this.forward = new three.Vector3();
      /** World point one unit ahead of the selected camera. */
      this.target = new three.Vector3();
    }

    /**
     * Follow articulation without changing the selected camera's field of view on resize.
     * Native camera annotations specify forward but no image roll. Display up uses
     * local positive Z, or local negative Y when forward is parallel to Z.
     * @param {CameraSource|null} source Selected native camera bound to its scene link.
     * @param {number} width Available viewport width in CSS pixels.
     * @param {number} height Available viewport height in CSS pixels.
     * @returns {boolean} Whether the viewpoint is ready to render.
     */
    update(source, width, height) {
      if (!source || !Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return false;
      const definition = source.definition;
      const fieldOfView = definition.verticalAngle * 180 / Math.PI;
      const aspect = Math.tan(definition.horizontalAngle / 2) / Math.tan(definition.verticalAngle / 2);
      if (this.camera.fov !== fieldOfView || this.camera.aspect !== aspect) {
        this.camera.fov = fieldOfView;
        this.camera.aspect = aspect;
        this.camera.updateProjectionMatrix();
      }
      source.link.updateWorldMatrix(true, false);
      source.link.getWorldPosition(this.camera.position);
      source.link.getWorldQuaternion(this.orientation);
      this.forward.fromArray(definition.forward).normalize();
      this.camera.up.set(0, 0, 1);
      if (Math.abs(this.forward.dot(this.camera.up)) > 1 - PROJECTION.parallelTolerance) {
        this.camera.up.set(0, -1, 0);
      }
      this.forward.applyQuaternion(this.orientation);
      this.camera.up.applyQuaternion(this.orientation);
      this.target.copy(this.camera.position).add(this.forward);
      this.camera.lookAt(this.target);
      this.camera.updateMatrixWorld(true);
      return true;
    }
  }

  global.RobotCamera = Object.freeze({sources: CameraSource.forModels, Pose: CameraPose});
})(window);
