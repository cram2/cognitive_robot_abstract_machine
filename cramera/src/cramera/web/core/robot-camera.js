/* %% viewpoints from native camera annotations */
(function (global) {
  'use strict';

  const PROJECTION = Object.freeze({near: 0.02, far: 200, parallelTolerance: 1e-6});
  /** Viewer clipping distances in meters and tolerance for choosing display up. */

  /** A native camera description bound to its loaded Three.js link. */
  class CameraSource {
    /**
     * @param {Object} model Loaded scene model with its published camera descriptions.
     * @param {Object} definition Native camera name, root, direction and view angles.
     */
    constructor(model, definition) {
      this.id = encodeURIComponent(model.name) + '/' + encodeURIComponent(definition.name);
      /** Stable identity across asynchronous reloads of the same model. */
      this.label = model.name + ' · ' + definition.name;
      /** Display label preserving the native camera name. */
      this.link = model.obj.links[definition.link];
      /** Loaded scene node whose articulation determines the camera pose. */
      this.definition = definition;
      /** Published native camera parameters, retained without inferred replacements. */
    }

    /**
     * Whether published projection and direction values can describe a perspective view.
     * @param {Object} definition Camera metadata read from the scene bundle.
     * @returns {boolean} Whether the numeric projection is finite and nondegenerate.
     */
    static hasProjection(definition) {
      const angles = [definition.horizontalAngle, definition.verticalAngle];
      return angles.every(angle => Number.isFinite(angle) && angle > 0 && angle < Math.PI) &&
        Array.isArray(definition.forward) && definition.forward.length === 3 &&
        definition.forward.every(Number.isFinite) && definition.forward.some(value => value !== 0);
    }

    /**
     * Bind annotated cameras to exact model links in a stable order, defaults first.
     * @param {Object[]} models Scene models, including models still loading.
     * @returns {CameraSource[]} Renderable native camera viewpoints.
     */
    static forModels(models) {
      return models.filter(model => model.robot && model.obj && model.obj.links)
        .slice().sort((first, second) => first.name.localeCompare(second.name))
        .flatMap(model => (model.cameras || [])
          .filter(definition => model.obj.links[definition.link] && CameraSource.hasProjection(definition))
          .slice().sort((first, second) => Number(second.default) - Number(first.default) ||
            first.name.localeCompare(second.name) || first.link.localeCompare(second.link))
          .map(definition => new CameraSource(model, definition)));
    }
  }

  // %% articulated camera pose
  /** A perspective view following a native camera's direction and field of view. */
  class CameraPose {
    /** @param {Object} three The viewer's Three.js namespace. */
    constructor(three) {
      this.camera = new three.PerspectiveCamera();
      /** Shared-renderer camera whose projection comes from the selected source. */
      this.camera.near = PROJECTION.near;
      this.camera.far = PROJECTION.far;
      this.camera.updateProjectionMatrix();
      this.orientation = new three.Quaternion();
      /** Selected link's current world orientation. */
      this.forward = new three.Vector3();
      /** Normalized viewing direction in the rendered world. */
      this.target = new three.Vector3();
      /** World point one unit ahead of the selected camera. */
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
