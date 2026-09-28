/* ============================================================================
 * core/collada-mesh.js — undoes THREE.ColladaLoader's automatic up-axis
 * correction for meshes referenced from a URDF.
 *
 * ColladaLoader rotates any <up_axis>Z_UP</up_axis> asset's root scene -90°
 * about X so it renders correctly on its own in three.js's Y-up world. A
 * URDF-referenced mesh must stay in its raw, un-rotated frame instead: the
 * URDF's own <origin> already places it in the model's Z-up frame, and the
 * whole assembled scene gets one single Z-up -> Y-up correction at the world
 * root. Left uncorrected, a Z_UP-tagged mesh (e.g. Gazebo-exported warehouse
 * assets) gets that rotation applied twice.
 * ==========================================================================*/
(function () {
  'use strict';

  function neutralizeUpAxisRotation(scene) {
    scene.quaternion.identity();
    return scene;
  }

  window.ColladaMeshUtil = { neutralizeUpAxisRotation: neutralizeUpAxisRotation };
})();
