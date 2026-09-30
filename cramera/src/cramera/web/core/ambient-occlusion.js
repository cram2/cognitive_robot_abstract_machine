/* ============================================================================
 * core/ambient-occlusion.js — the SSAO pass the 3D scene panel renders through.
 *
 * three's SSAOPass fills its depth and normal buffers by rendering the scene with an
 * override material. A textured scene background is a 2x2 plane at the origin in the
 * render list, so under that override it is drawn as real geometry, and its occlusion
 * edges appear as a rectangle standing on the floor of every scene. This pass hides
 * the background for exactly those renders; the beauty render keeps it.
 *
 * Defined only when three's SSAOPass is on the page, like the pass it extends.
 * ==========================================================================*/
(function () {
  'use strict';
  if (typeof THREE === 'undefined' || !THREE.SSAOPass) return;

  class BackgroundIgnoringSSAOPass extends THREE.SSAOPass {
    renderOverride(renderer, overrideMaterial, renderTarget, clearColor, clearAlpha) {
      const background = this.scene.background;
      this.scene.background = null;
      super.renderOverride(renderer, overrideMaterial, renderTarget, clearColor, clearAlpha);
      this.scene.background = background;
    }
  }

  window.BackgroundIgnoringSSAOPass = BackgroundIgnoringSSAOPass;
})();
