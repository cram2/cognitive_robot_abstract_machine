/* ============================================================================
 * core/shape-specs.js — how a live overlay body's published shapes become renderable.
 *
 * The bridge publishes any world body shape by shape (box/cylinder/sphere/mesh), each
 * with its own local pose, colour and dimensions, so the viewer can rebuild the body
 * without knowing how the world was constructed — the way RViz renders whatever the
 * world contains. This module holds the pure mapping from one published shape entry to
 * the build instruction the 3D code executes, so the geometry decisions (which
 * primitive, which axis convention, which URL) are testable without THREE.
 * ==========================================================================*/
(function (global) {
  'use strict';

  const FALLBACK_COLOR = '#cccccc';
  /* Colour used when neither the shape nor its object carries one. */

  const FALLBACK_SIZE = [0.06, 0.06, 0.12];
  /* Box extent used when a shape's dimensions are unusable. */

  /* One published shape as a build instruction:
     {type, position, quaternion, color, opacity, and per type:
      'box' → size; 'cylinder' → radius, height, rotateXDegrees (the world's cylinders
      run along Z while three.js cylinders run along Y); 'sphere' → radius;
      'mesh' → url, mtl (companion material URL or null), format, scale}. */
  function buildSpec(shape, objectColor, liveBase) {
    const common = {
      position: shape.position || [0, 0, 0],
      quaternion: shape.quaternion || [0, 0, 0, 1],
      color: shape.color || objectColor || FALLBACK_COLOR,
      opacity: typeof shape.opacity === 'number' ? shape.opacity : 1,
    };
    if (shape.kind === 'cylinder' && shape.radius > 0 && shape.height > 0) {
      common.type = 'cylinder';
      common.radius = shape.radius;
      common.height = shape.height;
      common.rotateXDegrees = 90;
      return common;
    }
    if (shape.kind === 'sphere' && shape.radius > 0) {
      common.type = 'sphere';
      common.radius = shape.radius;
      return common;
    }
    if (shape.kind === 'mesh' && shape.mesh) {
      common.type = 'mesh';
      common.url = (liveBase || '') + shape.mesh;
      common.mtl = shape.mtl ? (liveBase || '') + shape.mtl : null;
      common.format = (shape.format || '').toLowerCase();
      common.scale = shape.scale || [1, 1, 1];
      return common;
    }
    common.type = 'box';
    common.size = (shape.size && shape.size.length === 3) ? shape.size : FALLBACK_SIZE.slice();
    return common;
  }

  global.ShapeSpecs = {
    FALLBACK_COLOR: FALLBACK_COLOR,
    FALLBACK_SIZE: FALLBACK_SIZE,
    buildSpec: buildSpec,

    /* Every build instruction for one published object's shape list. */
    buildSpecs: function (shapes, objectColor, liveBase) {
      return (shapes || []).map(function (shape) {
        return buildSpec(shape, objectColor, liveBase);
      });
    },
  };
})(window);
