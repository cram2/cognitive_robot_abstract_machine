/* =============================================================================
 * highlight_arrow — where the arrow bouncing over a highlighted object sits.
 *
 * A highlighted object's emissive glow is easy to miss among similar colours, so
 * the scene panel also hangs an arrow over it, pointing down at it and bobbing.
 * The geometry of that arrow — its size, where it rests above the object, and how
 * far it has bobbed at a given moment — is pure math and lives here; the panel
 * only builds a mesh out of it.
 * ==========================================================================*/
(function () {
  'use strict';

  const CLEARANCE = 0.05;          // gap between the object's top and the arrow tip, m
  const HEIGHT = 0.13;             // the arrow cone's height, m
  const RADIUS = 0.045;            // the arrow cone's base radius, m
  const BOB_AMPLITUDE = 0.04;      // how far the arrow lifts off its rest, m
  const BOB_PERIOD_SECONDS = 1.1;  // one bounce
  const COLOR = '#39d5c8';         // the highlight teal the rest of the UI uses

  // the arrow's resting centre above an object whose top is at `top`, so its
  // downward tip ends `CLEARANCE` short of the object
  function restAltitude(top) {
    return top + CLEARANCE + HEIGHT / 2;
  }

  // how far above its rest the arrow has bobbed `seconds` into the animation:
  // an eased lift-and-return, at rest once per period
  function bobOffset(seconds) {
    const phase = (seconds % BOB_PERIOD_SECONDS) / BOB_PERIOD_SECONDS;
    return BOB_AMPLITUDE * 0.5 * (1 - Math.cos(phase * 2 * Math.PI));
  }

  window.HighlightArrow = {
    CLEARANCE: CLEARANCE,
    HEIGHT: HEIGHT,
    RADIUS: RADIUS,
    BOB_AMPLITUDE: BOB_AMPLITUDE,
    BOB_PERIOD_SECONDS: BOB_PERIOD_SECONDS,
    COLOR: COLOR,
    restAltitude: restAltitude,
    bobOffset: bobOffset,
  };
})();
