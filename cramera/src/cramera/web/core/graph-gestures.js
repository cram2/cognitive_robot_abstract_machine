/* ============================================================================
 * core/graph-gestures.js — wheel and touchpad input for a vis-network view.
 *
 * vis-network's built-in handler zooms a flat ±10% per wheel *event* and looks at
 * nothing but the sign of deltaY: not the distance scrolled, not deltaMode, not
 * ctrlKey, not deltaX. A mouse wheel sends one event per notch and gets away with
 * that; a touchpad sends dozens of small events per swipe, each taken as a full
 * notch, so a single two-finger flick multiplies the zoom many times over and the
 * graph disappears. Set `interaction.zoomView: false` and install this instead.
 *
 * It reads the gesture the way a map does:
 *   · two-finger scroll  → pan, by the distance actually scrolled
 *   · pinch (a ctrl-held wheel, which is how browsers report it) → zoom
 *   · mouse wheel (stepped units, or whole pixel notches) → zoom
 * Zooming is anchored under the pointer and bounded, so no gesture can throw the
 * graph off screen.
 *
 * Only the four public view calls of vis-network are used — getScale,
 * getViewPosition, DOMtoCanvas and moveTo — so nothing here depends on internals.
 * ==========================================================================*/
(function (global) {
  'use strict';

  const ZOOM = 'zoom';
  const PAN = 'pan';

  const MIN_SCALE = 0.05;
  /* How far out a gesture may zoom. */

  const MAX_SCALE = 8;
  /* How far in a gesture may zoom. */

  const ZOOM_PER_PIXEL = 0.0028;
  /* Scale change per pixel of wheel travel: a 100px mouse notch is about a third
     in or out, a few pixels of pinch about a percent. */

  const LINE_PIXELS = 16;
  /* What one deltaMode=DOM_DELTA_LINE unit is worth in pixels. */

  const PAGE_PIXELS = 400;
  /* What one deltaMode=DOM_DELTA_PAGE unit is worth in pixels. */

  const MOUSE_NOTCH_PIXELS = 100;
  /* The pixel delta a mouse wheel notch reports when the browser reports pixels.
     Touchpads report small, uneven deltas instead, which is what tells them apart. */

  function isMouseNotch(wheel) {
    const distance = Math.abs(wheel.deltaY);
    return wheel.deltaX === 0 && distance >= MOUSE_NOTCH_PIXELS
      && distance % MOUSE_NOTCH_PIXELS === 0;
  }

  /* The distance a wheel event travelled, in css pixels. */
  function pixelsOf(wheel) {
    const unit = wheel.deltaMode === 1 ? LINE_PIXELS : wheel.deltaMode === 2 ? PAGE_PIXELS : 1;
    return { x: wheel.deltaX * unit, y: wheel.deltaY * unit };
  }

  /* What the user meant by a wheel event. Panning is the default: zooming has to be
     asked for, by a pinch or by a wheel that reports itself as stepped. */
  function intentOf(wheel) {
    if (wheel.ctrlKey || wheel.metaKey) return ZOOM;
    if (wheel.deltaMode !== 0) return ZOOM;
    if (wheel.deltaX !== 0) return PAN;
    return isMouseNotch(wheel) ? ZOOM : PAN;
  }

  /* Keep a scale within the bounds, in the direction it is travelling. A view fitted
     out past the floor stays where it is instead of being pulled back in. */
  function bounded(scale, next) {
    if (next < scale) return Math.max(next, Math.min(scale, MIN_SCALE));
    return Math.min(next, Math.max(scale, MAX_SCALE));
  }

  /* The scale a wheel event leaves behind, growing with the distance scrolled. */
  function scaleAfter(scale, wheel) {
    return bounded(scale, scale * Math.exp(-pixelsOf(wheel).y * ZOOM_PER_PIXEL));
  }

  /* Move the view so `graphPoint` sits under `domPoint` again after a scale change. */
  function anchor(network, scale, domPoint, graphPoint) {
    network.moveTo({ scale: scale, animation: false });
    const moved = network.DOMtoCanvas(domPoint);
    const centre = network.getViewPosition();
    network.moveTo({
      scale: scale,
      position: { x: centre.x - (moved.x - graphPoint.x), y: centre.y - (moved.y - graphPoint.y) },
      animation: false,
    });
  }

  function zoomAt(network, container, wheel) {
    const box = container.getBoundingClientRect();
    const domPoint = { x: wheel.clientX - box.left, y: wheel.clientY - box.top };
    anchor(network, scaleAfter(network.getScale(), wheel), domPoint, network.DOMtoCanvas(domPoint));
  }

  function panBy(network, wheel) {
    const step = pixelsOf(wheel);
    const scale = network.getScale();
    const centre = network.getViewPosition();
    network.moveTo({
      position: { x: centre.x + step.x / scale, y: centre.y + step.y / scale },
      animation: false,
    });
  }

  global.GraphGestures = {
    ZOOM: ZOOM,
    PAN: PAN,
    MIN_SCALE: MIN_SCALE,
    MAX_SCALE: MAX_SCALE,
    LINE_PIXELS: LINE_PIXELS,
    PAGE_PIXELS: PAGE_PIXELS,
    intentOf: intentOf,
    pixelsOf: pixelsOf,
    scaleAfter: scaleAfter,

    /* Step the scale about the centre of the view, for an on-screen zoom control. */
    zoomBy: function (network, factor) {
      network.moveTo({ scale: bounded(network.getScale(), network.getScale() * factor),
                       position: network.getViewPosition(), animation: false });
    },

    /* Take wheel input on `container` over from vis-network. Returns the handle that
       removes the listener again. */
    install: function (network, container) {
      function onWheel(wheel) {
        wheel.preventDefault();
        if (intentOf(wheel) === ZOOM) zoomAt(network, container, wheel);
        else panBy(network, wheel);
      }
      container.addEventListener('wheel', onWheel, { passive: false });
      return { destroy: function () { container.removeEventListener('wheel', onWheel); } };
    },
  };
})(window);
