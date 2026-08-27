/* ============================================================================
 * config.js — *which* panels are shown *where*. This is the file you edit to swap
 * a visualization: remove an id, add your own (define it via Panels.define in
 * a new panels/<name>/panel.js and include that script in index.html).
 *
 * Slots are the data-slot elements in index.html ('left', 'right'); a slot
 * with several panel ids stacks them vertically.
 * ==========================================================================*/
window.CRAMERA_CONFIG = {
  layout: {
    left: ['robot-scene'],
    right: ['eql', 'graph'],
  },
};
