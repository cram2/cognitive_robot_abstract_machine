/* ============================================================================
 * core/split-sizing.js — the pane geometry behind the draggable dividers.
 *
 * Both dividers the shell installs answer the same question in a different
 * direction: given a container of `total` px and a pointer `offset` px from its
 * start, how much of it does the pane after the divider get? Keeping that here,
 * as arithmetic on plain numbers, keeps core/split-resize.js down to the DOM
 * wiring.
 *
 * Sizes are fractions of the container (0…1) rather than pixels, so a pane keeps
 * its share of the window when the window is resized.
 * ==========================================================================*/
(function (global) {
  'use strict';

  const MIN_PANE_PIXELS = 150;
  /* No drag may shrink a pane below this. It is the min-height app.css gives a
     stacked panel, so a pane never overflows the grid track it sits in. */

  const EVEN = 0.5;
  /* The share both panes get when no size is remembered and when a container is
     too small to honour the minimum twice. */

  function smallestPane(total) {
    return Math.min(MIN_PANE_PIXELS, total / 2);
  }

  function clampPane(total, pane) {
    const smallest = smallestPane(total);
    return Math.min(total - smallest, Math.max(smallest, pane));
  }

  function tenth(fraction) {
    return Math.round(fraction * 1000) / 10;
  }

  global.SplitSizing = {
    MIN_PANE_PIXELS: MIN_PANE_PIXELS,

    /* The share of `total` the second pane takes when the divider is dropped at
       `offset`, never starving either pane. */
    secondPaneFraction: function (total, offset) {
      if (!(total > 0)) return EVEN;
      return clampPane(total, total - offset) / total;
    },

    /* A remembered fraction, pulled back inside the minimums of the container it
       is being restored into. */
    clampFraction: function (total, fraction) {
      if (!(total > 0)) return EVEN;
      return clampPane(total, fraction * total) / total;
    },

    /* A three-track grid template — first pane, divider, second pane — giving the
       second pane `fraction` of the space the divider leaves. */
    template: function (fraction) {
      return 'minmax(0,' + tenth(1 - fraction) + 'fr) auto minmax(0,' + tenth(fraction) + 'fr)';
    },
  };
})(window);
