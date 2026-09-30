/* ============================================================================
 * core/folding.js — which sections the reader has folded away.
 *
 * The layers overlay covers the corner of the 3D view; the questions on offer
 * push the answer down the panel. Both have to fold down to their heading, and
 * to stay folded on the next page. Which ones are folded, and what the fold
 * button says about it, live here — nothing touches the DOM.
 * ==========================================================================*/
(function () {
  'use strict';

  const STORAGE_PREFIX = 'folded:';

  function storageKey(section) {
    return STORAGE_PREFIX + section;
  }

  // reading a store that refuses to answer (a private window, storage turned off)
  // is not worth a broken viewer: the section then simply opens unfolded
  function folded(store, section) {
    try {
      return store.getItem(storageKey(section)) === 'yes';
    } catch (error) {
      return false;
    }
  }

  function remember(store, section, isFolded) {
    try {
      store.setItem(storageKey(section), isFolded ? 'yes' : 'no');
    } catch (error) {
      /* a viewer that cannot remember the choice still honours it while it is open */
    }
  }

  // the fold button of one section, as it reads in each state
  function button(isFolded, what) {
    return isFolded
      ? { glyph: '▸', title: 'Show ' + what }
      : { glyph: '▾', title: 'Fold ' + what + ' away' };
  }

  window.Folding = {
    storageKey: storageKey,
    folded: folded,
    remember: remember,
    button: button,
  };
})();
