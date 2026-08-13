/* ============================================================================
 * core/panel-arrangement.js — where each Scene panel sits.
 *
 * The View menu assigns every panel a region (left or right column); place() moves
 * the panel there and the arrangement persists in the browser's localStorage —
 * install() replaces the configured layout with the stored one before the panels
 * mount, so every reload comes back to the same arrangement.
 *
 * The layout rules (normalizing a stored arrangement against the configured
 * panels, moving a panel to a slot position) are pure and testable under node;
 * only install() and place() touch the page.
 * ==========================================================================*/
(function (global) {
  'use strict';

  const STORAGE_KEY = 'cramera.panel-arrangement';
  /* localStorage key the arrangement persists under, as JSON {slotName: [ids]}. */

  /* A stored arrangement, cleaned against what is actually configured: ids that
     left the configuration are dropped, ids new to the configuration appear at
     their configured spot, and slots stay in the configured order. */
  function read(storage, configuredLayout) {
    let stored = {};
    try {
      stored = JSON.parse(storage.getItem(STORAGE_KEY) || '{}') || {};
    } catch (error) {
      stored = {};   // an unreadable value means "the configured layout"
    }
    return normalize(stored, configuredLayout);
  }

  function normalize(stored, configuredLayout) {
    const configuredIds = allIds(configuredLayout);
    const layout = {};
    const placed = {};
    Object.keys(configuredLayout).forEach(function (slotName) {
      layout[slotName] = (stored[slotName] || []).filter(function (id) {
        const known = configuredIds.indexOf(id) !== -1 && !placed[id];
        if (known) placed[id] = true;
        return known;
      });
    });
    Object.keys(configuredLayout).forEach(function (slotName) {
      configuredLayout[slotName].forEach(function (id) {
        if (placed[id]) return;
        placed[id] = true;
        layout[slotName].push(id);
      });
    });
    return layout;
  }

  function write(storage, layout) {
    storage.setItem(STORAGE_KEY, JSON.stringify(layout));
  }

  /* The layout after moving one panel into a slot at a position. */
  function moved(layout, id, targetSlot, index) {
    const next = {};
    Object.keys(layout).forEach(function (slotName) {
      next[slotName] = layout[slotName].filter(function (other) { return other !== id; });
    });
    if (!(targetSlot in next)) return layout;
    const at = Math.max(0, Math.min(index, next[targetSlot].length));
    next[targetSlot].splice(at, 0, id);
    return next;
  }

  function allIds(layout) {
    return Object.keys(layout).reduce(function (ids, slotName) {
      return ids.concat(layout[slotName]);
    }, []);
  }

  // %% wiring the page

  /* Replace the configured layout with the stored arrangement, before boot. */
  function install() {
    const config = global.CRAMERA_CONFIG || {};
    if (!config.layout) return;
    config.layout = read(global.localStorage, config.layout);
  }

  /* Move a panel to the end of a slot, on the page and in the stored layout. */
  function place(id, slotName) {
    const config = global.CRAMERA_CONFIG || {};
    config.layout = moved(config.layout || {}, id, slotName, Infinity);
    const panel = document.querySelector('[data-panel="' + id + '"]');
    const slot = document.querySelector('[data-slot="' + slotName + '"]');
    if (panel && slot) slot.appendChild(panel);
    write(global.localStorage, config.layout);
  }

  global.PanelArrangement = {
    STORAGE_KEY: STORAGE_KEY,
    read: read,
    write: write,
    normalize: normalize,
    moved: moved,
    install: install,
    place: place,
  };
})(typeof window !== 'undefined' ? window : this);
