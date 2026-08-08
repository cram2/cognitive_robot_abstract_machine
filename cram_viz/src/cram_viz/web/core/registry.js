/* ============================================================================
 * core/registry.js — the panel registry and mounting.
 *
 * A panel is a self-contained UI unit (3D scene, EQL console, graph view, …).
 * It registers a factory under an id; config.js decides which panels are
 * mounted into which layout slot. Swapping a visualization = editing
 * config.js, no other file changes.
 *
 *   Panels.define('my-panel', function (root, bus) {
 *     root.innerHTML = '…';               // the panel owns its DOM subtree
 *     bus.on('entity:highlight', …);      // talk to others via the bus only
 *     return { destroy: function () {…} } // optional cleanup
 *   });
 *
 * Slots are elements with a data-slot attribute in index.html. Panels.boot()
 * reads window.CRAM_VIZ_CONFIG.layout = {slotName: [panelId, …]} and mounts
 * every configured panel; unknown ids are reported, not fatal.
 * ==========================================================================*/
(function () {
  'use strict';

  const factories = {};   // id -> factory(root, bus)
  const mounted = [];     // {id, instance}

  function mountInto(slotEl, id) {
    const factory = factories[id];
    if (!factory) {
      console.error('[panels] "' + id + '" is configured but not defined — ' +
        'is its panel.js included in index.html?');
      return;
    }
    const root = document.createElement('section');
    root.className = 'panel panel-' + id;
    root.dataset.panel = id;
    slotEl.appendChild(root);
    try {
      mounted.push({ id: id, instance: factory(root, window.Bus) || {} });
    } catch (err) {
      console.error('[panels] mounting "' + id + '" failed:', err);
      root.innerHTML = '<div class="panel-error">panel "' + id + '" failed to mount — see console</div>';
    }
  }

  window.Panels = {
    define: function (id, factory) {
      if (factories[id]) console.warn('[panels] "' + id + '" defined twice; keeping the last one');
      factories[id] = factory;
    },
    boot: function () {
      const layout = (window.CRAM_VIZ_CONFIG || {}).layout || {};
      Object.keys(layout).forEach(function (slotName) {
        const slotEl = document.querySelector('[data-slot="' + slotName + '"]');
        if (!slotEl) {
          console.error('[panels] no slot element for "' + slotName + '"');
          return;
        }
        layout[slotName].forEach(function (id) { mountInto(slotEl, id); });
      });
    },
    // tear every mounted panel down, so its timers and observers stop
    unmountAll: function () {
      mounted.forEach(function (entry) {
        if (typeof entry.instance.destroy !== 'function') return;
        try {
          entry.instance.destroy();
        } catch (err) {
          console.error('[panels] destroying "' + entry.id + '" failed:', err);
        }
      });
      mounted.length = 0;
    },
    // introspection (used by tests and the console)
    defined: function () { return Object.keys(factories); },
    mounted: function () { return mounted.map(function (m) { return m.id; }); },
  };
})();
