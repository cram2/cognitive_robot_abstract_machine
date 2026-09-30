/* ============================================================================
 * core/panel-visibility.js — which panels the Scene page shows, and where.
 *
 * The topbar's View menu lists every configured panel with a checkbox (shown or
 * not) and a region choice (which column it sits in). A disabled panel stays
 * mounted and remembers its region, a slot whose panels are all hidden collapses
 * so the remaining column takes the full width, and choosing a column moves the
 * panel there (see core/panel-arrangement.js). Both choices persist in the
 * browser's localStorage, and the menu's "Reload now" button restarts the page
 * onto the stored layout.
 *
 * The state rules (defaults, persistence format, slot collapsing) are pure and
 * testable under node; only init() touches the page.
 * ==========================================================================*/
(function (global) {
  'use strict';

  const STORAGE_KEY = 'cramera.visible-panels';
  /* localStorage key the ticked state persists under, as JSON {panelId: bool}. */

  const PANEL_LABELS = {
    'robot-scene': 'Semantic Digital Twin Scene',
    'eql': 'EQL · entity query language',
    'graph': 'Knowledge & reasoning graphs',
  };
  /* What the View menu calls each panel; an unlisted id shows as its raw id. */

  /* The ticked state: everything configured defaults to visible, stored choices
     override, stored ids that are no longer configured are dropped. */
  function read(storage, configuredIds) {
    let stored = {};
    try {
      stored = JSON.parse(storage.getItem(STORAGE_KEY) || '{}') || {};
    } catch (error) {
      stored = {};   // an unreadable value means "defaults", not a broken page
    }
    const state = {};
    (configuredIds || []).forEach(function (id) {
      state[id] = typeof stored[id] === 'boolean' ? stored[id] : true;
    });
    return state;
  }

  function write(storage, state) {
    storage.setItem(STORAGE_KEY, JSON.stringify(state));
  }

  /* Which slots still have something to show: {slotName: [visible panel ids]},
     with empty slots left out so the layout can collapse them. */
  function visibleSlots(layout, state) {
    const slots = {};
    Object.keys(layout || {}).forEach(function (slotName) {
      const visible = layout[slotName].filter(function (id) { return state[id] !== false; });
      if (visible.length) slots[slotName] = visible;
    });
    return slots;
  }

  function labelOf(id) {
    return PANEL_LABELS[id] || id;
  }

  /* A slot name as the menu shows it: 'left' -> 'Left'. */
  function slotLabel(slotName) {
    return slotName.charAt(0).toUpperCase() + slotName.slice(1);
  }

  // %% wiring the page

  function configuredIds(layout) {
    return Object.keys(layout).reduce(function (ids, slotName) {
      return ids.concat(layout[slotName]);
    }, []);
  }

  /* Hide/show the mounted panels and collapse empty slots. */
  function apply(state) {
    const layout = (global.CRAMERA_CONFIG || {}).layout || {};
    document.querySelectorAll('[data-panel]').forEach(function (panel) {
      panel.classList.toggle('hidden', state[panel.dataset.panel] === false);
    });
    const remaining = visibleSlots(layout, state);
    let visibleSlotCount = 0;
    Object.keys(layout).forEach(function (slotName) {
      const slot = document.querySelector('[data-slot="' + slotName + '"]');
      if (!slot) return;
      const shown = slotName in remaining;
      slot.classList.toggle('hidden', !shown);
      if (shown) visibleSlotCount += 1;
    });
    const split = document.querySelector('main.split');
    if (split) {
      split.style.gridTemplateColumns = 'repeat(' + Math.max(visibleSlotCount, 1) + ', 1fr)';
    }
  }

  let currentState = null;
  /* The ticked state the page runs with, kept for refresh() after layout moves. */

  /* Re-apply the current selection, e.g. after a panel moved to another slot. */
  function refresh() {
    if (currentState) apply(currentState);
  }

  /* Build the topbar View menu and apply the stored selection. */
  function init() {
    const layout = (global.CRAMERA_CONFIG || {}).layout || {};
    const actions = document.querySelector('.topbar-actions');
    if (!actions || !Object.keys(layout).length) return;
    const state = read(global.localStorage, configuredIds(layout));
    currentState = state;

    const menu = document.createElement('div');
    menu.className = 'view-menu';
    const toggle = document.createElement('button');
    toggle.className = 'view-menu-toggle';
    toggle.textContent = '⊞ View';
    toggle.title = 'choose which panels are shown';
    const list = document.createElement('div');
    list.className = 'view-menu-list hidden';
    menu.appendChild(toggle);
    menu.appendChild(list);
    actions.appendChild(menu);

    configuredIds(layout).forEach(function (id) {
      const row = document.createElement('label');
      row.className = 'view-menu-row';
      const shown = document.createElement('input');
      shown.type = 'checkbox';
      shown.checked = state[id] !== false;
      shown.title = 'show or hide this panel';
      shown.addEventListener('change', function () {
        state[id] = shown.checked;
        write(global.localStorage, state);
        apply(state);
      });
      const name = document.createElement('span');
      name.textContent = labelOf(id);
      const region = document.createElement('select');
      region.title = 'which column this panel sits in';
      Object.keys(layout).forEach(function (slotName) {
        const option = document.createElement('option');
        option.value = slotName;
        option.textContent = slotLabel(slotName);
        region.appendChild(option);
      });
      region.value = slotOf(id);
      region.addEventListener('change', function () {
        if (global.PanelArrangement) global.PanelArrangement.place(id, region.value);
        apply(state);
      });
      row.appendChild(shown);
      row.appendChild(name);
      row.appendChild(region);
      list.appendChild(row);
    });

    const reload = document.createElement('button');
    reload.className = 'view-menu-reload';
    reload.textContent = '⟳ Reload now';
    reload.title = 'restart the page onto the chosen layout';
    reload.addEventListener('click', function () { global.location.reload(); });
    list.appendChild(reload);

    function slotOf(id) {
      const current = (global.CRAMERA_CONFIG || {}).layout || {};
      return Object.keys(current).find(function (slotName) {
        return current[slotName].indexOf(id) !== -1;
      }) || Object.keys(layout)[0];
    }

    toggle.addEventListener('click', function (event) {
      event.stopPropagation();
      list.classList.toggle('hidden');
    });
    document.addEventListener('click', function (event) {
      if (!menu.contains(event.target)) list.classList.add('hidden');
    });

    apply(state);
  }

  global.PanelVisibility = {
    STORAGE_KEY: STORAGE_KEY,
    PANEL_LABELS: PANEL_LABELS,
    read: read,
    write: write,
    visibleSlots: visibleSlots,
    labelOf: labelOf,
    slotLabel: slotLabel,
    init: init,
    refresh: refresh,
  };
})(typeof window !== 'undefined' ? window : this);
