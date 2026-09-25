/* ============================================================================
 * panels/eql/suggestions.js — the query box's suggestion menu.
 *
 * The DOM half of completion: core/completion.js decides what to offer for the
 * word under the caret, this shows it under the box and takes the keys that
 * move through it. Names come from whichever source answers the queries; the
 * members behind a dot are asked for per type and remembered, since a type's
 * members do not change while a query is being written.
 *
 * Keys, while the menu is open: ArrowUp/ArrowDown move, Enter/Tab accept,
 * Escape closes. Closed, it takes no keys at all, so Enter still runs the query.
 * ==========================================================================*/
(function () {
  'use strict';

  const VISIBLE_LIMIT = 12;
  const MINIMUM_HEIGHT = 140;    // px kept for the menu even in a cramped window

  // options: {input, anchor, entries(), fetchMembers(name) -> Promise<entries>}
  function of(options) {
    const input = options.input;
    const menu = document.createElement('div');
    menu.className = 'completion-menu';
    menu.style.display = 'none';
    options.anchor.appendChild(menu);

    let offered = [];              // what the menu currently shows
    let active = 0;                // which row is selected
    let token = null;              // the word the offers are for
    let vocabularyRevision = 0;
    const membersByOwner = {};     // owner name -> its members, once asked for

    function isOpen() {
      return menu.style.display !== 'none';
    }

    function close() {
      menu.style.display = 'none';
      offered = [];
      token = null;
    }

    // what the caret is on now, and what to offer for it
    function refresh() {
      const current = Completion.tokenAt(input.value, input.selectionStart);
      if (!current.prefix && !current.owner) return close();
      if (!current.owner) return show(current, options.entries());
      const known = membersByOwner[current.owner];
      if (known) return show(current, known);
      const requestedVocabulary = vocabularyRevision;
      options.fetchMembers(current.owner).then(function (members) {
        if (requestedVocabulary !== vocabularyRevision) return;
        membersByOwner[current.owner] = members;
        const latest = Completion.tokenAt(input.value, input.selectionStart);
        // the caret may have moved on while the members were being fetched
        if (latest.owner === current.owner) show(latest, members);
      });
    }

    function show(current, entries) {
      offered = Completion.suggest(entries, current, VISIBLE_LIMIT);
      token = current;
      if (!offered.length) return close();
      active = 0;
      render();
      place();
      menu.style.display = '';
    }

    // under the box, as wide as it, and never taller than the room left below it
    function place() {
      const box = input.getBoundingClientRect();
      menu.style.left = box.left + 'px';
      menu.style.width = box.width + 'px';
      menu.style.top = (box.bottom + 4) + 'px';
      menu.style.maxHeight =
        Math.max(MINIMUM_HEIGHT, window.innerHeight - box.bottom - 24) + 'px';
    }

    function render() {
      menu.innerHTML = '';
      offered.forEach(function (entry, position) {
        menu.appendChild(row(entry, position));
      });
    }

    function row(entry, position) {
      const element = document.createElement('div');
      element.className = 'completion-item' + (position === active ? ' active' : '');
      element.appendChild(cell('completion-kind', kindLabel(entry.kind)));
      element.appendChild(cell('completion-name', entry.name));
      element.appendChild(cell('completion-detail', entry.detail || ''));
      element.appendChild(cell('completion-origin', originLabel(entry)));
      // mousedown rather than click: the box must not lose focus before accepting
      element.addEventListener('mousedown', function (event) {
        event.preventDefault();
        active = position;
        accept();
      });
      element.addEventListener('mouseenter', function () {
        active = position;
        render();
      });
      return element;
    }

    function cell(className, text) {
      const element = document.createElement('span');
      element.className = className;
      element.textContent = text;
      return element;
    }

    function kindLabel(kind) {
      return String(kind || '').replace('_', ' ');
    }

    // where a name comes from, and how many other modules define one like it —
    // the only way to tell which of several same-named classes a query gets
    function originLabel(entry) {
      if (!entry.module) return '';
      if (!entry.further_modules) return entry.module;
      return entry.module + '  (+' + entry.further_modules + ' more)';
    }

    function accept() {
      const entry = offered[active];
      if (!entry) return close();
      const applied = Completion.applied(input.value, token, entry);
      input.value = applied.text;
      input.setSelectionRange(applied.caret, applied.caret);
      close();
      input.focus();
      // a class or variable is usually followed by reaching into it
      refresh();
    }

    function move(step) {
      active = (active + step + offered.length) % offered.length;
      render();
    }

    // true when the menu took the key, so the panel leaves it alone
    function handledKey(event) {
      if (!isOpen()) return false;
      if (event.key === 'ArrowDown') { move(1); return true; }
      if (event.key === 'ArrowUp') { move(-1); return true; }
      if (event.key === 'Enter' || event.key === 'Tab') { accept(); return true; }
      if (event.key === 'Escape') { close(); return true; }
      return false;
    }

    input.addEventListener('input', refresh);
    input.addEventListener('blur', function () {
      // a click on a row is a mousedown, which has already accepted by now
      close();
    });

    return {
      refresh: refresh,
      close: close,
      isOpen: isOpen,
      handledKey: handledKey,
      forget: function () {
        vocabularyRevision += 1;
        close();
        for (const owner in membersByOwner) delete membersByOwner[owner];
      },
    };
  }

  window.EqlSuggestions = { of: of };
})();
