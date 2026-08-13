/* ============================================================================
 * core/split-resize.js — draggable dividers and the maximize toggles.
 *
 * Two dividers, both driven by core/split-sizing.js:
 *   · a column divider between the scene slot and the knowledge slot
 *   · a row divider between the panels a slot stacks (EQL above the graph)
 *
 * A divider turns its container into a three-track grid — first pane, divider,
 * second pane — so the gap between panes stays a layout gap and the panes keep
 * their share of the container when the window resizes.
 *
 * Include after the DOM and after Panels.boot(): the row divider is placed
 * between panels the registry has already mounted.
 * ==========================================================================*/
(function () {
  'use strict';

  const split = document.querySelector('main.split');
  const left  = document.querySelector('[data-slot="left"]');
  const right = document.querySelector('[data-slot="right"]');
  if (!split || !left || !right) return;

  const page = location.pathname.split('/').pop();
  const GAP_PIXELS = '4px';
  /* 4 + the divider's 8px + 4 = the 16px gap app.css lays out without one. */

  // %% the two axes a divider can work along
  const COLUMNS = {
    className: 'pane-divider split-divider',
    title: 'Drag to resize the scene against the knowledge column · double-click to reset',
    template: 'gridTemplateColumns',
    defaultFraction: 0.5,
    prepare: function (container) { container.style.columnGap = GAP_PIXELS; },
    total: function (rect) { return rect.width; },
    offset: function (rect, event) { return event.clientX - rect.left; },
  };

  const ROWS = {
    className: 'pane-divider slot-divider',
    title: 'Drag to resize the panels above and below · double-click to reset',
    template: 'gridTemplateRows',
    defaultFraction: 0.6,
    /* The lower panel — the graph — starts with the larger share. */
    prepare: function (container) {
      container.style.display = 'grid';
      container.style.rowGap = GAP_PIXELS;
    },
    total: function (rect) { return rect.height; },
    offset: function (rect, event) { return event.clientY - rect.top; },
  };

  // %% let the canvases catch up with their new size
  function reflow() {
    window.dispatchEvent(new Event('resize'));
  }

  function refit() {
    if (window.Graph && Graph.resize) Graph.resize();
  }

  // %% one divider
  /* Insert a divider before `secondPane` and let it resize the two panes it sits
     between, remembering the size under `storeKey`. Returns the handles the
     maximize toggle needs to drop and restore the sizing. */
  function installDivider(container, secondPane, axis, storeKey) {
    const divider = document.createElement('div');
    divider.className = axis.className;
    divider.title = axis.title;
    container.insertBefore(divider, secondPane);
    axis.prepare(container);

    let fraction = parseFloat(localStorage.getItem(storeKey));

    function apply(next) {
      fraction = SplitSizing.clampFraction(axis.total(container.getBoundingClientRect()), next);
      container.style[axis.template] = SplitSizing.template(fraction);
    }

    function remember() {
      localStorage.setItem(storeKey, fraction.toFixed(3));
    }

    // a value outside (0,1) — no size remembered yet, or the percentages an
    // earlier build stored — starts from the axis default
    apply(fraction > 0 && fraction < 1 ? fraction : axis.defaultFraction);

    divider.addEventListener('pointerdown', function (event) {
      event.preventDefault();
      divider.setPointerCapture(event.pointerId);
      divider.classList.add('dragging');
      const rect = container.getBoundingClientRect();

      function onMove(moved) {
        fraction = SplitSizing.secondPaneFraction(axis.total(rect), axis.offset(rect, moved));
        container.style[axis.template] = SplitSizing.template(fraction);
        reflow();
      }
      function onUp() {
        divider.classList.remove('dragging');
        divider.removeEventListener('pointermove', onMove);
        divider.removeEventListener('pointerup', onUp);
        remember();
        refit();
      }
      divider.addEventListener('pointermove', onMove);
      divider.addEventListener('pointerup', onUp);
    });

    divider.addEventListener('dblclick', function () {
      apply(axis.defaultFraction);
      remember();
      reflow();
      refit();
    });

    return {
      restore: function () { apply(fraction); },
      release: function () { container.style[axis.template] = ''; },
    };
  }

  const columns = installDivider(split, right, COLUMNS, 'splitRight:' + page);

  // %% the panels a slot stacks
  /* A three-track grid holds one divider, so a slot is only made resizable when
     it stacks exactly two panels. */
  const stacked = Array.prototype.filter.call(right.children, function (child) {
    return child.dataset.panel;
  });
  if (stacked.length === 2) installDivider(right, stacked[1], ROWS, 'splitBottom:' + page);

  // %% maximize button on the knowledge panel
  const head = right.querySelector('.panel-head');
  if (head) {
    const btn = document.createElement('button');
    btn.className = 'kg-max-btn';
    btn.title = 'Maximize the knowledge graph';
    btn.textContent = '⛶';
    head.appendChild(btn);

    btn.addEventListener('click', () => {
      const max = split.classList.toggle('kg-maximized');
      btn.textContent = max ? '⊟' : '⛶';
      btn.title = max ? 'Back to the split view' : 'Maximize the knowledge graph';
      if (max) columns.release();
      else columns.restore();
      reflow();
      refit();
    });

    document.addEventListener('keydown', e => {
      if (e.key === 'Escape' && split.classList.contains('kg-maximized')) btn.click();
    });
  }

  // %% true fullscreen for the graph itself
  // The panel-maximize above only widens the right column; the query box and
  // answer panel still eat space. This makes just the graph cover the whole
  // window so it gets every pixel.
  const graphWrap = right.querySelector('.graph-wrap');
  if (graphWrap) {
    const gbtn = document.createElement('button');
    gbtn.className = 'graph-max-btn';
    gbtn.title = 'Graph to fullscreen';
    gbtn.textContent = '⛶';
    graphWrap.appendChild(gbtn);

    function toggleGraphFull() {
      const full = graphWrap.classList.toggle('graph-fullscreen');
      gbtn.textContent = full ? '⊟' : '⛶';
      gbtn.title = full ? 'Leave fullscreen (Esc)' : 'Graph to fullscreen';
      document.body.classList.toggle('graph-full-open', full);
      // let the layout settle, then let vis re-fit to the new size
      setTimeout(function () { reflow(); refit(); }, 60);
    }

    gbtn.addEventListener('click', toggleGraphFull);
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape' && graphWrap.classList.contains('graph-fullscreen')) toggleGraphFull();
    });
  }
})();
