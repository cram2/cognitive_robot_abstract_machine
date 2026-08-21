/* Draggable splitter + maximize toggle for the knowledge-graph panel.
   Include after the DOM (end of body). Needs main.split with the two
   [data-slot] elements index.html declares. */
(function () {
  'use strict';

  const split = document.querySelector('main.split');
  const left  = document.querySelector('[data-slot="left"]');
  const right = document.querySelector('[data-slot="right"]');
  if (!split || !left || !right) return;

  const storeKey = 'splitRight:' + location.pathname.split('/').pop();

  // %% divider
  const divider = document.createElement('div');
  divider.className = 'split-divider';
  divider.title = 'Drag to resize · double-click for 50/50';
  split.insertBefore(divider, right);
  split.style.columnGap = '4px'; /* 4 + 8px divider + 4 = former 16px gap */

  function applyRight(pct) {
    pct = Math.min(75, Math.max(25, pct));
    split.style.gridTemplateColumns = `minmax(0,${100 - pct}fr) auto minmax(0,${pct}fr)`;
    return pct;
  }

  let rightPct = parseFloat(localStorage.getItem(storeKey)) || 50;
  applyRight(rightPct);

  divider.addEventListener('pointerdown', e => {
    e.preventDefault();
    divider.setPointerCapture(e.pointerId);
    divider.classList.add('dragging');
    const rect = split.getBoundingClientRect();

    function onMove(ev) {
      rightPct = applyRight((rect.right - ev.clientX) / rect.width * 100);
    }
    function onUp() {
      divider.classList.remove('dragging');
      divider.removeEventListener('pointermove', onMove);
      divider.removeEventListener('pointerup', onUp);
      localStorage.setItem(storeKey, rightPct.toFixed(1));
    }
    divider.addEventListener('pointermove', onMove);
    divider.addEventListener('pointerup', onUp);
  });

  divider.addEventListener('dblclick', () => {
    rightPct = applyRight(50);
    localStorage.setItem(storeKey, '50');
  });

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
      if (max) split.style.gridTemplateColumns = '';
      else applyRight(rightPct);
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

    function reflow() {
      // let the layout settle, then let vis re-fit to the new size
      window.dispatchEvent(new Event('resize'));
      if (window.Graph && Graph.resize) Graph.resize();
    }

    function toggleGraphFull() {
      const full = graphWrap.classList.toggle('graph-fullscreen');
      gbtn.textContent = full ? '⊟' : '⛶';
      gbtn.title = full ? 'Leave fullscreen (Esc)' : 'Graph to fullscreen';
      document.body.classList.toggle('graph-full-open', full);
      setTimeout(reflow, 60);
    }

    gbtn.addEventListener('click', toggleGraphFull);
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape' && graphWrap.classList.contains('graph-fullscreen')) toggleGraphFull();
    });
  }
})();
