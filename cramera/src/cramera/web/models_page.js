/* ============================================================================
 * models_page.js — the Models tab: the probabilistic-model workbench page.
 *
 * A page script, not a mounted panel: it owns the whole document of models.html,
 * which is why it may address elements document-wide.
 *
 * A browser port of probabilistic_model's desktop GUI, backed by the workbench API:
 *   GET  /api/models/state         which model is loaded, its variables
 *   POST /api/models/load          {model, name} a circuit's JSON serialization
 *   POST /api/models/probability   {query, evidence} -> {probability}
 *   POST /api/models/posterior     {variables, evidence} -> {figures} (Plotly JSON)
 *   POST /api/models/mode          {evidence} -> {likelihood, modes}
 *
 * Constraint rows are mapped to API payloads by core/model-constraints.js.
 * ==========================================================================*/
(function () {
  'use strict';

  const statusEl = document.getElementById('model-status');
  const variablesEl = document.getElementById('variables');
  const fileInput = document.getElementById('model-file');

  let variables = [];      // [{name, kind, values?}] of the loaded model
  const rowReaders = {};   // rows-container id -> [read() -> row state]

  // %% model state and loading

  function refreshState() {
    fetch('/api/models/state').then(ResponseUtil.parseJson).then(function (state) {
      if (!state.ok) return showUnavailable(state.error || 'models API unavailable');
      variables = state.variables || [];
      statusEl.textContent = state.loaded
        ? (state.name || 'model') + ' · ' + variables.length + ' variables'
        : 'no model loaded';
      statusEl.classList.toggle('ready', !!state.loaded);
      renderVariables();
      renderPosteriorVariables();
      resetRows();
    }).catch(function (error) {
      showUnavailable(String(error));
    });
  }

  function showUnavailable(message) {
    statusEl.textContent = 'unavailable';
    variablesEl.innerHTML = '<div class="answer-empty">' + escapeHtml(message) + '</div>';
  }

  fileInput.addEventListener('change', function () {
    const file = fileInput.files && fileInput.files[0];
    if (!file) return;
    statusEl.textContent = 'loading ' + file.name + '…';
    file.text().then(function (text) {
      // the file travels as text: python's JSON reader accepts the Infinity/NaN
      // literals circuit files carry, the browser's JSON.parse does not
      return fetch('/api/models/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_text: text, name: file.name }),
      });
    }).then(ResponseUtil.parseJson).then(function (state) {
      if (!state.ok) {
        statusEl.textContent = 'load failed';
        variablesEl.innerHTML = '<div class="qerr">' + escapeHtml(state.error || 'load failed') + '</div>';
        return;
      }
      refreshState();
    }).catch(function (error) {
      statusEl.textContent = 'load failed';
      variablesEl.innerHTML = '<div class="qerr">' + escapeHtml(String(error)) + '</div>';
    });
    fileInput.value = '';
  });

  function renderVariables() {
    if (!variables.length) {
      variablesEl.innerHTML = '<div class="answer-empty">Load a probabilistic circuit (.json) to begin.</div>';
      return;
    }
    variablesEl.innerHTML = variables.map(function (variable) {
      const domain = variable.kind === 'symbolic'
        ? '{' + (variable.values || []).join(', ') + '}'
        : variable.kind;
      return '<div class="variable-item"><b>' + escapeHtml(variable.name) + '</b>' +
        '<span>' + escapeHtml(domain) + '</span></div>';
    }).join('');
  }

  // %% constraint rows

  function resetRows() {
    ['query-rows', 'query-evidence-rows', 'posterior-evidence-rows', 'mode-evidence-rows']
      .forEach(function (containerId) {
        document.getElementById(containerId).innerHTML = '';
        rowReaders[containerId] = [];
      });
  }

  function addRow(containerId) {
    if (!variables.length) return;
    const container = document.getElementById(containerId);
    const row = document.createElement('div');
    row.className = 'constraint-row';

    const variableSelect = document.createElement('select');
    variableSelect.innerHTML = '<option value="">variable…</option>' + variables.map(function (variable) {
      return '<option value="' + escapeHtml(variable.name) + '">' + escapeHtml(variable.name) + '</option>';
    }).join('');
    row.appendChild(variableSelect);

    const constraintHolder = document.createElement('span');
    constraintHolder.className = 'constraint-inputs';
    row.appendChild(constraintHolder);

    const removeButton = document.createElement('button');
    removeButton.className = 'row-remove';
    removeButton.textContent = '✕';
    removeButton.title = 'remove this constraint';
    row.appendChild(removeButton);

    let readConstraint = function () { return null; };

    variableSelect.addEventListener('change', function () {
      const variable = variables.find(function (candidate) { return candidate.name === variableSelect.value; });
      constraintHolder.innerHTML = '';
      if (!variable) { readConstraint = function () { return null; }; return; }
      if (variable.kind === 'symbolic') {
        const valueSelect = document.createElement('select');
        valueSelect.multiple = true;
        valueSelect.size = Math.min(4, (variable.values || []).length);
        valueSelect.innerHTML = (variable.values || []).map(function (value) {
          return '<option value="' + escapeHtml(value) + '">' + escapeHtml(value) + '</option>';
        }).join('');
        constraintHolder.appendChild(valueSelect);
        readConstraint = function () {
          return {
            variable: variable.name,
            kind: 'symbolic',
            values: Array.from(valueSelect.selectedOptions).map(function (option) { return option.value; }),
          };
        };
      } else {
        const step = variable.kind === 'integer' ? 1 : (variable.high - variable.low) / 200;
        const low = boundControl(variable, variable.low, step);
        const high = boundControl(variable, variable.high, step);
        constraintHolder.appendChild(low.element);
        constraintHolder.appendChild(document.createTextNode(' ≤ ' + variable.name + ' ≤ '));
        constraintHolder.appendChild(high.element);
        readConstraint = function () {
          return { variable: variable.name, kind: variable.kind, low: low.value(), high: high.value() };
        };
      }
    });

    const reader = function () { return readConstraint(); };
    rowReaders[containerId].push(reader);
    removeButton.addEventListener('click', function () {
      rowReaders[containerId] = rowReaders[containerId].filter(function (candidate) { return candidate !== reader; });
      row.remove();
    });
    container.appendChild(row);
  }

  // one interval bound: a slider over the variable's prior support, synced with a
  // number input for exact values
  function boundControl(variable, initial, step) {
    const holder = document.createElement('span');
    holder.className = 'bound-control';
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = variable.low;
    slider.max = variable.high;
    slider.step = step;
    slider.value = initial;
    const number = document.createElement('input');
    number.type = 'number';
    number.step = 'any';
    number.value = roundedForDisplay(initial, step);
    slider.addEventListener('input', function () {
      number.value = roundedForDisplay(parseFloat(slider.value), step);
    });
    number.addEventListener('input', function () { slider.value = number.value; });
    holder.appendChild(slider);
    holder.appendChild(number);
    return { element: holder, value: function () { return number.value; } };
  }

  function roundedForDisplay(value, step) {
    if (!isFinite(value)) return value;
    const decimals = step >= 1 ? 0 : Math.min(6, Math.max(0, Math.ceil(-Math.log10(step))));
    return parseFloat(value.toFixed(decimals));
  }

  function constraintsOf(containerId) {
    return ModelConstraints.payload((rowReaders[containerId] || []).map(function (read) { return read(); }));
  }

  document.querySelectorAll('.row-add').forEach(function (button) {
    button.addEventListener('click', function () { addRow(button.dataset.rows); });
  });

  // %% query

  document.getElementById('query-run').addEventListener('click', function () {
    const resultEl = document.getElementById('query-result');
    resultEl.innerHTML = '<div class="answer-empty">calculating…</div>';
    postJson('/api/models/probability', {
      query: constraintsOf('query-rows'),
      evidence: constraintsOf('query-evidence-rows'),
    }).then(function (payload) {
      if (!payload.ok) return showError(resultEl, payload.error);
      resultEl.innerHTML = '<div class="probability-result">P = <b>' +
        payload.probability.toFixed(6) + '</b></div>';
    }).catch(function (error) { showError(resultEl, String(error)); });
  });

  // %% posterior

  function renderPosteriorVariables() {
    const holder = document.getElementById('posterior-variables');
    holder.innerHTML = variables.map(function (variable) {
      return '<label class="lp-row"><input type="checkbox" value="' + escapeHtml(variable.name) + '" />' +
        escapeHtml(variable.name) + '</label>';
    }).join('') || '<div class="answer-empty">load a model first</div>';
  }

  document.getElementById('posterior-run').addEventListener('click', function () {
    const plotsEl = document.getElementById('posterior-plots');
    const chosen = Array.from(document.querySelectorAll('#posterior-variables input:checked'))
      .map(function (checkbox) { return checkbox.value; });
    if (!chosen.length) {
      plotsEl.innerHTML = '<div class="qerr">pick at least one query variable</div>';
      return;
    }
    plotsEl.innerHTML = '<div class="answer-empty">calculating…</div>';
    postJson('/api/models/posterior', {
      variables: chosen,
      evidence: constraintsOf('posterior-evidence-rows'),
    }).then(function (payload) {
      if (!payload.ok) return showError(plotsEl, payload.error);
      plotsEl.innerHTML = '';
      chosen.forEach(function (name) {
        const figure = payload.figures[name];
        if (!figure) return;
        const card = document.createElement('div');
        card.className = 'plot-card';
        plotsEl.appendChild(card);
        Plotly.newPlot(card, figure.data, figure.layout, { responsive: true, displaylogo: false });
      });
    }).catch(function (error) { showError(plotsEl, String(error)); });
  });

  // %% mode

  document.getElementById('mode-run').addEventListener('click', function () {
    const resultEl = document.getElementById('mode-result');
    resultEl.innerHTML = '<div class="answer-empty">calculating…</div>';
    postJson('/api/models/mode', {
      evidence: constraintsOf('mode-evidence-rows'),
    }).then(function (payload) {
      if (!payload.ok) return showError(resultEl, payload.error);
      const modes = payload.modes.map(function (mode, index) {
        const assignments = Object.keys(mode).map(function (name) {
          return '<div class="mode-assignment"><span>' + escapeHtml(name) + '</span>' +
            '<code>' + escapeHtml(mode[name]) + '</code></div>';
        }).join('');
        const title = payload.modes.length > 1
          ? 'mode ' + (index + 1) + ' of ' + payload.modes.length : 'mode';
        return '<div class="mode-card"><div class="lp-title">' + title + '</div>' + assignments + '</div>';
      }).join('');
      resultEl.innerHTML = '<div class="probability-result">likelihood <b>' +
        payload.likelihood.toExponential(4) + '</b></div>' + modes;
    }).catch(function (error) { showError(resultEl, String(error)); });
  });

  // %% helpers

  function postJson(url, body) {
    return fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(ResponseUtil.parseJson);
  }

  function showError(element, message) {
    element.innerHTML = '<div class="qerr">' + escapeHtml(message || 'unknown error') + '</div>';
  }

  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, function (character) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character];
    });
  }

  refreshState();
})();
