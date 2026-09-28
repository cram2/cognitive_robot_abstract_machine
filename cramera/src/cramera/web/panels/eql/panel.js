/* ============================================================================
 * panels/eql/panel.js — the EQL (Entity Query Language) console.
 *
 * Query bar + question display + presets + answer panel. Queries are typed into the
 * bar, picked as presets, or spoken through the bar's record button
 * (core/voice.js); underneath, the question asked is shown big, in English
 * (core/question_display.js) — the query's verbalization, with class and attribute
 * words linking to what explains them: their documentation, or their source. The
 * question, the presets and the answer scroll together under the bar. Queries go wherever
 * core/query_source.js points: the server, answering from the recorded episode
 * knowledge base, or an attached demo, answering from its own live state. The
 * knowledge-base overview (GET /api/knowledge) always provides the per-entity details
 * the describe panel shows.
 *
 * Bus events:
 *   emits    knowledge:ready {payload}          the /api/knowledge overview, once loaded
 *   emits    entity:highlight {ids, focus?}   results / described entity
 *   emits    voice:transcript {text}     a spoken question, as recognized text
 *   listens  voice:transcript {text}     match the question to a preset and run it
 *   listens  scene:part-clicked {id}     describe the clicked part
 *   listens  scene:step {step}           describe the running episode
 *   listens  entity:select {id, detail, relations}   node clicked in a graph
 *   listens  live:changed {on, url}      answer from the demo instead of the recording
 * ==========================================================================*/
Panels.define('eql', function (root, bus) {
  root.innerHTML =
    '<div class="panel-head">' +
    '  <h2>EQL · entity query language</h2>' +
    '  <span id="knowledge-status" class="knowledge-status">loading knowledge base…</span>' +
    '</div>' +
    '<div class="query-box" id="query-box">' +
    '  <div class="query-bar">' +
    '    <textarea id="query-input" rows="2" spellcheck="false" placeholder="the(entity(scene_object).where(scene_object.name == \'milk\'))   —  vars: scene_object, episode, arm, joint, robot"></textarea>' +
    '    <button id="query-run">Run</button>' +
    '    <button id="voice-ask" class="voice-ask" title="ask a question by voice">🎤</button>' +
    '  </div>' +
    '</div>' +
    '<div class="console-body">' +
    '  <div id="question" class="question"></div>' +
    '  <div id="presets" class="presets"></div>' +
    '  <div id="answer" class="answer"><div class="answer-empty">Loading the episode knowledge base…</div></div>' +
    '</div>';

  const knowledgeStatus = root.querySelector('#knowledge-status');
  const answerEl = root.querySelector('#answer');
  const input = root.querySelector('#query-input');
  const runBtn = root.querySelector('#query-run');
  const questionEl = root.querySelector('#question');
  const voiceButton = root.querySelector('#voice-ask');
  const presetsEl = root.querySelector('#presets');

  // %% showing an answer
  // The answer sits at the bottom of a scrolling console, under everything that can be
  // asked, so an answered question is scrolled to as well as written — typed, picked or
  // spoken. Only what was asked: the descriptions written here arrive unasked — a graph
  // selects a node as it loads, and the running episode replaces its own step as it goes.
  function showAnswer(html) {
    answerEl.innerHTML = html;
    answerEl.scrollIntoView({ block: 'nearest' });
  }

  const ASK_HINT = 'The question you ask appears here in English — run a query, or pick one below.';
  questionEl.innerHTML = QuestionDisplay.hint(ASK_HINT);

  let knowledge = null;   // /api/knowledge overview (presets + entity details)
  let source = QuerySource.of(null);   // where queries and presets are answered from
  let vocabulary = [];                 // every name the answering source offers
  let recordedStatus = '';             // what the recorded scene calls itself
  // which body of knowledge is asked about: the last preset picked
  let askedScope = null;
  let vocabularyRequest = 0;
  let voiceRequest = 0;
  let running = false;

  // %% boot
  fetch(SceneContext.withScene('/api/knowledge')).then(ResponseUtil.parseJson).then(boot).catch(function (err) {
    knowledgeStatus.textContent = 'EQL unavailable';
    answerEl.innerHTML = '<div class="qerr">EQL unavailable: ' + esc(errorText(err)) + '</div>';
  });

  function boot(payload) {
    if (!payload.ok) {
      knowledgeStatus.textContent = 'EQL unavailable';
      answerEl.innerHTML = '<div class="qerr">' + esc(payload.error || 'unknown error') + '</div>';
      return;
    }
    knowledge = payload;
    recordedStatus = payload.status;
    knowledgeStatus.classList.add('ready');
    if (!source.live) showSource(recordedStatus, payload.presets || []);
    answerEl.innerHTML = '';       // the loading note goes once the base is there
    bus.emit('knowledge:ready', { payload: payload });
  }

  // %% which source answers
  bus.on('live:changed', function (live) {
    source = QuerySource.of(live);
    const requestedSource = source;
    askedScope = null;
    running = false;
    runBtn.textContent = 'Run';
    answerEl.innerHTML = '';
    questionEl.innerHTML = QuestionDisplay.hint(ASK_HINT);
    vocabularyRequest += 1;
    voiceRequest += 1;
    vocabulary = [];
    suggestions.forget();
    if (!source.live) return showSource(recordedStatus, (knowledge && knowledge.presets) || []);
    knowledgeStatus.textContent = 'loading live queries…';
    buildPresets([]);
    fetch(requestedSource.presetsUrl).then(ResponseUtil.parseJson).then(function (payload) {
      if (requestedSource !== source) return;
      if (!payload.ok) throw new Error(payload.error || 'the demo offers no queries');
      showSource('live · ' + payload.title, payload.presets || [], payload.scopes);
    }).catch(function (err) {
      if (requestedSource !== source) return;
      showSource('live · no queries (' + errorText(err) + ')', []);
    });
  });

  function showSource(status, presets, scopes) {
    knowledgeStatus.textContent = status;
    buildPresets(presets, scopes);
    loadVocabulary();
  }

  // %% what the box may name
  // Re-asked whenever the answering source changes: a demo's own variables are not the
  // recorded scene's, and only the source that answers a query knows what it accepts.
  function loadVocabulary() {
    const request = ++vocabularyRequest;
    const requestedSource = source;
    vocabulary = [];
    suggestions.forget();
    fetch(requestedSource.vocabularyUrl(askedScope)).then(ResponseUtil.parseJson)
      .then(function (payload) {
        if (requestedSource !== source || request !== vocabularyRequest) return;
        vocabulary = (payload && payload.ok && payload.entries) || [];
      })
      .catch(function () {
        if (requestedSource === source && request === vocabularyRequest) vocabulary = [];
      });
  }

  function fetchMembers(owner) {
    return fetch(source.membersUrl(owner, askedScope)).then(ResponseUtil.parseJson)
      .then(function (payload) {
        return (payload && payload.ok && payload.members) || [];
      })
      .catch(function () { return []; });
  }

  const suggestions = EqlSuggestions.of({
    input: input,
    anchor: root.querySelector('#query-box'),
    entries: function () { return vocabulary; },
    fetchMembers: fetchMembers,
  });

  // %% describe an entity in the answer panel
  // Two sources: our own knowledge.details (scene clicks) and graph panels, which send
  // the full detail payload of the node the user clicked (entity:select).
  function describe(id, detail, relations) {
    const d = detail || (knowledge && knowledge.details && knowledge.details[id]);
    if (!d) return;
    let html = '<div class="goal">entity · ' + esc(id) + '</div>';
    html += '<div class="ansrow"><span class="tag" style="background:' + groupColor(d.group) + '">' +
      esc(d.group) + '</span><div class="body"><span class="name">' + esc(d.label) + '</span></div></div>';
    (d.lines || []).forEach(function (l) {
      html += '<div class="ansrow"><div class="body"><span class="name">' + esc(l) + '</span></div></div>';
    });
    if (relations && relations.length) {
      html += '<div class="ansub">Relations</div>';
      relations.slice(0, 40).forEach(function (r) {
        html += '<div class="ansrow"><div class="body"><span class="name">' + esc(r.s) +
          ' <span class="rel">' + esc(r.p) + '</span> ' + esc(r.o) + '</span></div></div>';
      });
      if (relations.length > 40) {
        html += '<div class="ansrow"><div class="body"><span class="sub">… ' + (relations.length - 40) + ' more</span></div></div>';
      }
    }
    answerEl.innerHTML = html;
    bus.emit('entity:highlight', { ids: [id], focus: id });
  }

  bus.on('scene:part-clicked', function (p) { describe(p.id); });
  bus.on('entity:select', function (p) { describe(p.id, p.detail, p.relations); });
  bus.on('scene:step', function (p) {
    if (p.step !== '__done__' && knowledge && knowledge.details && knowledge.details[p.step]) describe(p.step);
  });

  // %% presets
  function buildPresets(presets, scopes) {
    presetsEl.innerHTML = '';
    PresetGroups.of(presets, scopes).forEach(function (group) {
      const row = document.createElement('div');
      row.className = 'preset-row';
      group.presets.forEach(function (p) { row.appendChild(presetButton(p, group.name)); });
      // a group with a heading of its own folds away under it, so a long list of
      // questions stops pushing the answer down the panel; the only group there is has
      // no heading, and nothing to fold it under
      if (group.label) {
        presetsEl.appendChild(foldableHeading(group, row));
      }
      presetsEl.appendChild(row);
    });
  }

  // the heading of one group of questions, folding its row away and remembering it
  function foldableHeading(group, row) {
    const section = 'presets:' + group.name;
    const heading = document.createElement('div');
    heading.className = 'preset-group';
    const label = document.createElement('span');
    label.textContent = group.label;
    const fold = document.createElement('button');
    fold.className = 'layer-gear lp-fold';
    heading.appendChild(label);
    heading.appendChild(fold);

    let folded = Folding.folded(window.localStorage, section);
    function show() {
      const face = Folding.button(folded, 'these questions');
      fold.textContent = face.glyph;
      fold.title = face.title;
      row.classList.toggle('folded', folded);
    }
    show();
    heading.addEventListener('click', function (event) {
      event.preventDefault();
      folded = !folded;
      Folding.remember(window.localStorage, section, folded);
      show();
    });
    return heading;
  }

  function presetButton(p, scope) {
    // a bundle's own questions are about the demo it was recorded from, so they can
    // only be answered while that demo is attached
    const unanswerable = p.requires_live && !source.live;
    const b = document.createElement('div');
    b.className = unanswerable ? 'preset unavailable' : 'preset';
    b.textContent = p.text;
    b.title = unanswerable ? 'start the demo to answer this' : p.code;
    if (!unanswerable) {
      b.addEventListener('click', function () {
        input.value = p.code;
        // the box now asks about this preset's own body of knowledge, whose variables
        // are not the ones the previous scope offered
        if (askedScope !== scope) {
          askedScope = scope;
          loadVocabulary();
        }
        showQuestion(p);
        runQuery(p.code);
      });
    }
    return b;
  }

  // %% the asked question, shown big under the query bar
  function showQuestion(question) {
    questionEl.innerHTML = QuestionDisplay.markup(question);
    questionEl.title = question.code || '';
  }

  // %% asking by voice
  // The capture only produces text; the transcript goes over the bus, so any panel
  // can consume a spoken question.
  const voice = VoiceCapture.create({
    onTranscript: function (text) { bus.emit('voice:transcript', { text: text }); },
    onState: function (listening) {
      voiceButton.classList.toggle('listening', listening);
      if (listening) questionEl.innerHTML = QuestionDisplay.hint('Listening…');
    },
    onError: function (message) {
      questionEl.innerHTML = QuestionDisplay.hint(ASK_HINT);
      showAnswer('<div class="qerr">voice input failed: ' + esc(message) + '</div>');
    },
  });
  if (!voice.supported) {
    voiceButton.disabled = true;
    voiceButton.title = 'speech recognition is not available in this browser';
  }
  voiceButton.addEventListener('click', function () {
    if (voice.listening) voice.stop(); else voice.start();
  });

  // the default consumer: recognize the spoken question as one of the presets on
  // offer and run it as if its button had been clicked — or say it can't be answered
  bus.on('voice:transcript', function (p) { askSpokenQuestion(p.text); });

  async function askSpokenQuestion(text) {
    text = (text || '').trim(); if (!text) return;
    const request = ++voiceRequest;
    const requestedSource = source;
    questionEl.innerHTML = QuestionDisplay.hint('You asked: “' + text + '”');
    try {
      const r = await fetch(requestedSource.questionUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text }),
      });
      const res = await ResponseUtil.parseJson(r);
      if (requestedSource !== source || request !== voiceRequest) return;
      if (!res.ok) throw new Error(res.error || 'question matching failed');
      if (!res.matched) {
        showAnswer('<div class="nores">' + esc(res.reply) + '</div>');
        bus.emit('entity:highlight', { ids: [] });
        return;
      }
      input.value = res.preset.code;
      if (askedScope !== res.preset.scope) {
        askedScope = res.preset.scope;
        loadVocabulary();
      }
      showQuestion(res.preset);
      runQuery(res.preset.code);
    } catch (err) {
      if (requestedSource !== source || request !== voiceRequest) return;
      showAnswer('<div class="qerr">' + esc(errorText(err)) + '</div>');
    }
  }

  // %% run an EQL query
  async function runQuery(code) {
    code = (code || '').trim(); if (!code || running) return;
    const requestedSource = source;
    voiceRequest += 1;
    running = true;
    runBtn.textContent = '…';
    try {
      const r = await fetch(requestedSource.runUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: code, scope: askedScope }),
      });
      const response = await ResponseUtil.parseJson(r);
      if (requestedSource !== source) return;
      render(code, response);
    } catch (err) {
      if (requestedSource !== source) return;
      render(code, { ok: false, error: errorText(err) });
    }
    running = false;
    runBtn.textContent = 'Run';
  }

  function render(code, res) {
    // the answered query's own wording is definitive: it replaces whatever label the
    // question was picked by
    if (res.verbalization) {
      showQuestion({ text: code, code: code, verbalization: res.verbalization });
    }
    let html = '<div class="goal">&gt;&gt;&gt; ' + esc(code) + '</div>';
    if (!res.ok) {
      showAnswer(html + '<div class="qerr">' + esc(res.error || 'query failed') + '</div>');
      bus.emit('entity:highlight', { ids: [] });
      return;
    }
    if (!res.count) {
      showAnswer(html + '<div class="nores">No solutions — the query returned nothing.</div>');
      bus.emit('entity:highlight', { ids: [] });
      return;
    }
    html += '<p class="headline"><b>' + res.count + '</b> result' + (res.count === 1 ? '' : 's') +
      (res.more ? ' (truncated)' : '') + '.</p>' + answerTable(res.rows, res.replay);
    showAnswer(html);
    wireReplayButtons();
    bus.emit('entity:highlight', { ids: res.highlight || [] });
  }

  // %% replaying an answered moment
  // The popup is its own viewer window playing the bridge's recording, so the live
  // view in this window keeps running untouched.
  function wireReplayButtons() {
    answerEl.querySelectorAll('.replay-btn').forEach(function (button) {
      button.addEventListener('click', function () {
        openReplay({
          start: parseFloat(button.dataset.start),
          end: parseFloat(button.dataset.end),
        });
      });
    });
  }

  function openReplay(replayWindow) {
    window.open(
      Replay.popupUrl(window.location.pathname, window.location.search, replayWindow),
      'cramera-replay',
      'popup=yes,width=980,height=720'
    );
  }

  // the answer as one table: stable columns, every value coloured by what it is, and
  // a replay button on rows naming a moment there is a recording of — the bridge's while
  // a demo is attached, the loaded scene's own trajectory otherwise
  function answerTable(rows, replay) {
    const table = AnswerTable.of(rows, replay);
    if (!table.columns.length) return '';
    const typed = table.rows.some(function (row) { return row.type; });
    const replayable = table.rows.some(function (row) { return row.replay; });
    let html = '<div class="anstable-wrap"><table class="anstable"><thead><tr>';
    if (typed) html += '<th class="ans-type"></th>';
    table.columns.forEach(function (column) { html += '<th>' + esc(column) + '</th>'; });
    if (replayable) html += '<th class="ans-replay"></th>';
    html += '</tr></thead><tbody>';
    table.rows.forEach(function (row) {
      html += '<tr>' + (typed ? '<td class="ans-type">' + typeTag(row.type) + '</td>' : '');
      row.cells.forEach(function (cell) {
        html += '<td class="ans-' + cell.kind + '">' + esc(cell.text) + '</td>';
      });
      if (replayable) html += '<td class="ans-replay">' + replayButton(row.replay) + '</td>';
      html += '</tr>';
    });
    return html + '</tbody></table></div>';
  }

  function replayButton(replay) {
    if (!replay) return '';
    return '<button class="replay-btn" data-start="' + Number(replay.start) +
      '" data-end="' + Number(replay.end) +
      '" title="replay the demo recording around this moment">▶ replay</button>';
  }

  function typeTag(type) {
    if (!type) return '';
    return '<span class="tag" style="background:' + groupColor(groupOfType(type)) + '">' +
      esc(type) + '</span>';
  }

  runBtn.addEventListener('click', function () { runQuery(input.value); });
  input.addEventListener('keydown', function (e) {
    // the suggestion menu owns the arrows, Tab, Escape and Enter while it is open
    if (suggestions.handledKey(e)) { e.preventDefault(); return; }
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runQuery(input.value); }
  });

  // %% helpers
  const TYPE_GROUP = {
    BenchObject: 'object', ActionEpisode: 'event', Arm: 'robot', Gripper: 'robot',
    Robot: 'robot', JointMotion: 'robot', Point3: 'other',
    Package: 'package', SubPackage: 'subpackage', PythonClass: 'python_class',
  };
  function groupOfType(t) { return TYPE_GROUP[t] || 'other'; }
  const GROUP_COLOR = {
    root: '#e8eefb', subpackage: '#5b8cff', external_class: '#8c9bbd',
    robot: '#ff7a9c', object: '#39d5c8', event: '#b98cff', plan: '#ffb648', package: '#4bd38a', other: '#7f8db0',
    base: '#4bd38a', left_arm: '#ff7a9c', right_arm: '#b98cff', gripper: '#39d5c8', sensor: '#ffb648',
    action: '#b98cff', motion: '#ff7a9c', condition: '#ffb648', attachment: '#39d5c8', other_plan_node: '#7f8db0',
    task: '#ff7a9c', monitor: '#4bd38a', motion_goal: '#5b8cff', motion_end: '#b98cff',
  };
  function groupColor(g) { return GROUP_COLOR[g] || '#5b8cff'; }
  // an Error carries the useful text on .message; anything else is shown as-is
  function errorText(err) { return String((err && err.message) || err); }
  function esc(s) { return String(s).replace(/[&<>]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]; }); }
});
