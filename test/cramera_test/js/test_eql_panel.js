const BrowserSource = require('./browser_source');
// Demo tests for panels/eql/panel.js (node:test): asking a question end to end.
//
// panel.js is loaded with its free variables bound as explicit function parameters
// (the test_graph_panel.js pattern). QuestionDisplay, PresetGroups, AnswerTable,
// ResponseUtil and SceneContext are the real core modules, so the flow a viewer
// drives — presets load, a preset is picked, the question shows big in English, the
// query runs, the answer renders — is exercised against the real string building,
// with only the DOM and fetch stubbed.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');
const SOURCE = BrowserSource.read(path.join(WEB, 'panels/eql/panel.js'));

function loadCore(name, scope) {
  new Function('window', BrowserSource.read(path.join(WEB, name)))(scope);
}

function coreModules(recognizer) {
  const scope = { location: { search: '' } };
  if (recognizer) scope.SpeechRecognition = recognizer;
  loadCore('core/scene.js', scope);
  loadCore('core/query_source.js', scope);
  loadCore('core/question_display.js', scope);
  loadCore('core/preset_groups.js', scope);
  loadCore('core/answer_table.js', scope);
  loadCore('core/response.js', scope);
  loadCore('core/voice.js', scope);
  loadCore('core/folding.js', scope);
  return scope;
}

// a scripted recognizer standing in for the browser's SpeechRecognition
function recognizerClass() {
  const instances = [];
  function ScriptedRecognizer() { instances.push(this); }
  ScriptedRecognizer.prototype.start = function () {};
  ScriptedRecognizer.prototype.stop = function () { this.onend(); };
  ScriptedRecognizer.instances = instances;
  return ScriptedRecognizer;
}

function speak(Recognizer, text) {
  const recognition = Recognizer.instances[Recognizer.instances.length - 1];
  recognition.onresult({ results: [[{ transcript: text }]] });
  recognition.onend();
}

function flush() {
  return new Promise(function (resolve) { setTimeout(resolve, 0); });
}

// The folding fixture waits for the same pending fetch/DOM work as the query fixture.
const tick = flush;

// %% a miniature DOM: just what the panel reaches for
function makeElement(tag) {
  const listeners = {};
  return {
    tagName: tag || 'div',
    innerHTML: '',
    textContent: '',
    title: '',
    className: '',
    value: '',
    children: [],
    classList: {
      classes: new Set(),
      add(c) { this.classes.add(c); },
      remove(c) { this.classes.delete(c); },
      toggle(c, on) { if (on) this.classes.add(c); else this.classes.delete(c); },
      contains(c) { return this.classes.has(c); },
    },
    appendChild(child) { child.parentNode = this; this.children.push(child); return child; },
    scrolledIntoView: 0,
    scrollIntoView() { this.scrolledIntoView += 1; },
    addEventListener(event, cb) { (listeners[event] = listeners[event] || []).push(cb); },
    click() { (listeners.click || []).forEach(function (cb) { cb(); }); },
    dispatch(event, detail) { (listeners[event] || []).forEach(function (cb) { cb(detail); }); },
    querySelectorAll() { return []; },
  };
}

function makeRoot() {
  const byId = {
    '#knowledge-status': makeElement('span'),
    '#answer': makeElement(),
    '#query-input': makeElement('textarea'),
    '#query-run': makeElement('button'),
    '#question': makeElement(),
    '#voice-ask': makeElement('button'),
    '#presets': makeElement(),
  };
  return {
    innerHTML: '',
    children: Object.values(byId),
    querySelector(selector) { return byId[selector]; },
    part(selector) { return byId[selector]; },
  };
}

function makeBus() {
  const handlers = {};
  const emitted = [];
  return {
    on(event, cb) { (handlers[event] = handlers[event] || []).push(cb); },
    emit(event, payload) {
      emitted.push({ event: event, payload: payload });
      (handlers[event] || []).forEach(function (cb) { cb(payload); });
    },
    emitted: emitted,
  };
}

function makeFetch(routes, requests) {
  return async function fetch(url, options) {
    requests.push({ url: url, options: options });
    const answer = routes[url.split('?')[0]];
    if (!answer) return { ok: false, status: 404 };
    return { ok: true, status: 200, json: async function () { return answer; } };
  };
}

// every button the presets area currently shows, at any depth
function presetButtons(presetsEl) {
  const buttons = [];
  (function walk(children) {
    children.forEach(function (child) {
      if (child.className === 'preset' || child.className === 'preset unavailable') {
        buttons.push(child);
      }
      walk(child.children);
    });
  })(presetsEl.children);
  return buttons;
}

// %% the harness
const WORDED_PRESET = {
  text: 'which robot is this?',
  code: 'the(entity(robot))',
  requires_live: false,
  scope: 'current_state',
  verbalization: {
    text: 'The Robot.',
    html: '<span style="color:#ff7a9c">The Robot</span>.',
  },
};

const UNWORDED_PRESET = {
  text: 'success rate per shape',
  code: 'set_of(shape.name)',
  requires_live: false,
  scope: 'current_state',
  verbalization: null,
};

const ANSWER = {
  ok: true,
  kind: 'entities',
  rows: [{ __entity__: 'tracy', __type__: 'Robot' }],
  count: 1,
  more: false,
  highlight: ['tracy'],
  verbalization: {
    text: 'The one Robot there is.',
    html: '<span style="color:#5b8cff">The one</span> <span style="color:#ff7a9c">Robot</span> there is.',
  },
};

// what a browser remembers between pages, for as long as one test runs
function remembering() {
  const data = {};
  return {
    getItem: function (key) { return key in data ? data[key] : null; },
    setItem: function (key, value) { data[key] = value; },
  };
}

function mountPanel(overrides, recognizer) {
  const core = coreModules(recognizer);
  const root = makeRoot();
  const bus = makeBus();
  const requests = [];
  const routes = Object.assign(
    {
      '/api/knowledge': {
        ok: true,
        status: 'EQL ready',
        presets: [WORDED_PRESET, UNWORDED_PRESET],
        details: {},
      },
      '/api/eql': ANSWER,
    },
    overrides || {}
  );
  let panelFactory = null;
  let completionOptions = null;
  const define = function (name, factory) { panelFactory = factory; };
  new Function(
    'Panels', 'SceneContext', 'QuerySource', 'QuestionDisplay', 'PresetGroups',
    'AnswerTable', 'ResponseUtil', 'VoiceCapture', 'EqlSuggestions', 'Replay',
    'Folding', 'fetch', 'window', 'document',
    SOURCE
  )(
    { define: define }, core.SceneContext, core.QuerySource, core.QuestionDisplay,
    core.PresetGroups, core.AnswerTable, core.ResponseUtil, core.VoiceCapture,
    { of(options) { completionOptions = options; return { forget() {}, handledKey() { return false; } }; } },
    { popupUrl() { return ''; } },
    core.Folding,
    makeFetch(routes, requests),
    { location: { pathname: '/', search: '' }, open() {}, localStorage: remembering() },
    { createElement: makeElement }
  );
  panelFactory(root, bus);
  return { root: root, bus: bus, requests: requests, completion: completionOptions };
}

// %% the flow a viewer drives
test('before anything is asked the display shows how to ask', async function () {
  const panel = mountPanel();
  await flush();
  const question = panel.root.part('#question').innerHTML;
  assert.ok(question.indexOf('question-hint') >= 0, question);
});

test('picking a preset shows its wording big and runs its query', async function () {
  const panel = mountPanel();
  await flush(); await flush();

  const button = presetButtons(panel.root.part('#presets'))[0];
  assert.strictEqual(button.textContent, WORDED_PRESET.text);
  button.click();

  // the picked query fills the bar, and the question is on display under it
  // before the answer arrives
  assert.strictEqual(panel.root.part('#query-input').value, WORDED_PRESET.code);
  assert.strictEqual(panel.root.part('#question').innerHTML, WORDED_PRESET.verbalization.html);
  assert.strictEqual(panel.root.part('#question').title, WORDED_PRESET.code);

  await flush(); await flush();

  const run = panel.requests.find(function (r) { return r.url === '/api/eql'; });
  assert.deepStrictEqual(JSON.parse(run.options.body), {
    code: WORDED_PRESET.code,
    scope: 'current_state',
  });
  const answer = panel.root.part('#answer').innerHTML;
  assert.ok(answer.indexOf('<b>1</b> result') >= 0, answer);
  assert.ok(answer.indexOf('tracy') >= 0, answer);
});

test('the answered query\'s own wording replaces the label it was picked by', async function () {
  const panel = mountPanel();
  await flush(); await flush();

  presetButtons(panel.root.part('#presets'))[1].click();
  // unworded until the answer arrives: the plain label stands in
  assert.strictEqual(panel.root.part('#question').innerHTML, UNWORDED_PRESET.text);

  await flush(); await flush();

  assert.strictEqual(panel.root.part('#question').innerHTML, ANSWER.verbalization.html);
});

test('the answer highlights what it names', async function () {
  const panel = mountPanel();
  await flush(); await flush();

  presetButtons(panel.root.part('#presets'))[0].click();
  await flush(); await flush();

  const highlight = panel.bus.emitted.filter(function (e) { return e.event === 'entity:highlight'; }).pop();
  assert.deepStrictEqual(highlight.payload.ids, ['tracy']);
});

test('a typed query runs from the bar and its wording appears under it', async function () {
  const panel = mountPanel();
  await flush(); await flush();

  panel.root.part('#query-input').value = 'the(entity(robot))';
  panel.root.part('#query-run').click();
  await flush(); await flush();

  const run = panel.requests.find(function (r) { return r.url === '/api/eql'; });
  assert.deepStrictEqual(JSON.parse(run.options.body), {
    code: 'the(entity(robot))',
    scope: null,
  });
  // the answered query's own wording shows under the bar
  assert.strictEqual(panel.root.part('#question').innerHTML, ANSWER.verbalization.html);
  assert.ok(panel.root.part('#answer').innerHTML.indexOf('tracy') >= 0);
});

test('a failed query is reported in the answer area, not swallowed', async function () {
  const panel = mountPanel({ '/api/eql': { ok: false, error: 'NameError: shape' } });
  await flush(); await flush();

  presetButtons(panel.root.part('#presets'))[0].click();
  await flush(); await flush();

  const answer = panel.root.part('#answer').innerHTML;
  assert.ok(answer.indexOf('NameError: shape') >= 0, answer);
});

// %% asking by voice — the full demo flow, with a scripted microphone
const MATCHED = { ok: true, matched: true, similarity: 95.0, preset: WORDED_PRESET };
const UNMATCHED = {
  ok: true, matched: false, similarity: 40.0,
  reply: 'Sorry, I cannot answer that question.',
};

test('a spoken question that matches runs as if its button had been clicked', async function () {
  const Recognizer = recognizerClass();
  const panel = mountPanel({ '/api/question': MATCHED }, Recognizer);
  await flush(); await flush();

  panel.root.part('#voice-ask').click();
  assert.ok(panel.root.part('#question').innerHTML.indexOf('Listening…') >= 0);
  speak(Recognizer, 'which robot is this');
  await flush(); await flush(); await flush();

  // the transcript went over the bus, for any consumer
  const transcript = panel.bus.emitted.find(function (e) { return e.event === 'voice:transcript'; });
  assert.deepStrictEqual(transcript.payload, { text: 'which robot is this' });

  // the default consumer asked the matcher, then ran the recognized preset
  const asked = panel.requests.find(function (r) { return r.url === '/api/question'; });
  assert.deepStrictEqual(JSON.parse(asked.options.body), { text: 'which robot is this' });
  const run = panel.requests.find(function (r) { return r.url === '/api/eql'; });
  assert.deepStrictEqual(JSON.parse(run.options.body), {
    code: WORDED_PRESET.code,
    scope: WORDED_PRESET.scope,
  });

  // as if clicked: the bar carries the recognized preset's code,
  // the question is on display, and its answer rendered
  assert.strictEqual(panel.root.part('#query-input').value, WORDED_PRESET.code);
  assert.strictEqual(panel.root.part('#question').innerHTML, ANSWER.verbalization.html);
  assert.ok(panel.root.part('#answer').innerHTML.indexOf('tracy') >= 0);
});

test('a spoken question nothing answers gets the sorry reply', async function () {
  const Recognizer = recognizerClass();
  const panel = mountPanel({ '/api/question': UNMATCHED }, Recognizer);
  await flush(); await flush();

  panel.root.part('#voice-ask').click();
  speak(Recognizer, 'what is the weather like today');
  await flush(); await flush();

  // the reply is the server's own words, and no query ran
  const answer = panel.root.part('#answer').innerHTML;
  assert.ok(answer.indexOf(UNMATCHED.reply) >= 0, answer);
  assert.ok(!panel.requests.some(function (r) { return r.url === '/api/eql'; }));
  // the question display still says what was asked
  const question = panel.root.part('#question').innerHTML;
  assert.ok(question.indexOf('what is the weather like today') >= 0, question);
});

test('a browser without speech recognition disables the button', async function () {
  const panel = mountPanel();
  await flush();

  assert.strictEqual(panel.root.part('#voice-ask').disabled, true);
});

test('any consumer can feed a transcript over the bus', async function () {
  // the mic button is one producer; the contract is the bus event
  const panel = mountPanel({ '/api/question': MATCHED });
  await flush(); await flush();

  panel.bus.emit('voice:transcript', { text: 'which robot is this' });
  await flush(); await flush(); await flush();

  const run = panel.requests.find(function (r) { return r.url === '/api/eql'; });
  assert.ok(run, 'the matched preset ran');
});

// %% the answer sits under everything asked, so it is scrolled to when it arrives
test('an answered query is scrolled to', async function () {
  const panel = mountPanel();
  await flush(); await flush();
  const answer = panel.root.part('#answer');
  assert.strictEqual(answer.scrolledIntoView, 0);

  presetButtons(panel.root.part('#presets'))[0].click();
  await flush(); await flush();

  assert.strictEqual(answer.scrolledIntoView, 1);
});

test('a spoken question nothing answers is scrolled to like any other', async function () {
  const Recognizer = recognizerClass();
  const panel = mountPanel({ '/api/question': UNMATCHED }, Recognizer);
  await flush(); await flush();
  const answer = panel.root.part('#answer');
  assert.strictEqual(answer.scrolledIntoView, 0);

  panel.root.part('#voice-ask').click();
  speak(Recognizer, 'what is the weather like today');
  await flush(); await flush();

  assert.strictEqual(answer.scrolledIntoView, 1);
});

test('a described entity is shown where the answer is, without scrolling to it', async function () {
  const panel = mountPanel();
  await flush(); await flush();

  panel.bus.emit('entity:select', {
    id: 'tracy', detail: { group: 'robot', label: 'Tracy', lines: [] }, relations: [],
  });

  const answer = panel.root.part('#answer');
  assert.ok(answer.innerHTML.indexOf('Tracy') >= 0, answer.innerHTML);
  assert.strictEqual(answer.scrolledIntoView, 0);
});


// %% folding a group of questions away

const GROUPED_KNOWLEDGE = {
  ok: true,
  status: 'EQL ready',
  presets: [
    { text: 'which robot is this?', code: 'the(entity(robot))', scope: 'current_state' },
    { text: 'give me all pick up events', code: 'an(entity(event))', scope: 'detected_events' },
  ],
  scopes: [
    { name: 'current_state', label: 'Current State Queries', variables: [] },
    { name: 'detected_events', label: 'Detected Events Queries', variables: [] },
  ],
  details: {},
};

function headings(root) {
  const found = [];
  (function walk(children) {
    children.forEach(function (child) {
      if (child.className === 'preset-group') found.push(child);
      walk(child.children);
    });
  })(root.children);
  return found;
}

function foldButtonOf(heading) {
  return heading.children.filter(function (child) {
    return child.className.indexOf('lp-fold') >= 0;
  })[0];
}

test('every group of questions offers to fold away', async function () {
  const panel = mountPanel({ '/api/knowledge': GROUPED_KNOWLEDGE });
  await tick();

  const found = headings(panel.root);

  assert.strictEqual(found.length, 2);
  found.forEach(function (heading) { assert.ok(foldButtonOf(heading)); });
});

test('a group starts open and folds when its heading is clicked', async function () {
  const panel = mountPanel({ '/api/knowledge': GROUPED_KNOWLEDGE });
  await tick();
  const heading = headings(panel.root)[1];
  const row = heading.parentNode.children[heading.parentNode.children.indexOf(heading) + 1];

  assert.ok(!row.classList.contains('folded'));
  heading.dispatch('click', { preventDefault: function () {} });

  assert.ok(row.classList.contains('folded'));
  assert.strictEqual(foldButtonOf(heading).textContent, '\u25b8');
});

test('folding one group leaves the others open', async function () {
  const panel = mountPanel({ '/api/knowledge': GROUPED_KNOWLEDGE });
  await tick();
  const found = headings(panel.root);
  const rowOf = function (heading) {
    const siblings = heading.parentNode.children;
    return siblings[siblings.indexOf(heading) + 1];
  };

  found[1].dispatch('click', { preventDefault: function () {} });

  assert.ok(!rowOf(found[0]).classList.contains('folded'));
  assert.ok(rowOf(found[1]).classList.contains('folded'));
});

// %% replies belong to the source that was asked
/** Release a response only after the answering source has changed. */
class PendingAnswer {
  constructor() {
    this.promise = new Promise((resolve, reject) => {
      this.resolve = resolve;
      this.reject = reject;
    });
  }
}

const LIVE_PRESETS = {ok: true, title: 'running demo', presets: [UNWORDED_PRESET]};
const RECORDED_VOCABULARY = {ok: true, entries: [{name: 'recorded_robot'}]};

function attach(panel, on, url = 'http://bridge') {
  panel.bus.emit('live:changed', {on, url});
}

function runTyped(panel, code) {
  panel.root.part('#query-input').value = code;
  panel.root.part('#query-run').click();
}

test('late live presets cannot replace the recorded source after detach', async function () {
  const pending = new PendingAnswer();
  const panel = mountPanel({'http://bridge/presets': pending.promise});
  await flush();
  attach(panel, true);
  attach(panel, false);
  const recordedStatus = panel.root.part('#knowledge-status').textContent;
  pending.resolve(LIVE_PRESETS);
  await flush();
  assert.strictEqual(panel.root.part('#knowledge-status').textContent, recordedStatus);
});

test('a rejected old preset request cannot replace a new bridge status', async function () {
  const pending = new PendingAnswer();
  const panel = mountPanel({
    'http://bridge/presets': pending.promise,
    'http://replacement/presets': LIVE_PRESETS,
  });
  await flush();
  attach(panel, true);
  attach(panel, true, 'http://replacement');
  await flush();
  const replacementStatus = panel.root.part('#knowledge-status').textContent;
  pending.reject(new Error('old bridge offline'));
  await flush();
  assert.strictEqual(panel.root.part('#knowledge-status').textContent, replacementStatus);
});

for (const fails of [false, true]) {
  test('an old vocabulary ' + (fails ? 'failure' : 'answer') + ' cannot replace current completions', async function () {
    const pending = new PendingAnswer();
    const panel = mountPanel({
      'http://bridge/presets': LIVE_PRESETS,
      'http://bridge/vocabulary': pending.promise,
      '/api/eql/vocabulary': RECORDED_VOCABULARY,
    });
    await flush();
    attach(panel, true);
    await flush();
    attach(panel, false);
    await flush();
    if (fails) pending.reject(new Error('old vocabulary unavailable'));
    else pending.resolve({ok: true, entries: [{name: 'old_live_object'}]});
    await flush();
    assert.deepStrictEqual(panel.completion.entries(), RECORDED_VOCABULARY.entries);
  });
}

test('changing the source releases Run while the old query remains pending', async function () {
  const pending = new PendingAnswer();
  const panel = mountPanel({
    'http://bridge/presets': LIVE_PRESETS,
    'http://bridge/eql': pending.promise,
  });
  await flush();
  attach(panel, true);
  await flush();
  runTyped(panel, 'live_query');
  attach(panel, false);
  runTyped(panel, 'recorded_query');
  await flush();
  assert.deepStrictEqual(panel.requests.filter(request => request.options?.method === 'POST')
    .map(request => request.url), ['http://bridge/eql', '/api/eql']);
  pending.resolve(ANSWER);
  await flush();
});

for (const fails of [false, true]) {
  test('an old query ' + (fails ? 'failure' : 'answer') + ' cannot replace the detached answer', async function () {
    const pending = new PendingAnswer();
    const panel = mountPanel({
      'http://bridge/presets': LIVE_PRESETS,
      'http://bridge/eql': pending.promise,
    });
    await flush();
    attach(panel, true);
    await flush();
    runTyped(panel, 'live_query');
    attach(panel, false);
    const detachedAnswer = panel.root.part('#answer').innerHTML;
    if (fails) pending.reject(new Error('old query failed'));
    else pending.resolve(ANSWER);
    await flush();
    assert.strictEqual(panel.root.part('#answer').innerHTML, detachedAnswer);
  });
}

test('a voice match from an old source is never executed on its replacement', async function () {
  const pending = new PendingAnswer();
  const panel = mountPanel({
    'http://bridge/presets': LIVE_PRESETS,
    'http://bridge/question': pending.promise,
    'http://replacement/presets': LIVE_PRESETS,
    'http://replacement/eql': ANSWER,
  });
  await flush();
  attach(panel, true);
  await flush();
  panel.bus.emit('voice:transcript', {text: 'which robot is this'});
  attach(panel, true, 'http://replacement');
  await flush();
  pending.resolve({ok: true, matched: true, preset: WORDED_PRESET});
  await flush();
  assert.deepStrictEqual(panel.requests.filter(request => request.url.endsWith('/eql')), []);
});

test('a stale voice error cannot replace the current answer', async function () {
  const pending = new PendingAnswer();
  const panel = mountPanel({
    'http://bridge/presets': LIVE_PRESETS,
    'http://bridge/question': pending.promise,
  });
  await flush();
  attach(panel, true);
  await flush();
  panel.bus.emit('voice:transcript', {text: 'which robot is this'});
  attach(panel, false);
  runTyped(panel, WORDED_PRESET.code);
  await flush();
  const recordedAnswer = panel.root.part('#answer').innerHTML;
  pending.reject(new Error('old voice question failed'));
  await flush();
  assert.strictEqual(panel.root.part('#answer').innerHTML, recordedAnswer);
});

test('a selected query scope does not leak into the replacement source', async function () {
  const panel = mountPanel({
    'http://bridge/presets': {ok: true, title: 'memory', presets: [
      {...UNWORDED_PRESET, scope: 'episodic_memory'},
    ]},
    'http://bridge/eql': ANSWER,
  });
  await flush();
  attach(panel, true);
  await flush();
  presetButtons(panel.root.part('#presets')).at(-1).click();
  await flush();
  attach(panel, false);
  runTyped(panel, WORDED_PRESET.code);
  await flush();
  const request = panel.requests.filter(request => request.url === '/api/eql').at(-1);
  assert.strictEqual(JSON.parse(request.options.body).scope, null);
});
