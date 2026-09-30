const BrowserSource = require('./browser_source');
// Unit tests for web/core/voice.js (node:test): one press, one spoken question, as text.
// The recognizer constructor is injectable, so the capture's state machine runs against
// a scripted stand-in instead of a browser microphone.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load(scope) {
  const bound = scope || {};
  new Function('window', BrowserSource.read(path.join(WEB, 'core/voice.js')))(bound);
  return bound.VoiceCapture;
}

// %% a scripted recognizer standing in for the browser's SpeechRecognition
function recognizerClass() {
  const instances = [];
  function ScriptedRecognizer() {
    this.started = false;
    this.stopRequested = false;
    instances.push(this);
  }
  ScriptedRecognizer.prototype.start = function () { this.started = true; };
  ScriptedRecognizer.prototype.stop = function () {
    this.stopRequested = true;
    this.onend();
  };
  ScriptedRecognizer.instances = instances;
  return ScriptedRecognizer;
}

function makeCapture(Recognizer) {
  const seen = { transcripts: [], states: [], errors: [] };
  const capture = load().create({
    recognizer: Recognizer,
    onTranscript: function (text) { seen.transcripts.push(text); },
    onState: function (listening) { seen.states.push(listening); },
    onError: function (message) { seen.errors.push(message); },
  });
  return { capture: capture, seen: seen };
}

function speak(recognition, text) {
  recognition.onresult({ results: [[{ transcript: text }]] });
  recognition.onend();
}

// %% capturing one question
test('one press captures one transcript and closes the microphone', function () {
  const Recognizer = recognizerClass();
  const made = makeCapture(Recognizer);

  assert.strictEqual(made.capture.start(), true);
  assert.strictEqual(made.capture.listening, true);
  const recognition = Recognizer.instances[0];
  assert.strictEqual(recognition.started, true);

  speak(recognition, 'which robot is this');

  assert.deepStrictEqual(made.seen.transcripts, ['which robot is this']);
  assert.strictEqual(made.capture.listening, false);
  assert.deepStrictEqual(made.seen.states, [true, false]);
});

test('the recognizer is configured for one utterance, final results only', function () {
  const Recognizer = recognizerClass();
  const made = makeCapture(Recognizer);
  made.capture.start();

  const recognition = Recognizer.instances[0];
  assert.strictEqual(recognition.continuous, false);
  assert.strictEqual(recognition.interimResults, false);
  assert.strictEqual(recognition.maxAlternatives, 1);
});

test('starting again while listening is refused', function () {
  const Recognizer = recognizerClass();
  const made = makeCapture(Recognizer);
  made.capture.start();

  assert.strictEqual(made.capture.start(), false);
  assert.strictEqual(Recognizer.instances.length, 1);
});

test('each capture is a fresh recognition, so asking again works', function () {
  const Recognizer = recognizerClass();
  const made = makeCapture(Recognizer);

  made.capture.start();
  speak(Recognizer.instances[0], 'first question');
  made.capture.start();
  speak(Recognizer.instances[1], 'second question');

  assert.deepStrictEqual(made.seen.transcripts, ['first question', 'second question']);
  assert.strictEqual(Recognizer.instances.length, 2);
});

// %% stopping and failing
test('stop asks the in-flight recognition to wrap up', function () {
  const Recognizer = recognizerClass();
  const made = makeCapture(Recognizer);
  made.capture.start();

  made.capture.stop();

  assert.strictEqual(Recognizer.instances[0].stopRequested, true);
  assert.strictEqual(made.capture.listening, false);
});

test('stopping while not listening is a no-op', function () {
  const made = makeCapture(recognizerClass());
  made.capture.stop();
  assert.deepStrictEqual(made.seen.states, []);
});

test('a recognition error is reported and listening still ends', function () {
  const Recognizer = recognizerClass();
  const made = makeCapture(Recognizer);
  made.capture.start();

  const recognition = Recognizer.instances[0];
  recognition.onerror({ error: 'not-allowed' });
  recognition.onend();

  assert.deepStrictEqual(made.seen.errors, ['not-allowed']);
  assert.strictEqual(made.capture.listening, false);
});

// %% browser support
test('a browser without speech recognition is unsupported and start is refused', function () {
  const made = makeCapture(null);
  assert.strictEqual(made.capture.supported, false);
  assert.strictEqual(made.capture.start(), false);
  assert.deepStrictEqual(made.seen.states, []);
});

test('the browser recognizer is picked up, prefixed or not', function () {
  const Recognizer = recognizerClass();
  assert.strictEqual(
    load({ SpeechRecognition: Recognizer }).create({}).supported, true);
  assert.strictEqual(
    load({ webkitSpeechRecognition: Recognizer }).create({}).supported, true);
  assert.strictEqual(load({}).create({}).supported, false);
});
