const BrowserSource = require('./browser_source');
// Unit tests for web/core/question_display.js (node:test).
// The asked question is shown big, in English, where the query text box used to be:
// the verbalized wording when the server worded it, the preset's plain label when not.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  global.window = {};
  new Function(BrowserSource.read(path.join(WEB, 'core/question_display.js')))();
  return window.QuestionDisplay;
}

const WORDED = {
  text: 'which robot is this?',
  code: 'the(entity(robot))',
  verbalization: {
    text: 'The Robot.',
    html: '<span style="color:#5b8cff">The</span> <span style="color:#ff7a9c">Robot</span>.',
  },
};

test('a worded question shows its coloured verbalization as-is', function () {
  // the markup is krrood's own colouring, escaped server-side, so it goes in unchanged
  assert.strictEqual(load().markup(WORDED), WORDED.verbalization.html);
});

test('an unworded question falls back to its plain label', function () {
  const markup = load().markup({ text: 'success rate per shape', code: 'x' });
  assert.strictEqual(markup, 'success rate per shape');
});

test('a wording with no markup falls back to the label too', function () {
  const markup = load().markup({ text: 'label', verbalization: { text: 'words', html: '' } });
  assert.strictEqual(markup, 'label');
});

test('a label cannot smuggle markup into the page', function () {
  const markup = load().markup({ text: '<img src=x onerror=alert(1)>' });
  assert.strictEqual(markup, '&lt;img src=x onerror=alert(1)&gt;');
});

test('nothing asked renders nothing', function () {
  assert.strictEqual(load().markup(null), '');
});

test('the hint is escaped text in its own styleable span', function () {
  assert.strictEqual(
    load().hint('Pick a question <below>'),
    '<span class="question-hint">Pick a question &lt;below&gt;</span>'
  );
});
