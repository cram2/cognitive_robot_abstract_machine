/* ============================================================================
 * core/question_display.js — the asked question, shown big where the query text
 * box used to be.
 *
 * What is asked is a question in English, not EQL source: the display shows the
 * query's verbalization (krrood's own <span> colouring, display text escaped
 * server-side) and falls back to the preset's plain label when nothing worded it.
 * Pure string building, so it is unit-testable without a DOM.
 * ==========================================================================*/
(function () {
  'use strict';

  function esc(s) {
    return String(s).replace(/[&<>]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c];
    });
  }

  // the resting state: nothing asked yet, say how to ask
  function hint(text) {
    return '<span class="question-hint">' + esc(text) + '</span>';
  }

  // question = {text, verbalization} — a preset payload, or anything shaped like one.
  // The verbalization's html is already-escaped coloured markup; the plain label is
  // ours to escape.
  function markup(question) {
    if (!question) return '';
    if (question.verbalization && question.verbalization.html) {
      return question.verbalization.html;
    }
    return esc(question.text || '');
  }

  window.QuestionDisplay = { markup: markup, hint: hint };
})();
