/* ============================================================================
 * core/query_source.js — where the EQL panel sends its questions.
 *
 * Two things can answer a query: the server, from the recorded scene bundle, and a
 * running demo, from its own live bridge. They speak different URLs, so this decides
 * between them once and hands the panel a pair of endpoints to use.
 *
 * A live bridge serves one demo and has no scene bundle, so ?scene= is only ever
 * appended to the recorded routes.
 * ==========================================================================*/
(function () {
  'use strict';

  const RECORDED_PRESETS = '/api/knowledge';
  const RECORDED_RUN = '/api/eql';
  const RECORDED_VOCABULARY = '/api/eql/vocabulary';
  const RECORDED_MEMBERS = '/api/eql/members';
  const RECORDED_QUESTION = '/api/question';

  function withParameter(url, name, value) {
    return url + (url.indexOf('?') >= 0 ? '&' : '?') + name + '=' + encodeURIComponent(value);
  }

  function recorded() {
    return {
      live: false,
      presetsUrl: window.SceneContext.withScene(RECORDED_PRESETS),
      runUrl: window.SceneContext.withScene(RECORDED_RUN),
      // a recorded scene has one body of knowledge, so it is asked about no scope
      vocabularyUrl: function () {
        return window.SceneContext.withScene(RECORDED_VOCABULARY);
      },
      membersUrl: function (name) {
        return withParameter(
          window.SceneContext.withScene(RECORDED_MEMBERS), 'name', name);
      },
      questionUrl: window.SceneContext.withScene(RECORDED_QUESTION),
    };
  }

  // live carries {on, url} exactly as the robot scene panel publishes it on
  // 'live:changed'; a detached viewer publishes on:false and the url is stale.
  function of(live) {
    if (!live || !live.on || !live.url) return recorded();
    const base = String(live.url).replace(/\/+$/, '');
    return {
      live: true,
      presetsUrl: base + '/presets',
      runUrl: base + '/eql',
      // a demo offers several bodies of knowledge, each with its own variables
      vocabularyUrl: function (scope) {
        const url = base + '/vocabulary';
        return scope ? withParameter(url, 'scope', scope) : url;
      },
      membersUrl: function (name, scope) {
        const url = withParameter(base + '/members', 'name', name);
        return scope ? withParameter(url, 'scope', scope) : url;
      },
      questionUrl: base + '/question',
    };
  }

  window.QuerySource = { of: of };
})();
