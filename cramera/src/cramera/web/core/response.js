/* ============================================================================
 * core/response.js — turns a fetch Response into parsed JSON or a clean error.
 *
 * A host with no matching backend route (a static site, or a stale reverse
 * proxy) answers with an HTML error page instead of JSON, often with a non-2xx
 * status. Parsing that page as JSON throws a raw "SyntaxError: JSON.parse:
 * unexpected character…" that tells a user nothing; checking the status first
 * lets panels report "no server for this route" instead.
 * ==========================================================================*/
(function () {
  'use strict';

  function parseJson(response) {
    if (!response.ok) {
      throw new Error('no server for this route (HTTP ' + response.status + ')');
    }
    return response.json();
  }

  window.ResponseUtil = { parseJson: parseJson };
})();
