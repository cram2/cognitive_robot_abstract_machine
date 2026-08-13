// Unit tests for web/core/response.js (node:test).
// A route with no backend (a static host, or a stale reverse proxy) answers with an
// HTML error page instead of JSON. ResponseUtil.parseJson must report that cleanly
// rather than let a panel blindly JSON.parse the HTML and surface a raw SyntaxError.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  global.window = {};
  new Function(fs.readFileSync(path.join(WEB, 'core/response.js'), 'utf8'))();
}

function fakeResponse(ok, status, jsonPayload) {
  return {
    ok: ok,
    status: status,
    json: function () {
      return jsonPayload === undefined
        ? Promise.reject(new SyntaxError('JSON.parse: unexpected character at line 1 column 1 of the JSON data'))
        : Promise.resolve(jsonPayload);
    },
  };
}

test('parseJson() resolves the parsed body for an ok response', async function () {
  load();
  const payload = await window.ResponseUtil.parseJson(fakeResponse(true, 200, { ok: true }));
  assert.deepStrictEqual(payload, { ok: true });
});

test('parseJson() throws a clean error for a 404 instead of parsing its HTML body', function () {
  load();
  assert.throws(
    function () { window.ResponseUtil.parseJson(fakeResponse(false, 404)); },
    /no server for this route \(HTTP 404\)/
  );
});

test('parseJson() throws a clean error for any non-ok status', function () {
  load();
  assert.throws(
    function () { window.ResponseUtil.parseJson(fakeResponse(false, 500)); },
    /no server for this route \(HTTP 500\)/
  );
});
