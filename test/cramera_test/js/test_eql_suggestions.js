'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const BrowserSource = require('./browser_source');

const WEB = path.join(__dirname, '../../../cramera/src/cramera/web');

// %% completion menu fixture
/** A member request held until another source or scope has been selected. */
class PendingMembers {
  constructor() {
    this.promise = new Promise(resolve => { this.resolve = resolve; });
  }
}

/** DOM operations needed to inspect the production completion menu. */
class CompletionElement {
  constructor() {
    this.style = {};
    this.children = [];
    this.value = 'robot.';
    this.selectionStart = this.value.length;
  }

  set innerHTML(value) { this.children = []; }
  appendChild(child) { this.children.push(child); }
  addEventListener() {}
  getBoundingClientRect() { return {left: 0, bottom: 40, width: 300}; }
}

/** Load the real completion matcher and menu against controlled member replies. */
function mountSuggestions(responses) {
  const scope = {innerHeight: 600};
  const input = new CompletionElement();
  const anchor = new CompletionElement();
  let requests = 0;
  new Function('window', BrowserSource.read(path.join(WEB, 'core/completion.js')))(scope);
  new Function('window', 'document', 'Completion',
    BrowserSource.read(path.join(WEB, 'panels/eql/suggestions.js')))(
    scope, {createElement: () => new CompletionElement()}, scope.Completion
  );
  const suggestions = scope.EqlSuggestions.of({
    input, anchor, entries: () => [],
    fetchMembers() { requests += 1; return responses.shift(); },
  });
  return {
    suggestions,
    requestCount: () => requests,
    offeredNames: () => anchor.children[0].children.map(row => row.children[1].textContent),
  };
}

function flush() {
  return new Promise(resolve => setImmediate(resolve));
}

const OLD_MEMBERS = [{name: 'old_field', kind: 'field'}];
const NEW_MEMBERS = [{name: 'new_field', kind: 'field'}];

// %% asynchronous source and scope boundaries
test('forgetting a source prevents its pending members from replacing the new cache', async () => {
  const older = new PendingMembers();
  const newer = new PendingMembers();
  const panel = mountSuggestions([older.promise, newer.promise]);
  panel.suggestions.refresh();
  panel.suggestions.forget();
  panel.suggestions.refresh();
  newer.resolve(NEW_MEMBERS);
  await flush();
  older.resolve(OLD_MEMBERS);
  await flush();
  panel.suggestions.refresh();
  assert.deepEqual(panel.offeredNames(), NEW_MEMBERS.map(member => member.name));
  assert.equal(panel.requestCount(), 2);
});

test('forgetting a scope closes its already visible member menu', async () => {
  const panel = mountSuggestions([Promise.resolve(OLD_MEMBERS)]);
  panel.suggestions.refresh();
  await flush();
  assert.equal(panel.suggestions.isOpen(), true);
  panel.suggestions.forget();
  assert.equal(panel.suggestions.isOpen(), false);
});

test('a forgotten member response cannot reopen or populate the completion menu', async () => {
  const older = new PendingMembers();
  const panel = mountSuggestions([older.promise, Promise.resolve(NEW_MEMBERS)]);
  panel.suggestions.refresh();
  panel.suggestions.forget();
  older.resolve(OLD_MEMBERS);
  await flush();
  assert.equal(panel.suggestions.isOpen(), false);
  panel.suggestions.refresh();
  await flush();
  assert.equal(panel.requestCount(), 2);
  assert.deepEqual(panel.offeredNames(), NEW_MEMBERS.map(member => member.name));
});
