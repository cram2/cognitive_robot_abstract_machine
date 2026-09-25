'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const ScenePanelFunctions = require('./scene_panel_functions');

// %% selection controls keep scene labels as text
class SceneChoice {
  constructor() {
    this.children = [];
    this.style = {};
    this.innerHTML = '';
    this.listeners = {};
  }

  appendChild(child) { this.children.push(child); }
  replaceChildren(...children) { this.children = children; }
  addEventListener(name, listener) { this.listeners[name] = listener; }
}

test('scene, robot and environment labels preserve markup characters as text', () => {
  const elements = {};
  for (const name of ['robot-select', 'environment-select', 'robot-picker',
    'environment-picker', 'scene-name-select', 'scene-name-picker']) {
    elements[name] = new SceneChoice();
  }
  const first = {name: 'first', robot: 'Robot "<&>', environment: 'Room "<&>',
    task: '</option><img src=x onerror=alert(1)>'};
  const second = {name: 'second', robot: 'Another robot', environment: null, task: 'Second task'};
  const panel = new ScenePanelFunctions({
    $: name => elements[name], LIVE_SCENE_NAME: '__live__',
    document: {createElement() { return new SceneChoice(); }},
    window: {location: {search: ''}},
  }, ['scene_picker.js', 'recording-mode.js']);

  panel.scope.wireScenePickers([first, second], first.name);

  assert.equal(elements['scene-name-select'].children[0].textContent, first.task);
  assert.equal(elements['scene-name-select'].children[0].value, first.name);
  assert.equal(elements['scene-name-select'].children[0].selected, true);
  assert.equal(elements['robot-select'].children[0].textContent, first.robot);
  assert.equal(elements['environment-select'].children[0].textContent, first.environment);
  for (const name of ['scene-name-select', 'robot-select', 'environment-select']) {
    assert.equal(elements[name].innerHTML, '');
  }
});
