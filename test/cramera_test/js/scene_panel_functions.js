'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const vm = require('node:vm');
const BrowserSource = require('./browser_source');

// %% execute panel functions without a WebGL renderer
class ScenePanelFunctions {
  constructor(state, modules = []) {
    const webDirectory = path.join(__dirname, '../../../cramera/src/cramera/web');
    this.scope = vm.createContext({window: {}, ...state});
    for (const filename of modules) {
      vm.runInContext(BrowserSource.read(path.join(webDirectory, 'core', filename)), this.scope);
    }
    Object.assign(this.scope, this.scope.window);
    const panelPath = path.join(webDirectory, 'panels/robot_scene/panel.js');
    // Keep all sibling declarations together, including duplicate names, so their
    // hoisting matches the mounted panel. Single-line helpers are not needed here.
    const declarations = BrowserSource.read(panelPath).match(/^  function \w+\([^\n]*\) \{\n[\s\S]*?^  \}/gm);
    assert.ok(declarations.length > 0);
    vm.runInContext(declarations.join('\n'), this.scope, {filename: panelPath});
  }
}

module.exports = ScenePanelFunctions;
