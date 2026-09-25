const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

/** Preserve a browser source's filename when a fixture evaluates it dynamically. */
class BrowserSource {
  static read(filename) {
    const source = fs.readFileSync(filename, 'utf8');
    if (path.extname(filename) !== '.js') return source;
    return source + '\n//# sourceURL=' + pathToFileURL(filename).href + '\n';
  }
}

module.exports = BrowserSource;
