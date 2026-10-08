// One-time extraction from the installed, pinned Gradio distribution.
const fs = require('fs');
const vm = require('vm');
const crypto = require('crypto');
const path = require('path');

const assets = process.argv[2];
const source = fs.readFileSync(path.join(assets, 'core-BXBFKSfx.js'), 'utf8');
const start = source.indexOf(',ze=`English`');
const end = source.indexOf(',dt=', start);
if (start < 0 || end < start) throw new Error('Pinned Gradio English dictionary not found');
// Evaluate only the literal dictionary declarations, never the frontend runtime.
const dictionary = vm.runInNewContext('const ' + source.slice(start + 1, end) + '; Z', {}, { timeout: 1000 });
const messages = {};
function flatten(value, prefix = '') {
  for (const [key, item] of Object.entries(value)) {
    const name = prefix ? prefix + '.' + key : key;
    if (typeof item === 'string') messages[name] = item;
    else if (item && typeof item === 'object') flatten(item, name);
    else throw new Error('Unexpected translation value: ' + name);
  }
}
flatten(dictionary);
const locales = [...new Set([...source.matchAll(/\.\/lang\/([A-Za-z0-9-]+)\.json/g)].map(match => match[1]))].sort();
console.log(JSON.stringify({
  source: {
    package: 'gradio', version: '6.29.0', license: 'Apache-2.0',
    url: 'https://github.com/gradio-app/gradio/tree/gradio%406.29.0',
    asset: 'core-BXBFKSfx.js',
    sha256: crypto.createHash('sha256').update(source).digest('hex'),
  },
  locales, messages,
}));
