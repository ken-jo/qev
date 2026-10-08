const { chromium } = require('playwright');
const path = require('path');
const assert = require('assert/strict');
const fs = require('fs');
const presets = JSON.parse(fs.readFileSync(path.join(__dirname, '../../src/qev/playground/presets.json')));
const labels = {
  support: 'Text / Route a support request', policy: 'Text / Apply an order policy',
  uncertain: 'Text / Check insufficient evidence', photo: 'Image / Choose the material',
  photo_score: 'Image / Score visibility', photo_truth: 'Image / Judge a proposition',
  photo_policy: 'Image + text / Apply a sorting policy',
};

let browser;
let page;
let phase = 'launch';
(async () => {
  browser = await chromium.launch({ headless: true });
  page = await browser.newPage({ locale: 'ko-KR', viewport: { width: 1440, height: 1100 } });
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto('http://127.0.0.1:7863/', { waitUntil: 'networkidle' });
  await page.getByRole('heading', { name: 'QEV', exact: true }).waitFor();
  const text = await page.locator('body').innerText();
  assert(!/[\uac00-\ud7af]/.test(text), 'English UI must not follow Korean browser locale');
  assert(text.includes('Drop Image Here') && text.includes('Click to Upload'));
  assert(!text.includes('2048'));
  phase = 'initial sample images';
  await page.waitForFunction(() => [...document.querySelectorAll('.gallery-item img')].every(image => image.complete && image.naturalWidth > 0));
  assert.equal(await page.locator('.gallery-item').count(), 6);
  await page.screenshot({ path: path.join(__dirname, 'playground-desktop.png'), fullPage: true });
  const checkedPresets = [];
  for (const [key, label] of Object.entries(labels)) {
    phase = 'preset ' + key;
    await page.getByRole('combobox').first().click();
    await page.getByRole('option').filter({ hasText: label }).click();
    const question = presets[key].questions[0];
    await page.waitForFunction(expected => [...document.querySelectorAll('textarea')].some(input => input.value === expected), question.instructions);
    await page.waitForFunction(kind => document.querySelector('input[type=radio]:checked')?.value === kind, question.type);
    checkedPresets.push(key);
  }
  for (const kind of ['score', 'noul', 'choice']) {
    phase = 'decision type ' + kind;
    await page.locator('input[type=radio][value="' + kind + '"]').check();
    const expected = { choice: 'yes | The evidence meets the criterion.', score: 'Low: the criterion is not met.', noul: 'false | The proposition is false.' }[kind];
    await page.waitForFunction(prefix => [...document.querySelectorAll('textarea')].some(input => input.value.startsWith(prefix)), expected);
  }
  const selectedSamples = new Set();
  for (const index of [1, 2, 3, 4, 5, 0]) {
    phase = 'sample ' + index;
    await page.locator('.gallery-item').nth(index).click();
    const filename = 'sample-' + String(index + 1).padStart(2, '0') + '.jpg';
    await page.waitForFunction(expected => [...document.querySelectorAll('img')].some(image => !image.closest('.gallery-item') && image.src.endsWith(expected) && image.complete && image.naturalWidth > 0), filename);
    selectedSamples.add(filename);
  }
  assert.equal(selectedSamples.size, 6);
  await page.screenshot({ path: path.join(__dirname, 'playground-photo.png'), fullPage: true });
  const localeChecks = ['ko-KR'];
  for (const locale of ['ja-JP', 'de-DE']) {
    const localized = await browser.newPage({ locale, viewport: { width: 1440, height: 1100 } });
    await localized.goto('http://127.0.0.1:7863/', { waitUntil: 'networkidle' });
    assert((await localized.locator('body').innerText()).includes('Drop Image Here'), locale);
    localeChecks.push(locale);
    await localized.close();
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: path.join(__dirname, 'playground-mobile.png'), fullPage: true });
  const mobile = await page.evaluate(() => ({ width: innerWidth, contentWidth: document.documentElement.scrollWidth }));
  assert(mobile.contentWidth <= mobile.width + 1, 'Mobile viewport must not overflow horizontally');
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({
    title: await page.title(),
    hasKoreanText: /[\uac00-\ud7af]/.test(text),
    has2048: text.includes('2048'),
    controls: await page.getByRole('combobox').count(),
    radios: await page.getByRole('radio').allTextContents(),
    locale: await page.evaluate(() => navigator.language),
    checkedPresets, decisionTypes: ['choice', 'score', 'noul'], samplesLoaded: 6,
    localeChecks, mobile, inferenceCalls: 0,
    errors,
    scope: 'UI rendering only; model loading and inference disabled',
  }));
  await browser.close();
})().catch(async error => {
  console.error(JSON.stringify({ phase, error: error.stack }));
  if (page) console.error(JSON.stringify(await page.evaluate(() => ({
    inputs: [...document.querySelectorAll('textarea')].map(input => input.value.slice(0, 180)),
    radios: [...document.querySelectorAll('input[type=radio]')].map(input => ({ value: input.value, checked: input.checked, aria: input.getAttribute('aria-checked') })),
  }))));
  if (browser) await browser.close();
  process.exitCode = 1;
});
