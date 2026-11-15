// End-to-end browser test of the web console against a running server.
//   uvicorn attackmap.api:app --port 8000 &
//   node tests/e2e/ui_test.js reports/screenshots
const { chromium, devices } = require('playwright');
const fs = require('fs');

const BASE = process.env.ATTACKMAP_URL || 'http://localhost:8000/';
const OUT = process.argv[2] || 'reports/screenshots';
const T = { timeout: 180000 };

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const errors = [];
  const results = [];
  const check = (name, ok, extra = '') => results.push(`${ok ? 'PASS' : 'FAIL'}  ${name} ${extra}`);

  for (const scheme of ['dark', 'light']) {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 2, colorScheme: scheme });
    const page = await ctx.newPage();
    page.on('pageerror', e => errors.push(`${scheme} pageerror: ${e.message}`));
    page.on('console', m => { if (m.type() === 'error') errors.push(`${scheme} console: ${m.text()}`); });
    await page.goto(BASE, { waitUntil: 'networkidle' });
    await page.waitForFunction(() => document.querySelector('#status').textContent.includes('models'));
    check(`[${scheme}] status chip`, true, (await page.textContent('#status')).trim());

    // alert triage via preset
    await page.click('#presets button:has-text("LSASS dump")');
    await page.waitForSelector('.hero .tname', T);
    const top = (await page.textContent('.hero .tid')).trim();
    check(`[${scheme}] LSASS preset -> T1003`, top === 'T1003', top);
    check(`[${scheme}] 5 ranked rows`, (await page.$$('.rank tr')).length === 5);
    check(`[${scheme}] 15 kill-chain cells`, (await page.$$('.chain div')).length === 15);
    check(`[${scheme}] evidence terms`, (await page.$$('.term')).length > 0);
    check(`[${scheme}] IOC card visible`, await page.isVisible('#iocCard'));
    check(`[${scheme}] sub-techniques`, (await page.$$('.hero .sub')).length > 0,
      (await page.textContent('.hero .subs')).replace(/\s+/g, ' ').trim());
    await page.waitForTimeout(900);
    await page.screenshot({ path: `${OUT}/alert-${scheme}.png`, fullPage: true });
    if (scheme === 'light') { await ctx.close(); continue; }

    // typed alert + keyboard shortcut
    await page.fill('#alertText', 'Event 4698 scheduled task created: schtasks /create /tn Updater /tr C:\\ProgramData\\svc.exe /sc minute');
    await page.press('#alertText', 'Control+Enter');
    await page.waitForFunction(() => document.querySelector('.hero .tid')?.textContent.trim() === 'T1053', null, T);
    check('[dark] Ctrl+Enter typed alert -> T1053', true);

    // random held-out sample shows ground truth
    await page.click('#presets button:has-text("random")');
    await page.waitForFunction(() => document.querySelector('.hero')?.textContent.includes('Ground truth'), null, T);
    check('[dark] random test sample shows ground truth', true);

    // CTI report
    await page.click('#tabs button[data-v="report"]');
    await page.click('#repSample');
    await page.click('#repBtn');
    await page.waitForSelector('#repOut .kpis', T);
    const nTech = (await page.$$('#repOut .techitem')).length;
    const hits = (await page.$$('#repOut .s.hit')).length;
    check('[dark] CTI report >= 6 techniques', nTech >= 6, `${nTech} techniques, ${hits} highlighted sentences`);
    await page.waitForTimeout(500);
    await page.screenshot({ path: `${OUT}/report-dark.png`, fullPage: true });

    // batch & matrix
    await page.click('#tabs button[data-v="batch"]');
    await page.waitForSelector('.mcell');
    await page.click('#batchSample');
    await page.waitForFunction(() => document.querySelector('#batchText').value.split('\n').length >= 20);
    await page.click('#batchBtn');
    await page.waitForSelector('#batchTable tr:nth-child(5)', T);
    const heat = await page.$$eval('.mcell', els => els.filter(e => /\bh[1-4]\b/.test(e.className)).length);
    check('[dark] batch lights matrix cells', heat > 0, `${heat} cells, ${await page.textContent('#mxMeta')}`);
    const [dl] = await Promise.all([page.waitForEvent('download'), page.click('#navBtn')]);
    const layer = JSON.parse(fs.readFileSync(await dl.path(), 'utf8'));
    check('[dark] Navigator layer download', layer.domain === 'enterprise-attack' && layer.techniques.length > 0,
      `${layer.techniques.length} techniques`);
    await page.screenshot({ path: `${OUT}/batch-dark.png`, fullPage: true });

    // benchmarks
    await page.click('#tabs button[data-v="bench"]');
    await page.waitForSelector('#benchOut table.data');
    check('[dark] benchmark table', (await page.$$('#benchOut table.data tr')).length >= 4);
    await page.screenshot({ path: `${OUT}/bench-dark.png`, fullPage: true });

    // theme toggle
    await page.click('#theme');
    check('[dark] theme toggle -> light', (await page.getAttribute('html', 'data-theme')) === 'light');
    await ctx.close();
  }

  // phone
  const m = await browser.newContext({ ...devices['iPhone 13'] });
  const mp = await m.newPage();
  mp.on('pageerror', e => errors.push(`mobile pageerror: ${e.message}`));
  await mp.goto(BASE, { waitUntil: 'networkidle' });
  await mp.click('#presets button:has-text("Phishing")');
  await mp.waitForSelector('.hero .tname', T);
  const overflow = await mp.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  check('[mobile] no horizontal page scroll', overflow <= 1, `overflow=${overflow}px`);
  const mt = (await mp.textContent('.hero .tid')).trim();
  check('[mobile] phishing preset -> T1566', mt === 'T1566', mt);
  await mp.screenshot({ path: `${OUT}/alert-mobile.png`, fullPage: true });
  await browser.close();

  console.log(results.join('\n'));
  console.log(errors.length ? 'JS ERRORS:\n' + errors.join('\n') : 'no JS errors');
  process.exit(results.some(r => r.startsWith('FAIL')) || errors.length ? 1 : 0);
})().catch(e => { console.error('TEST CRASH', e); process.exit(1); });
