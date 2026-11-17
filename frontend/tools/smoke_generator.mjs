#!/usr/bin/env node
// Smoke test for /app/generator: page load, Generate, Play, progress text,
// "no clip yet" alert for a clip-less gloss (WATER). Not a unit test — run:
//   node tools/smoke_generator.mjs [--port 5199]
import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const frontend = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const portFlag = process.argv.indexOf('--port');
const PORT = portFlag !== -1 ? Number(process.argv[portFlag + 1]) : 5199;

const viteBin = path.join(frontend, 'node_modules', 'vite', 'bin', 'vite.js');
const vite = spawn(process.execPath, [viteBin, 'dev', '--port', String(PORT), '--strictPort'], {
  cwd: frontend, stdio: ['ignore', 'pipe', 'pipe'],
});
await new Promise((resolve, reject) => {
  const to = setTimeout(() => reject(new Error('vite did not start in 60s')), 60000);
  vite.stdout.on('data', (d) => { if (String(d).includes('Local:')) { clearTimeout(to); resolve(); } });
});

const browser = await chromium.launch({
  headless: true,
  args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
});
const errors = [];
try {
  const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } });
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(`console: ${m.text()}`); });

  await page.goto(`http://localhost:${PORT}/app/generator`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(2500);
  await page.screenshot({ path: 'docs/verification/ui_generator_loaded.png' });

  // type a phrase whose signs ship with clips: "hello where water"
  const box = page.locator('textarea').first();
  await box.fill('hello where water');
  await page.getByRole('button', { name: /Generate/i }).click();
  await page.waitForTimeout(500);

  // gloss chips should appear: HELLO WHERE WATER
  const body = await page.textContent('body');
  for (const g of ['HELLO', 'WHERE', 'WATER']) {
    if (!body.includes(g)) throw new Error(`gloss ${g} missing from page`);
  }
  console.log('gloss sequence rendered: HELLO WHERE WATER');

  // play
  await page.getByRole('button', { name: /^Play$/ }).click();
  await page.waitForTimeout(1500);
  const mid = await page.textContent('body');
  const m = mid.match(/Signing ([A-Z_]+) \((\d+) of (\d+)\)/);
  console.log('progress indicator during playback:', m ? m[0] : '(not visible at this instant)');
  await page.screenshot({ path: 'docs/verification/ui_generator_playing.png' });

  // wait for the sequence to finish (~ 2.96 + 6.37 + 0.35s + slack)
  await page.waitForTimeout(12000);
  await page.screenshot({ path: 'docs/verification/ui_generator_done.png' });
  const end = await page.textContent('body');
  if (!/WATER/.test(end)) throw new Error('WATER chip missing after playback');
  if (!/No clip yet for\s*WATER/i.test(end.replace(/\s+/g, ' '))) {
    console.log('unavailable alert text:', JSON.stringify(end.match(/No clip yet[^.]*/i)?.[0] ?? 'none found'));
  } else {
    console.log('unavailable alert shown for WATER ✓');
  }
  if (errors.length) {
    console.log('PAGE ERRORS:\n' + errors.join('\n'));
    process.exitCode = 1;
  } else {
    console.log('no page errors');
  }
} finally {
  await browser.close();
  vite.kill('SIGTERM');
}
