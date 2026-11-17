#!/usr/bin/env node
// Render start/mid/end frames of every sign clip with the real GLB, via the
// DEV-only /dev/verify-clips page + headless Chromium (SwiftShader).
// Usage: node tools/render_verification.mjs [--port 5199]
import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
import { mkdirSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const frontend = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = path.resolve(frontend, '..');
const outDir = path.join(root, 'docs', 'verification');
mkdirSync(outDir, { recursive: true });

const portFlag = process.argv.indexOf('--port');
const PORT = portFlag !== -1 ? Number(process.argv[portFlag + 1]) : 5199;
const GLOSSES = ['hello', 'thank_you', 'where', 'please', 'food', 'name', 'how'];
const FRACS = [0.1, 0.5, 0.9];
const FRAC_NAME = { 0.1: 'start', 0.5: 'mid', 0.9: 'end' };

// start vite dev server
const viteBin = path.join(frontend, 'node_modules', 'vite', 'bin', 'vite.js');
const vite = spawn(process.execPath, [viteBin, 'dev', '--port', String(PORT), '--strictPort'], {
  cwd: frontend, stdio: ['ignore', 'pipe', 'pipe'],
});
await new Promise((resolve, reject) => {
  const to = setTimeout(() => reject(new Error('vite dev server did not start in 60s')), 60000);
  vite.stdout.on('data', (d) => { if (String(d).includes('Local:')) { clearTimeout(to); resolve(); } });
  vite.stderr.on('data', (d) => process.stderr.write(d));
});
console.log(`vite dev on :${PORT}`);

const browser = await chromium.launch({
  headless: true,
  args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--disable-gpu-sandbox'],
});
try {
  for (const gloss of GLOSSES) {
    for (const frac of FRACS) {
      const page = await browser.newPage({ viewport: { width: 640, height: 700 } });
      await page.goto(`http://localhost:${PORT}/dev/verify-clips?gloss=${gloss}&frac=${frac}`, { waitUntil: 'networkidle' });
      await page.waitForFunction(() => window.__verifyReady === true, null, { timeout: 60000 });
      const canvas = page.locator('#verify-canvas');
      const file = path.join(outDir, `${gloss}_${FRAC_NAME[frac]}.png`);
      await canvas.screenshot({ path: file });
      console.log('saved', file);
      await page.close();
    }
  }
} finally {
  await browser.close();
  vite.kill('SIGTERM');
}
