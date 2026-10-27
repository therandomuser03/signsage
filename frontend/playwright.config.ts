import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: 0,
  workers: 1,
  reporter: 'line',
  webServer: [
    {
      command: 'cd ../backend && SIGNSAGE_IMAGE_ALPHABET_BUNDLE_DIR=../research/image_classification/artifacts/alphabet SIGNSAGE_IMAGE_WORD_BUNDLE_DIR=../research/image_classification/artifacts/words SIGNSAGE_MODEL_PROMOTED=false SIGNSAGE_ALLOW_UNPROMOTED_DEV=false .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000',
      url: 'http://localhost:8000/api/v1/health',
      reuseExistingServer: !process.env.CI,
      timeout: 15000,
    },
    {
      command: 'npm run preview',
      url: 'http://localhost:4173',
      reuseExistingServer: !process.env.CI,
      timeout: 30000,
    },
  ],
  use: {
    baseURL: 'http://localhost:4173',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    actionTimeout: 10000,
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});
