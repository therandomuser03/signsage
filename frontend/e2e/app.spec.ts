import { test, expect } from '@playwright/test';

test('landing page has both direction cards', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveTitle(/SignSage/);
  await expect(page.getByRole('link', { name: 'SignSage' })).toBeVisible();
  await expect(page.getByRole('button', { name: /Sign to Text/i })).toBeVisible();
  await expect(page.getByRole('button', { name: /Text to Sign/i })).toBeVisible();
});

test('sign card navigates to workspace', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: /Sign to Text/i }).click();
  await expect(page.getByRole('tab', { name: 'Sign to text' })).toBeVisible();
});

test('text card navigates to workspace', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: /Text to Sign/i }).click();
  await expect(page.getByRole('tab', { name: 'Text to sign' })).toBeVisible();
});

test('back button returns to landing', async ({ page }) => {
  await page.goto('/#/translate/sign');
  await page.getByRole('button', { name: /Back/i }).click();
  await expect(page.getByRole('button', { name: /Sign to Text/i })).toBeVisible();
});
